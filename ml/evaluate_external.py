"""
Zero-shot test of the trained models on an external dataset, plus a train/val/test gap check.

    python -m ml.evaluate_external --csv ../test_db/biofouling_historical_test_45days.csv --out ../test_db

The external file has columns hour, temp_in, temp_out, flow, dp, heat_duty_mw, u_overall; it has
no true Rf and no cleaning log. Nothing is refitted. Results: <out>/results_45days.md, .json, .png.
"""

import argparse
import json
import math
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import ml  # noqa: F401
import torch

from core import config
from ml import baselines as B
from ml import data as D
from ml import evaluate as E
from ml import forecast as F
from ml import physics_torch as pt
from ml import train as T

HORIZONS = (1, 24, 72, 168)
STRIDE_H = 6
JUMP_U = 150.0       # W/m²K rise within one hour taken as a cleaning
JUMP_DP = -0.5       # kPa fall within one hour taken as a cleaning
_STATE_AT_ORIGIN = F.state_at_origin


# -----------------------------------------------------------------------------
# Loading and mapping
# -----------------------------------------------------------------------------

def load_external(path: Path):
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "timestamp": raw["timestamp"],
        "time_hours": raw["hour"].astype(int),
        "T_cool_in_meas": raw["temp_in"].astype(float),
        "mass_flow_cool_meas": raw["flow"].astype(float),
        "U_overall_meas": raw["u_overall"].astype(float),
        "delta_P_meas_kPa": raw["dp"].astype(float),
        "Rf_fouling_resistance": np.nan,   # not available in the external file
    })
    du, ddp = df["U_overall_meas"].diff(), df["delta_P_meas_kPa"].diff()
    cleanings = df.index[(du > JUMP_U) & (ddp < JUMP_DP)].tolist()
    df["cleaning_event"] = 0
    df["cleaning_type"] = "None"
    df.loc[cleanings, "cleaning_event"] = 1
    df.loc[cleanings, "cleaning_type"] = "Mechanical_Balls"   # type unknown; assumed mechanical
    # dP flat-lined after the cleaning: a constant reading repeated for the rest of the file
    stuck = pd.Series(False, index=df.index)
    for c in cleanings:
        tail = df["delta_P_meas_kPa"].iloc[c:]
        if tail.nunique() == 1:
            stuck.iloc[c:] = True
    df["dp_valid"] = ~stuck
    return raw, df, [int(df["time_hours"].iloc[c]) for c in cleanings]


def u_only_state(df, origin, block_h=D.BURN_IN_HOURS):
    """State from U alone (used when dP is flagged invalid)."""
    st = _STATE_AT_ORIGIN(df, origin, block_h)
    hist = df[df["time_hours"] <= origin]
    last_clean = max([h for h, _ in F._log_from(hist, origin)], default=-1)
    blk = hist[hist["time_hours"] >= max(last_clean, origin - block_h + 1, 0)]
    d = pt.delta_from_u(torch.tensor(blk["U_overall_meas"].to_numpy(float), dtype=torch.float64)).numpy()
    st["delta"] = float(np.median(d))
    st["delta_se"] = float(1.2533 * np.std(d) / math.sqrt(len(d))) if len(d) > 1 else 5e-6
    return st


# -----------------------------------------------------------------------------
# Forecast evaluation on observables
# -----------------------------------------------------------------------------

def evaluate(df, regressors, exog, dp_as_missing=False):
    last = int(df["time_hours"].max())
    origins = list(range(D.BURN_IN_HOURS + 12, last - max(HORIZONS) + 1, STRIDE_H))
    H = max(HORIZONS)
    idx = df.set_index("time_hours")
    preds = {}
    patch = mock.patch.object(F, "state_at_origin", u_only_state if dp_as_missing else _STATE_AT_ORIGIN)
    with patch:
        for kind in ("ode", "ude"):
            states = [F.state_at_origin(df, o) for o in origins]
            rows = [F.future_inputs(df, o, H, exog, 0.0, F.dataset_planned_cleanings(df, o, H)) for o in origins]
            hours, Tm, flow, codes = (np.stack([r[i] for r in rows]) for i in range(4))
            traj = F.rollout_trajectories(F.load_ensemble(kind), states, hours, Tm, flow, codes)
            rf, u, dp = F.observables(traj, flow[:, None, :])
            preds[kind] = {"rf": rf, "u": u, "dp": dp}
    # persistence of the observables themselves (last 24 h medians)
    pu = np.array([idx.loc[o - 23:o, "U_overall_meas"].median() for o in origins])
    pdp = np.array([idx.loc[o - 23:o, "delta_P_meas_kPa"].median() for o in origins])
    preds["persistence"] = {"u": np.repeat(pu[:, None], H, 1), "dp": np.repeat(pdp[:, None], H, 1)}
    for kind, reg in regressors.items():
        rf_r = np.zeros((len(origins), H))
        for h in HORIZONS:
            rf_r[:, h - 1] = reg.predict(df, origins, h)
        _, u, dp = F.observables(rf_r * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK, np.full(H, config.COOLING_WATER_MASS_FLOW_KG_S))
        preds[kind] = {"u": u, "dp": dp}

    out = {}
    for model, p in preds.items():
        out[model] = {}
        for h in HORIZONS:
            tgt = [o + h for o in origins]
            row = {}
            for q, col in (("u", "U_overall_meas"), ("dp", "delta_P_meas_kPa")):
                arr = p[q]
                point = arr[:, :, h - 1].mean(axis=1) if arr.ndim == 3 else arr[:, h - 1]
                y = idx.loc[tgt, col].to_numpy(float)
                ok = np.ones(len(tgt), bool) if q == "u" else idx.loc[tgt, "dp_valid"].to_numpy(bool)
                r = E.regression_metrics(y[ok], point[ok])
                r["mape_pct"] = float(np.mean(np.abs(point[ok] - y[ok]) / np.abs(y[ok])) * 100)
                if arr.ndim == 3:
                    r["coverage_90"] = E.interval_coverage(y[ok], np.quantile(arr[ok, :, h - 1], 0.05, axis=1),
                                                           np.quantile(arr[ok, :, h - 1], 0.95, axis=1))
                row[q] = r
            out[model][str(h)] = row
    return out, origins, preds


def growth_rate_check(df, cleaning_hours):
    """Observed vs model-implied Rf growth per day (from U), per segment, 168 h free-run from each segment start."""
    res = []
    bounds = [0] + [c + 1 for c in cleaning_hours] + [int(df["time_hours"].max()) + 1]
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = df[(df["time_hours"] >= a + 24) & (df["time_hours"] < b)]
        rf_obs = 1.0 / seg["U_overall_meas"] - 1.0 / config.U_CLEAN_DESIGN_W_M2K
        slope_obs = float(np.polyfit(seg["time_hours"], rf_obs, 1)[0]) * 24
        o = a + 24
        H = min(b - o - 1, 24 * 21)
        row = {"segment_h": [a, b - 1], "observed_rf_per_day": slope_obs}
        for kind in ("ode", "ude"):
            f = F.forecast_at(df, o, H, kind, [], "oracle", 0.0, with_truth=False)
            rf = np.array(f["Rf"]["mean"])
            row[f"{kind}_rf_per_day"] = float(np.polyfit(np.arange(H), rf, 1)[0]) * 24
        res.append(row)
    return res


def split_gap():
    """Observation loss of ODE and UDE on nominal train / validation / test windows."""
    cfg = T.TrainConfig()
    df = D.load("nominal")
    L = cfg.window_days * 24
    out = {}
    for split, rng in D.split_ranges(df).items():
        w = D.make_windows(df, D.window_starts(df, rng, L), L)
        out[split] = {"windows": len(w.starts)}
        for kind in ("ode", "ude"):
            vals = []
            for model in F.load_ensemble(kind):
                with torch.no_grad():
                    vals.append(float(T.huber_loss(w, model, cfg)[1]))
            out[split][kind] = float(np.mean(vals))
    return out


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def plot(df, origins, preds, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    idx = df.set_index("time_hours")
    axes[0].plot(idx.index, idx["U_overall_meas"], color="0.6", lw=0.8, label="measured U")
    axes[1].plot(idx.index, idx["delta_P_meas_kPa"], color="0.6", lw=0.8, label="measured dP")
    for k, o in enumerate(origins[::24]):
        i = origins.index(o)
        hrs = np.arange(o + 1, o + 1 + max(HORIZONS))
        for kind, color in (("ude", "#0072B2"), ("ode", "#E69F00")):
            axes[0].plot(hrs, preds[kind]["u"][i].mean(axis=0), color=color, lw=1.4, label=f"{kind.upper()} 168 h forecast" if k == 0 else None)
            axes[1].plot(hrs, preds[kind]["dp"][i].mean(axis=0), color=color, lw=1.4, label=f"{kind.upper()} 168 h forecast" if k == 0 else None)
    for ax in axes:
        for h in df.loc[df["cleaning_event"] == 1, "time_hours"]:
            ax.axvline(h, color="k", ls=":", lw=1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("U (W/m²K)")
    axes[1].set_ylabel("dP (kPa)")
    axes[1].set_xlabel("hour")
    axes[0].set_title("External 45-day dataset: forecasts from origins every 6 days (oracle T and flow); dotted = inferred cleaning")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    raw, df, cleanings = load_external(args.csv)
    regressors = {k: B.HorizonRegressors(k, HORIZONS).fit(D.load("nominal")) for k in ("ridge", "gbm")}
    results = {}
    for name, exog, missing in (("oracle", "oracle", False), ("oracle_dp_missing", "oracle", True), ("climatology", "climatology", False)):
        res, origins, preds = evaluate(df, regressors, exog, missing)
        results[name] = res
        if name == "oracle":
            main_origins, main_preds = origins, preds
    payload = {
        "dataset": str(args.csv.name), "rows": len(df), "hours": [int(df.time_hours.min()), int(df.time_hours.max())],
        "inferred_cleanings_h": cleanings, "dp_stuck_hours": int((~df["dp_valid"]).sum()),
        "origins": [main_origins[0], main_origins[-1], STRIDE_H], "n_origins": len(main_origins),
        "results": results, "growth_rate": growth_rate_check(df, cleanings), "nominal_split_loss": split_gap(),
        "seconds": time.time() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results_45days.json").write_text(json.dumps(payload, indent=1, default=lambda x: None))
    plotted = plot(df, main_origins, main_preds, args.out / "results_45days.png")
    print(json.dumps({k: payload[k] for k in ("inferred_cleanings_h", "dp_stuck_hours", "n_origins", "growth_rate", "nominal_split_loss")}, indent=1))
    print("plot" if plotted else "no plot", f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
