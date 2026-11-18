"""
Leak-free evaluation of persistence, Ridge, GBM, plain ODE and the UDE ensemble.

    python -m ml.evaluate [--stride 24]

Origins every `stride` hours; horizons 1, 24, 72, 168 h. Main setting: climatology future
inputs (seasonal T, nominal flow); ablation: oracle future T and flow (ODE and UDE only).
Planned cleanings inside the horizon are known to every model.
Datasets: nominal-test (cycles 10-12), and the three stress / mismatch files (final
evaluation only; all models are trained on nominal-train). Writes ml/artifacts/metrics.json
and plots (if matplotlib is installed).
"""

import argparse
import json
import math
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

import ml  # noqa: F401
import config
from ml import baselines as B
from ml import data as D
from ml import forecast as F
from ml.device import ARTIFACTS_DIR

HORIZONS = (1, 24, 72, 168)
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
EVAL_SETS = {
    "nominal_test": "nominal",
    "stress_no_clean": "stress_no_clean",
    "stress_long_interval": "stress_long_interval",
    "mismatch_stress": "mismatch_stress",
}
MODEL_ORDER = ("persistence", "ridge", "gbm", "ode", "ude")
TTL_MAX_LEAD_H = 60 * 24


# -----------------------------------------------------------------------------
# Metric helpers (tested on toy cases)
# -----------------------------------------------------------------------------

def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(math.sqrt(np.mean(err ** 2))),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), "n": int(len(y))}


def interval_coverage(y_true, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, float) for a in (y_true, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def first_crossing_hour(values: Sequence[float], hours: Sequence[int], limit: float):
    v = np.asarray(values, float)
    idx = np.flatnonzero(v > limit)
    return int(np.asarray(hours)[idx[0]]) if len(idx) else None


def linear_trend_crossing(hours: Sequence[float], values: Sequence[float], limit: float, origin: float):
    """Least-squares line through (hours, values); hour at which it reaches `limit` (None if never ahead)."""
    h, v = np.asarray(hours, float), np.asarray(values, float)
    if len(h) < 2:
        return None
    slope, intercept = np.polyfit(h, v, 1)
    if slope <= 0:
        return None
    t = (limit - intercept) / slope
    return float(max(t, origin))


# -----------------------------------------------------------------------------
# Batched forecasts for many origins
# -----------------------------------------------------------------------------

def eval_origins(df: pd.DataFrame, name: str, stride: int) -> List[int]:
    lo, hi = D.split_ranges(df)["test"] if name == "nominal_test" else (0, len(df))
    return list(range(lo + 24, hi - max(HORIZONS), stride))


def physics_forecasts(df, origins, kind, exog, horizon):
    states = [F.state_at_origin(df, o) for o in origins]
    rows = [F.future_inputs(df, o, horizon, exog, 0.0, F.dataset_planned_cleanings(df, o, horizon)) for o in origins]
    hours, T, flow, codes = (np.stack([r[i] for r in rows]) for i in range(4))
    traj = F.rollout_trajectories(F.load_ensemble(kind), states, hours, T, flow, codes)   # [N, M, H]
    rf, u, dp = F.observables(traj, flow[:, None, :])
    return {"rf": rf, "u": u, "dp": dp, "hours": hours}


def evaluate_dataset(df, name, stride, regressors) -> dict:
    origins = eval_origins(df, name, stride)
    H = max(HORIZONS)
    idx = df.set_index("time_hours")
    truth = {k: idx[c] for k, c in (("rf", "Rf_fouling_resistance"), ("u", "U_overall_meas"), ("dp", "delta_P_meas_kPa"))}
    nominal_flow = np.full(H, config.COOLING_WATER_MASS_FLOW_KG_S)

    preds = {}  # model -> {"rf","u","dp": [N, H] or [N, M, H]}
    for kind in ("ode", "ude"):
        preds[kind] = physics_forecasts(df, origins, kind, "climatology", H)
        preds[f"{kind}_oracle_exog"] = physics_forecasts(df, origins, kind, "oracle", H)
    rf_p = np.array([F.persistence_rf(df, o) for o in origins])[:, None].repeat(H, axis=1)
    preds["persistence"] = dict(zip(("rf", "u", "dp"), F.observables(rf_p * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK, nominal_flow)))
    for kind, reg in regressors.items():
        rf_r = np.full((len(origins), H), np.nan)
        for h in HORIZONS:
            rf_r[:, h - 1] = reg.predict(df, origins, h)
        rf_r = np.where(np.isnan(rf_r), 0.0, rf_r)
        preds[kind] = dict(zip(("rf", "u", "dp"), F.observables(rf_r * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK, nominal_flow)))

    result = {"origins": len(origins), "origin_stride_h": stride, "first_origin": origins[0], "last_origin": origins[-1],
              "metrics": {}, "coverage_90": {}}
    for model, p in preds.items():
        result["metrics"][model] = {}
        for h in HORIZONS:
            target_hours = [o + h for o in origins]
            row = {}
            for q in ("rf", "u", "dp"):
                arr = p[q]
                point = arr[:, :, h - 1].mean(axis=1) if arr.ndim == 3 else arr[:, h - 1]
                row[q] = regression_metrics(truth[q].loc[target_hours].to_numpy(float), point)
            result["metrics"][model][str(h)] = row
        if p["rf"].ndim == 3:
            result["coverage_90"][model] = {
                str(h): interval_coverage(truth["rf"].loc[[o + h for o in origins]].to_numpy(float),
                                          np.quantile(p["rf"][:, :, h - 1], 0.05, axis=1),
                                          np.quantile(p["rf"][:, :, h - 1], 0.95, axis=1))
                for h in HORIZONS}
    return result, preds, origins


# -----------------------------------------------------------------------------
# Time to limit
# -----------------------------------------------------------------------------

def time_to_limit(df: pd.DataFrame, stride: int) -> dict:
    """From origins before the first crossing, predict the crossing hour; compare with a 7-day linear trend."""
    out = {}
    hours_all = df["time_hours"].to_numpy()
    for limit_name, col, limit, traj_key in (("rf", "Rf_fouling_resistance", config.CRITICAL_RF_M2K_W, "rf"),
                                             ("dp", "delta_P_true_kPa", config.CRITICAL_DELTA_P_KPA, "dp")):
        actual = first_crossing_hour(df[col].to_numpy(float), hours_all, limit)
        if actual is None:
            out[limit_name] = {"actual_first_crossing_h": None, "note": "limit never crossed in this dataset"}
            continue
        origins = [o for o in range(max(24, actual - TTL_MAX_LEAD_H), actual - 23, stride)]
        horizon = actual - origins[0] + 30 * 24
        rows = {}
        for kind in ("ode", "ude"):
            p = physics_forecasts(df, origins, kind, "climatology", horizon)
            preds = []
            for i, o in enumerate(origins):
                c = F._first_crossing(p[traj_key][i], limit, p["hours"][i])
                filled = np.where(np.isnan(c), np.inf, c)
                med, lo, hi = (float(np.quantile(filled, q)) for q in (0.5, 0.05, 0.95))
                preds.append((med, lo, hi))
            rows[kind] = preds
        trend = []
        for o in origins:
            hist = df[(df["time_hours"] <= o) & (df["time_hours"] > o - 7 * 24)]
            last_clean = max([h for h, _ in F._log_from(df, o)], default=-1)
            hist = hist[hist["time_hours"] > last_clean]
            if limit_name == "rf":
                vals = 1.0 / hist["U_overall_meas"].to_numpy(float) - 1.0 / config.U_CLEAN_DESIGN_W_M2K
            else:
                vals = hist["delta_P_meas_kPa"].to_numpy(float)
            t = linear_trend_crossing(hist["time_hours"].to_numpy(float), vals, limit, o)
            trend.append(np.inf if t is None else t)

        def summarise(pred_med, lo=None, hi=None):
            pred_med = np.asarray(pred_med, float)
            leads = np.array([actual - o for o in origins], float)
            err = np.where(np.isfinite(pred_med), pred_med - actual, np.nan)
            res = {"n_origins": len(origins), "never_predicted": int(np.sum(~np.isfinite(pred_med)))}
            for name, sel in (("lead_1_7d", leads <= 7 * 24), ("lead_7_30d", (leads > 7 * 24) & (leads <= 30 * 24)),
                              ("lead_30_60d", leads > 30 * 24), ("all", np.ones_like(leads, bool))):
                e = err[sel]
                e = e[~np.isnan(e)]
                res[name] = {"mae_h": float(np.mean(np.abs(e))) if len(e) else None,
                             "bias_h": float(np.mean(e)) if len(e) else None, "n": int(sel.sum())}
            if lo is not None:
                res["interval_90_coverage"] = float(np.mean((np.asarray(lo) <= actual) & (actual <= np.asarray(hi))))
            return res

        out[limit_name] = {
            "actual_first_crossing_h": int(actual),
            "origins": [origins[0], origins[-1], stride],
            "ude": summarise(*zip(*rows["ude"])),
            "ode": summarise(*zip(*rows["ode"])),
            "linear_trend_7d": summarise(trend),
        }
    return out


# -----------------------------------------------------------------------------
# Verdict (plain statements of wins and losses, from the numbers only)
# -----------------------------------------------------------------------------

def build_verdict(results: dict) -> List[str]:
    lines = []
    others = ("ode", "persistence", "ridge", "gbm")
    for name, res in results.items():
        m = res["metrics"]
        rmse = lambda k, h: m[k][str(h)]["rf"]["rmse"]
        wins = {o: [h for h in HORIZONS if rmse("ude", h) < rmse(o, h)] for o in others}
        parts = []
        for o in others:
            w = wins[o]
            if len(w) == len(HORIZONS):
                parts.append(f"beats {o} at every horizon")
            elif not w:
                parts.append(f"loses to {o} at every horizon")
            else:
                parts.append(f"beats {o} at {', '.join(str(h) + ' h' for h in w)} only")
        best168 = min(MODEL_ORDER, key=lambda k: rmse(k, 168))
        lines.append(f"{name}: UDE Rf RMSE {', '.join(parts)}; best at 168 h: {best168} "
                     f"(UDE {rmse('ude', 168):.2e} vs ODE {rmse('ode', 168):.2e}, persistence {rmse('persistence', 168):.2e} m²K/W).")
        cov = res.get("coverage_90", {}).get("ude")
        if cov:
            lines.append(f"{name}: UDE 90% band coverage " + ", ".join(f"{h} h {cov[str(h)]:.2f}" for h in HORIZONS)
                         + (" (under-covers: bands too narrow)" if min(cov.values()) < 0.8 else ""))
        for lim, r in (res.get("time_to_limit") or {}).items():
            if not r.get("actual_first_crossing_h"):
                continue
            u, t = r["ude"]["all"]["mae_h"], r["linear_trend_7d"]["all"]["mae_h"]
            o = r["ode"]["all"]["mae_h"]
            if u is not None and t is not None:
                lines.append(f"{name}: time to {lim} limit, mean abs error UDE {u:.0f} h, ODE {o:.0f} h, 7-day linear trend {t:.0f} h"
                             f" ({'UDE better than' if u < t else 'UDE worse than'} the trend).")
    return lines


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def make_plots(all_preds, all_results, slices):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib not installed: plots skipped"]
    written = []
    for name, (df, preds, origins) in all_preds.items():
        i = len(origins) // 2
        o = origins[i]
        hours = np.arange(o + 1, o + 1 + max(HORIZONS))
        fig, ax = plt.subplots(figsize=(8, 4))
        truth = df.set_index("time_hours")["Rf_fouling_resistance"].loc[hours]
        ax.plot(hours, truth * 1e4, "k-", lw=1.5, label="true Rf (synthetic)")
        for kind, color in (("ude", "#0072B2"), ("ode", "#E69F00")):
            r = preds[kind]["rf"][i] * 1e4
            ax.fill_between(hours, np.quantile(r, 0.05, axis=0), np.quantile(r, 0.95, axis=0), color=color, alpha=0.2)
            ax.plot(hours, r.mean(axis=0), color=color, label=f"{kind.upper()} mean, 5-95%")
        ax.plot(hours, preds["persistence"]["rf"][i] * 1e4, ":", color="#009E73", label="persistence")
        ax.axhline(config.CRITICAL_RF_M2K_W * 1e4, color="#D55E00", ls="--", label="Rf limit")
        ax.set(xlabel="hour", ylabel="Rf (1e-4 m²K/W)", title=f"SYNTHETIC - {name}: 168 h forecast from hour {o}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = ARTIFACTS_DIR / f"forecast_{name}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p.name)

    fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 3.5), sharey=False)
    markers = {"persistence": "o", "ridge": "s", "gbm": "^", "ode": "D", "ude": "*"}
    for ax, (name, res) in zip(axes, all_results.items()):
        for m in MODEL_ORDER:
            ax.plot(HORIZONS, [res["metrics"][m][str(h)]["rf"]["rmse"] * 1e4 for h in HORIZONS], marker=markers[m], label=m)
        ax.set(xscale="log", xlabel="horizon (h)", title=name)
    axes[0].set_ylabel("Rf RMSE (1e-4 m²K/W)")
    axes[0].legend(fontsize=8)
    fig.suptitle("SYNTHETIC data - error vs horizon")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "error_vs_horizon.png", dpi=110)
    plt.close(fig)
    written.append("error_vs_horizon.png")

    fig, axes = plt.subplots(1, len(slices["slices"]), figsize=(4 * len(slices["slices"]), 3.2))
    for ax, s in zip(axes, slices["slices"]):
        ax.fill_between(s["x"], s["g_min"], s["g_max"], alpha=0.25, color="#0072B2")
        ax.plot(s["x"], s["g"], color="#0072B2", label="learned g (ensemble)")
        if "generator_equivalent" in s:
            ax.plot(s["x"], s["generator_equivalent"], "k--", label="generator nutrient factor (rescaled)")
        ax.axhline(1.0, color="grey", lw=0.8)
        ax.set(xlabel=s["name"], ylabel="g")
        ax.legend(fontsize=7)
    fig.suptitle("Learned correction g: partial dependence (others at reference)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "correction_slices.png", dpi=110)
    plt.close(fig)
    written.append("correction_slices.png")
    return written


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate all models")
    ap.add_argument("--stride", type=int, default=24, help="origin spacing in hours")
    args = ap.parse_args()
    t0 = time.time()

    nominal = D.load("nominal")
    regressors = {k: B.HorizonRegressors(k, HORIZONS).fit(nominal) for k in ("ridge", "gbm")}
    print(f"baselines fitted [{time.time() - t0:.0f}s]")

    results, all_preds = {}, {}
    for name, ds in EVAL_SETS.items():
        df = D.load(ds)
        res, preds, origins = evaluate_dataset(df, name, args.stride, regressors)
        if name != "nominal_test":
            res["time_to_limit"] = time_to_limit(df, args.stride)
        results[name] = res
        all_preds[name] = (df, preds, origins)
        print(f"{name}: {res['origins']} origins [{time.time() - t0:.0f}s]")

    from ml import train as T
    log = json.loads(T.LOG_PATH.read_text()) if T.LOG_PATH.exists() else {}
    slices = F.correction_slices()
    metrics = {
        "synthetic": True,
        "disclaimer": "Synthetic data; academic demonstration, not for plant use.",
        "horizons_h": list(HORIZONS),
        "origin_stride_h": args.stride,
        "exogenous_main": "climatology (seasonal T from config, nominal flow)",
        "exogenous_ablation": "oracle (measured future T and flow), ODE and UDE only: *_oracle_exog",
        "trained_on": "nominal train cycles 0-7 (hours %d-%d)" % tuple(D.split_ranges(nominal)["train"]),
        "fitted_parameters": {"config": log.get("config_values"), "ode": log.get("ode"),
                              "ude_members": log.get("ude_members")},
        "datasets": results,
        "correction_slices": slices,
        "seconds": time.time() - t0,
    }
    metrics["verdict"] = build_verdict(results)
    metrics["plots"] = make_plots(all_preds, results, slices)
    METRICS_PATH.write_text(json.dumps(metrics, indent=1, allow_nan=False, default=lambda x: None))
    print(f"wrote {METRICS_PATH} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
