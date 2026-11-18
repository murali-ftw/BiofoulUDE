"""
Forecasting with the trained models (CPU inference, float64, deterministic).

Origin semantics: the origin t0 is the last observed hour. The state at the end of t0 is
estimated from observations at hours <= t0 only (no truth). Forecast hours are t0+1 .. t0+H.
Future inputs are either climatology (seasonal T from config + offset, nominal flow) or
"oracle" (the dataset's measured future T and flow + offset). Future U, dP and truth are never
read. Planned cleanings in the horizon are known inputs.

Uncertainty: every ensemble member is rolled out from three initial states,
delta0 + z * se(delta0) with z in (-1.645, 0, +1.645); bands are the 5-95% range over all
trajectories. The ODE (one model) therefore has a band from state uncertainty only;
persistence has no band.
"""

import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import ml  # noqa: F401
import torch

from core import config
from ml import data as D
from ml import physics_torch as pt
from ml.ude import CLEANING_CODES, NONE, FoulingUDE, cum_clean_hours_from_log, hours_since_cleaning_from_log

DT = torch.float64
STATE_Z = (-1.645, 0.0, 1.645)
MODELS = ("ude", "ode", "persistence")
MAX_HORIZON_H = 24 * 120


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_ensemble(kind: str) -> tuple:
    from ml import train as T
    if kind == "ode":
        paths = [T.ODE_PATH]
    elif kind == "ude":
        paths = [T.ude_path(s) for s in T.ENSEMBLE_SEEDS]
    else:
        raise ValueError(f"no ensemble for model {kind!r}")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing model artifacts (run python -m ml.train): {missing}")
    return tuple(T.load_model(p, DT) for p in paths)


# -----------------------------------------------------------------------------
# State at the origin (observations up to t0 only)
# -----------------------------------------------------------------------------

def _log_from(df: pd.DataFrame, upto_hour: int):
    ev = df[(df["cleaning_event"] == 1) & (df["time_hours"] <= upto_hour)]
    return [(int(h), CLEANING_CODES[t]) for h, t in zip(ev["time_hours"], ev["cleaning_type"])]


def state_at_origin(df: pd.DataFrame, origin: int, block_h: int = D.BURN_IN_HOURS) -> Dict[str, float]:
    """delta estimate (and standard error) from the last block_h hours <= origin, after the last cleaning."""
    hist = df[df["time_hours"] <= origin]
    log = _log_from(hist, origin)
    last_clean = max([h for h, _ in log], default=-1)
    lo = max(last_clean, origin - block_h + 1, 0)  # the cleaning hour's record is already post-cleaning
    blk = hist[hist["time_hours"] >= lo]
    u, dp, flow = (blk[c].to_numpy(float) for c in ("U_overall_meas", "delta_P_meas_kPa", "mass_flow_cool_meas"))
    hours = [h for h, _ in log]
    return {
        "delta": D.estimate_delta(u, dp, flow),
        "delta_se": D.estimate_delta_spread(u, dp, flow) if len(u) > 1 else 5e-6,
        "cum": cum_clean_hours_from_log(hours, origin + 1),
        "hsc": hours_since_cleaning_from_log(hours, origin + 1),
    }


def rf_breakdown(df: pd.DataFrame, origin: int, block_h: int = D.BURN_IN_HOURS) -> Dict[str, float]:
    """
    The fouling-resistance calculation used for the model's starting state, step by step, from the
    last block_h hours <= origin (after the last cleaning). Measured values are block medians.
      step 1: Rf_U = 1/U - 1/U_clean            -> thickness_U = k_biofilm * Rf_U
      step 2: thickness_dP from inverting Darcy-Weisbach/Haaland at the measured flow
      result: thickness = mean of the two (the median of two estimates); Rf = thickness / k_biofilm
    """
    hist = df[df["time_hours"] <= origin]
    last_clean = max([h for h, _ in _log_from(hist, origin)], default=-1)
    blk = hist[hist["time_hours"] >= max(last_clean, origin - block_h + 1, 0)]
    u, dp, flow, temp = (blk[c].to_numpy(float) for c in ("U_overall_meas", "delta_P_meas_kPa", "mass_flow_cool_meas", "T_cool_in_meas"))
    d_u = pt.delta_from_u(torch.tensor(u, dtype=DT)).numpy()
    d_p = pt.delta_from_dp(torch.tensor(dp, dtype=DT), torch.tensor(flow, dtype=DT)).numpy()
    thick_u, thick_p = float(np.median(d_u)), float(np.median(d_p))
    thickness = D.estimate_delta(u, dp, flow)
    k = config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK
    return {
        "hour": int(origin), "hours_used": int(len(u)),
        "water_temp_c": float(np.median(temp)), "flow_kg_s": float(np.median(flow)),
        "u_measured": float(np.median(u)), "dp_measured_kpa": float(np.median(dp)),
        "rf_from_u": thick_u / k, "thickness_from_u_um": thick_u * 1e6, "thickness_from_dp_um": thick_p * 1e6,
        "thickness_um": thickness * 1e6, "rf": thickness / k,
    }


def persistence_rf(df: pd.DataFrame, origin: int) -> float:
    """Median of 1/U - 1/U_clean over the last 24 h <= origin (after the last cleaning)."""
    hist = df[df["time_hours"] <= origin]
    last_clean = max([h for h, _ in _log_from(hist, origin)], default=-1)
    blk = hist[hist["time_hours"] >= max(last_clean, origin - 23, 0)]
    rf = 1.0 / blk["U_overall_meas"].to_numpy(float) - 1.0 / config.U_CLEAN_DESIGN_W_M2K
    return float(max(0.0, np.median(rf)))


# -----------------------------------------------------------------------------
# Future inputs
# -----------------------------------------------------------------------------

def future_inputs(df: Optional[pd.DataFrame], origin: int, horizon: int, exog: str = "climatology",
                  temp_offset_c: float = 0.0, planned_cleanings: Optional[Sequence[dict]] = None):
    hours = np.arange(origin + 1, origin + 1 + horizon)
    if exog == "climatology":
        T = D.climatology_temperature(hours, temp_offset_c)
        flow = np.full(horizon, config.COOLING_WATER_MASS_FLOW_KG_S)
    elif exog == "oracle":
        if df is None:
            raise ValueError("oracle exogenous inputs need the dataset")
        fut = df.set_index("time_hours").reindex(hours)
        T = fut["T_cool_in_meas"].to_numpy(float) + temp_offset_c
        flow = fut["mass_flow_cool_meas"].to_numpy(float)
        clim_T = D.climatology_temperature(hours, temp_offset_c)
        T = np.where(np.isnan(T), clim_T, T)  # past the end of the dataset: climatology
        flow = np.where(np.isnan(flow), config.COOLING_WATER_MASS_FLOW_KG_S, flow)
    else:
        raise ValueError(f"exog must be 'climatology' or 'oracle', got {exog!r}")
    codes = np.zeros(horizon, dtype=np.int64)
    for ev in planned_cleanings or []:
        k = int(ev["hour"]) - origin - 1
        if 0 <= k < horizon:
            codes[k] = CLEANING_CODES[ev["type"]]
    return hours, T, flow, codes


def dataset_planned_cleanings(df: pd.DataFrame, origin: int, horizon: int) -> List[dict]:
    """The dataset's own cleanings inside the horizon (known in advance in evaluation)."""
    ev = df[(df["cleaning_event"] == 1) & (df["time_hours"] > origin) & (df["time_hours"] <= origin + horizon)]
    return [{"hour": int(h), "type": t} for h, t in zip(ev["time_hours"], ev["cleaning_type"])]


# -----------------------------------------------------------------------------
# Batched rollout over many origins
# -----------------------------------------------------------------------------

@torch.no_grad()
def rollout_trajectories(models: Sequence[FoulingUDE], states: Sequence[dict], hours: np.ndarray, T: np.ndarray,
                         flow: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """
    states: N origin states; hours/T/flow/codes: [N, H]. Returns delta trajectories
    [N, n_models * len(STATE_Z), H].
    """
    N, H = T.shape
    z = torch.tensor(STATE_Z, dtype=DT)
    d0 = torch.tensor([s["delta"] for s in states], dtype=DT)
    se = torch.tensor([s["delta_se"] for s in states], dtype=DT)
    d0 = torch.clamp(d0[:, None] + z[None, :] * se[:, None], 0.0, pt.DELTA_MAX)   # [N, Z]
    K = len(STATE_Z)
    rep = lambda a: torch.tensor(a, dtype=DT).repeat_interleave(K, dim=0)
    exog = {"T": rep(T), "flow": rep(flow), "hour": rep(hours.astype(float))}
    cleaning = torch.tensor(codes).repeat_interleave(K, dim=0)
    cum = torch.tensor([s["cum"] for s in states], dtype=DT).repeat_interleave(K)
    hsc = torch.tensor([s["hsc"] for s in states], dtype=DT).repeat_interleave(K)
    out = []
    for m in models:
        out.append(m.rollout(d0.reshape(-1), exog, cleaning, cum, hsc).reshape(N, K, H))
    return torch.cat(out, dim=1).numpy()


def observables(delta: np.ndarray, flow: np.ndarray):
    """delta [..., H] and flow broadcastable -> (Rf, U, dP kPa) arrays."""
    d = torch.tensor(delta, dtype=DT)
    f = torch.tensor(np.broadcast_to(flow, delta.shape).copy(), dtype=DT)
    rf, u, dp = pt.observe(d, f)
    return rf.numpy(), u.numpy(), dp.numpy()


def _band(x: np.ndarray, axis: int = 0) -> Dict[str, list]:
    return {"mean": np.mean(x, axis=axis), "lo": np.quantile(x, 0.05, axis=axis), "hi": np.quantile(x, 0.95, axis=axis)}


def _first_crossing(traj: np.ndarray, limit: float, hours: np.ndarray):
    """traj [M, H] -> first hour above limit per trajectory (nan if never)."""
    above = traj > limit
    idx = np.where(above.any(axis=1), above.argmax(axis=1), -1)
    return np.where(idx >= 0, hours[np.clip(idx, 0, None)].astype(float), np.nan)


def _ttl(crossings: np.ndarray):
    if np.all(np.isnan(crossings)):
        return None
    fraction = float(np.mean(~np.isnan(crossings)))
    filled = np.where(np.isnan(crossings), np.inf, crossings)
    q = lambda p: np.quantile(filled, p)
    as_num = lambda v: None if not np.isfinite(v) else float(v)
    return {"median": as_num(q(0.5)), "lo": as_num(q(0.05)), "hi": as_num(q(0.95)), "fraction_crossing": fraction}


# -----------------------------------------------------------------------------
# Public forecast (one origin)
# -----------------------------------------------------------------------------

def forecast_at(df: pd.DataFrame, origin_hour: int, horizon_h: int, model: str = "ude",
                planned_cleanings: Optional[Sequence[dict]] = None, exog: str = "climatology",
                temp_offset_c: float = 0.0, with_truth: bool = True) -> dict:
    """Forecast from a dataset at origin_hour; only rows <= origin are used for the state."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if not (1 <= horizon_h <= MAX_HORIZON_H):
        raise ValueError(f"horizon_h must be in 1..{MAX_HORIZON_H}")
    last = int(df["time_hours"].max())
    if not (D.BURN_IN_HOURS - 1 <= origin_hour <= last):
        raise ValueError(f"origin_hour must be in {D.BURN_IN_HOURS - 1}..{last}")
    hours, T, flow, codes = future_inputs(df, origin_hour, horizon_h, exog, temp_offset_c, planned_cleanings)
    notes = []
    if model == "persistence":
        rf0 = persistence_rf(df, origin_hour)
        delta = np.full((1, horizon_h), rf0 * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK)
        notes.append("persistence: Rf held at its last-24 h estimate; planned cleanings are ignored; no band")
    else:
        state = state_at_origin(df, origin_hour)
        delta = rollout_trajectories(load_ensemble(model), [state], hours[None], T[None], flow[None], codes[None])[0]
        notes.append(f"{model}: {delta.shape[0]} trajectories ({len(load_ensemble(model))} model(s) x {len(STATE_Z)} initial states)")
    rf, u, dp = observables(delta, flow[None])
    out = {
        "hours": hours.tolist(),
        "Rf": {k: v.tolist() for k, v in _band(rf).items()},
        "U": {k: v.tolist() for k, v in _band(u).items()},
        "dP_kpa": {k: v.tolist() for k, v in _band(dp).items()},
        "time_to_limit": {"rf": _ttl(_first_crossing(rf, config.CRITICAL_RF_M2K_W, hours)),
                          "dp": _ttl(_first_crossing(dp, config.CRITICAL_DELTA_P_KPA, hours))},
        "truth": {"Rf": None},
        "model": model,
        "notes": notes + [f"exogenous inputs: {exog}" + (f", temperature offset {temp_offset_c:+g} °C" if temp_offset_c else "")],
    }
    if with_truth:
        fut = df.set_index("time_hours").reindex(hours)["Rf_fouling_resistance"]
        if fut.notna().any():
            out["truth"]["Rf"] = [None if np.isnan(v) else float(v) for v in fut.to_numpy(float)]
    return out


def forecast(history_df: pd.DataFrame, horizon_hours: int, planned_cleanings=None, exog=None, model: str = "ude",
             temp_offset_c: float = 0.0) -> dict:
    """
    Forecast from a history ending at the origin (last row). `exog` may be None (climatology) or a
    dict with arrays "T" and "flow" of length horizon_hours (user-supplied future inputs).
    """
    origin = int(history_df["time_hours"].iloc[-1])
    if exog is None or isinstance(exog, str):
        return forecast_at(history_df, origin, horizon_hours, model, planned_cleanings, "climatology", temp_offset_c, with_truth=False)
    hours = np.arange(origin + 1, origin + 1 + horizon_hours)
    fut = pd.DataFrame({"time_hours": hours, "T_cool_in_meas": np.asarray(exog["T"], float),
                        "mass_flow_cool_meas": np.asarray(exog["flow"], float)})
    joined = pd.concat([history_df, fut], ignore_index=True)
    return forecast_at(joined, origin, horizon_hours, model, planned_cleanings, "oracle", temp_offset_c, with_truth=False)


# -----------------------------------------------------------------------------
# Learned correction: partial-dependence slices of g
# -----------------------------------------------------------------------------

CORRECTION_REFERENCE = {"T": 18.0, "delta_um": 50.0, "hours_since_cleaning": 240.0, "hour": 4380.0}


@torch.no_grad()
def correction_slices(n: int = 61) -> dict:
    """
    g = exp(net(x)) for the UDE ensemble, varying one input at a time around a reference state
    (T 18 °C, delta 50 µm at nominal flow, 240 h since cleaning, hour 4380). The season slice also
    returns the generator's nutrient factor rescaled by config K_DEPOSITION / fitted K_dep, i.e.
    the correction that would make the UDE deposition equal the generator's on nominal data.
    """
    models = load_ensemble("ude")
    ref = CORRECTION_REFERENCE
    flow = torch.tensor(config.COOLING_WATER_MASS_FLOW_KG_S, dtype=DT)

    def g_of(T, delta, hsc, hour):
        T, delta, hsc, hour = (torch.as_tensor(np.broadcast_to(a, (n,)).copy(), dtype=DT) for a in (T, delta, hsc, hour))
        v = pt.hydraulics(flow.expand(n), delta)[1]
        gs = np.stack([torch.exp(m.log_correction(T, v, hour, delta, hsc)).numpy() for m in models])
        return gs.mean(axis=0), gs.min(axis=0), gs.max(axis=0)

    d_ref = ref["delta_um"] * 1e-6
    spec = [
        ("water temperature (°C)", np.linspace(6.0, 32.0, n), lambda x: g_of(x, d_ref, ref["hours_since_cleaning"], ref["hour"])),
        ("day of year", np.linspace(0, 365, n), lambda x: g_of(ref["T"], d_ref, ref["hours_since_cleaning"], x * 24.0)),
        ("biofilm thickness (µm)", np.linspace(0.0, 200.0, n), lambda x: g_of(ref["T"], x * 1e-6, ref["hours_since_cleaning"], ref["hour"])),
        ("hours since cleaning", np.linspace(0.0, 720.0, n), lambda x: g_of(ref["T"], d_ref, x, ref["hour"])),
    ]
    slices = []
    k_dep_fit = float(np.mean([m.k_dep.item() for m in models]))
    for name, x, fn in spec:
        mean, lo, hi = fn(x)
        s = {"name": name, "x": x.tolist(), "g": mean.tolist(), "g_min": lo.tolist(), "g_max": hi.tolist()}
        if name == "day of year":
            nut = 1.0 + 0.3 * np.sin(2.0 * np.pi * (x * 24.0 - 2190.0) / 8760.0)
            s["generator_equivalent"] = (nut * config.K_DEPOSITION / k_dep_fit).tolist()
        slices.append(s)
    return {"inputs": ["T", "v", "season_sin", "season_cos", "delta/delta_max", "hours_since_cleaning/720 (clipped to [0, 1])"],
            "reference": ref, "ensemble_size": len(models), "slices": slices}


# -----------------------------------------------------------------------------
# First-cycle risk for candidate cleaning intervals (chance constraint, not wired into the optimiser)
# -----------------------------------------------------------------------------

def risk_by_interval(df: pd.DataFrame, origin_hour: int, t_min_days: int = 1, t_max_days: int = 90,
                     model: str = "ude", exog: str = "climatology", temp_offset_c: float = 0.0) -> dict:
    """
    For each candidate interval T (days), the fraction of ensemble trajectories (members x initial
    states) in which Rf or dP crosses its limit before the first cleaning at origin + T days,
    simulated from the origin with no cleaning. This is the first-cycle chance constraint
    P(cross before T) <= epsilon of docs/model.md, Section 6.
    """
    if model not in ("ude", "ode"):
        raise ValueError("risk needs an ensemble model: 'ude' or 'ode'")
    if not (1 <= t_min_days <= t_max_days <= 120):
        raise ValueError("need 1 <= t_min_days <= t_max_days <= 120")
    horizon = t_max_days * 24
    hours, T, flow, codes = future_inputs(df, origin_hour, horizon, exog, temp_offset_c, planned_cleanings=[])
    state = state_at_origin(df, origin_hour)
    delta = rollout_trajectories(load_ensemble(model), [state], hours[None], T[None], flow[None], codes[None])[0]
    rf, _, dp = observables(delta, flow[None])
    c_rf = _first_crossing(rf, config.CRITICAL_RF_M2K_W, hours)
    c_dp = _first_crossing(dp, config.CRITICAL_DELTA_P_KPA, hours)
    rows = []
    for days in range(t_min_days, t_max_days + 1):
        end = origin_hour + days * 24
        hit_rf = np.nan_to_num(c_rf, nan=np.inf) <= end
        hit_dp = np.nan_to_num(c_dp, nan=np.inf) <= end
        rows.append({"interval_days": days, "p_rf": float(hit_rf.mean()), "p_dp": float(hit_dp.mean()),
                     "p_any": float((hit_rf | hit_dp).mean())})
    return {"origin_hour": origin_hour, "model": model, "trajectories": int(delta.shape[0]),
            "state_estimate_um": state["delta"] * 1e6, "intervals": rows}
