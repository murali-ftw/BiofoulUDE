"""
Datasets, cycle-based splits, multiple-shooting windows and truth-free state initialisation.

Only measured columns are used for training and forecasting:
    T_cool_in_meas, mass_flow_cool_meas, U_overall_meas, delta_P_meas_kPa,
plus the time index and the maintenance log (cleaning_event, cleaning_type).
True delta / Rf columns are read only by evaluation code, never here.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import ml  # noqa: F401
import torch

from core import config
from ml import physics_torch as pt
from ml.ude import CLEANING_CODES, NONE, cum_clean_hours_from_log, hours_since_cleaning_from_log

DB_DIR = Path(__file__).resolve().parent.parent / "db"
DATASET_FILES = {
    "nominal": "biofouling_synthetic_dataset.csv",
    "stress_no_clean": "stress_no_clean.csv",
    "stress_long_interval": "stress_long_interval.csv",
    "mismatch_stress": "mismatch_stress.csv",
}
OBSERVED = ["T_cool_in_meas", "mass_flow_cool_meas", "U_overall_meas", "delta_P_meas_kPa"]
BURN_IN_HOURS = 12

# Sensor noise scales used to standardise residuals.
SIGMA_U = 25.0       # W/m²K, generator's U noise
SIGMA_DP_KPA = 0.20  # kPa: 1.5% dP sensor noise plus ~2% from using measured (1% noisy) flow, at ~8 kPa

# Nominal split by whole cleaning cycles (cycle k starts at the k-th cleaning hour; cycle 0 at hour 0)
SPLIT_CYCLES = {"train": range(0, 8), "val": range(8, 10), "test": range(10, 13)}

_cache: Dict[str, pd.DataFrame] = {}


def load(name: str) -> pd.DataFrame:
    if name not in DATASET_FILES:
        raise KeyError(f"unknown dataset {name!r}")
    if name not in _cache:
        df = pd.read_csv(DB_DIR / DATASET_FILES[name], keep_default_na=False)
        _cache[name] = df
    return _cache[name]


def cleaning_log(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """[(hour, code)] from the maintenance log."""
    ev = df[df["cleaning_event"] == 1]
    return [(int(h), CLEANING_CODES[t]) for h, t in zip(ev["time_hours"], ev["cleaning_type"])]


def cycle_starts(df: pd.DataFrame) -> List[int]:
    return [0] + [h for h, _ in cleaning_log(df)]


def split_ranges(df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """Hour ranges [start, end) of the nominal train/val/test splits, aligned to cleaning cycles."""
    starts = cycle_starts(df) + [len(df)]
    out = {}
    for name, cycles in SPLIT_CYCLES.items():
        out[name] = (starts[cycles.start], starts[min(cycles.stop, len(starts) - 1)])
    return out


def climatology_temperature(hours: np.ndarray, temp_offset_c: float = 0.0) -> np.ndarray:
    """Seasonal water temperature from config (no diurnal term, no noise) plus an offset."""
    phase = 2.0 * np.pi * (np.asarray(hours, float) - 2190.0) / 8760.0
    return config.ANNUAL_MEAN_WATER_TEMP_C + config.SEASONAL_TEMP_AMPLITUDE_C * np.sin(phase) + temp_offset_c


def estimate_delta(u_meas: np.ndarray, dp_meas: np.ndarray, flow_meas: np.ndarray) -> float:
    """
    Truth-free thickness estimate from a short block of observations: robust median of the
    U-based and dP-based inversions (each a median over the block); with two estimators the
    median is their mean.
    """
    dt = torch.float64
    d_u = pt.delta_from_u(torch.tensor(np.asarray(u_meas, float), dtype=dt))
    d_p = pt.delta_from_dp(torch.tensor(np.asarray(dp_meas, float), dtype=dt), torch.tensor(np.asarray(flow_meas, float), dtype=dt))
    return float(np.median([float(torch.median(d_u)), float(torch.median(d_p))]))


def estimate_delta_spread(u_meas, dp_meas, flow_meas) -> float:
    """Standard error of the state estimate, from the scatter of per-hour inversions."""
    dt = torch.float64
    d_u = pt.delta_from_u(torch.tensor(np.asarray(u_meas, float), dtype=dt)).numpy()
    d_p = pt.delta_from_dp(torch.tensor(np.asarray(dp_meas, float), dtype=dt), torch.tensor(np.asarray(flow_meas, float), dtype=dt)).numpy()
    n = len(d_u)
    se_u = 1.2533 * np.std(d_u) / math.sqrt(n)  # standard error of a median
    se_p = 1.2533 * np.std(d_p) / math.sqrt(n)
    return float(0.5 * math.sqrt(se_u ** 2 + se_p ** 2) + 0.5 * abs(np.median(d_u) - np.median(d_p)))


@dataclass
class WindowBatch:
    """Multiple-shooting windows; rollout starts after the burn-in. All tensors [B] or [B, L]."""
    delta0: torch.Tensor
    cum0: torch.Tensor
    hsc0: torch.Tensor
    exog: Dict[str, torch.Tensor]
    cleaning: torch.Tensor
    u_obs: torch.Tensor
    dp_obs: torch.Tensor
    starts: List[int]

    def to(self, device, dtype):
        conv = lambda t: t.to(device=device, dtype=dtype)
        return WindowBatch(conv(self.delta0), conv(self.cum0), conv(self.hsc0), {k: conv(v) for k, v in self.exog.items()},
                           self.cleaning.to(device), conv(self.u_obs), conv(self.dp_obs), self.starts)


def window_starts(df: pd.DataFrame, hour_range: Tuple[int, int], length_h: int, stride_h: int = 24, offset_h: int = 0) -> List[int]:
    """Starts right after each cleaning (and at hour 0) plus every stride_h within cycles; no cleaning in the burn-in."""
    lo, hi = hour_range
    clean_hours = {h for h, _ in cleaning_log(df)}
    candidates = set()
    for c in cycle_starts(df):
        first = c + 1 if c in clean_hours else c
        candidates.add(first)
        candidates.update(range(first + offset_h, first + 720 * 3, stride_h))
    starts = []
    for s in sorted(candidates):
        if s < lo or s + length_h > hi:
            continue
        if any(h in clean_hours for h in range(s, s + BURN_IN_HOURS)):
            continue
        starts.append(s)
    return starts


def make_windows(df: pd.DataFrame, starts: Sequence[int], length_h: int) -> WindowBatch:
    log = cleaning_log(df)
    log_hours = [h for h, _ in log]
    codes = np.zeros(len(df), dtype=np.int64)
    for h, c in log:
        codes[h] = c
    T = df["T_cool_in_meas"].to_numpy(float)
    flow = df["mass_flow_cool_meas"].to_numpy(float)
    U = df["U_overall_meas"].to_numpy(float)
    dP = df["delta_P_meas_kPa"].to_numpy(float)

    d0, cum0, hsc0, sl = [], [], [], []
    for s in starts:
        b = slice(s, s + BURN_IN_HOURS)
        d0.append(estimate_delta(U[b], dP[b], flow[b]))
        r0 = s + BURN_IN_HOURS
        cum0.append(cum_clean_hours_from_log(log_hours, r0))
        hsc0.append(hours_since_cleaning_from_log(log_hours, r0))
        sl.append(slice(r0, s + length_h))
    dt = torch.float64
    stack = lambda a: torch.tensor(np.stack([a[x] for x in sl]), dtype=dt)
    return WindowBatch(
        delta0=torch.tensor(d0, dtype=dt), cum0=torch.tensor(cum0, dtype=dt), hsc0=torch.tensor(hsc0, dtype=dt),
        exog={"T": stack(T), "flow": stack(flow), "hour": stack(np.arange(len(df), dtype=float))},
        cleaning=torch.tensor(np.stack([codes[x] for x in sl])),
        u_obs=stack(U), dp_obs=stack(dP), starts=list(starts),
    )


def observation_residuals(model, batch: WindowBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardised U and dP residuals over the rollout, plus log g (for the penalty)."""
    deltas, log_g = model.rollout(batch.delta0, batch.exog, batch.cleaning, batch.cum0, batch.hsc0, return_log_g=True)
    _, u_hat, dp_hat = model.observe(deltas, batch.exog["flow"])
    return (u_hat - batch.u_obs) / SIGMA_U, (dp_hat - batch.dp_obs) / SIGMA_DP_KPA, log_g
