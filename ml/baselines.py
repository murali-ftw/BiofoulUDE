"""
Leak-free baselines. Everything is computed from information available at the origin t0:
observations at hours <= t0 and the planned cleanings (maintenance schedule) in the horizon.

- persistence: Rf(t0 + h) = median over the last 24 h of 1/U - 1/U_clean.
- Ridge and GradientBoosting, one model per horizon, trained on NOMINAL-train origins with
  target = true Rf at t0 + h (the baselines see Rf targets in training; the UDE never does).

Features (from hours t0-23 .. t0 only): medians of U, dP, T, flow; hours since cleaning;
season sin/cos; current Rf estimate; plus, from the maintenance schedule, whether a planned
cleaning falls inside (t0, t0 + h] and how many hours before t0 + h it happens.
"""

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ml  # noqa: F401
from ml import data as D
from ml import forecast as F

FEATURES = ["U_med", "dP_med", "T_med", "flow_med", "hours_since_cleaning", "season_sin", "season_cos",
            "rf_est", "clean_in_horizon", "hours_after_clean_at_target"]


def features_at(df: pd.DataFrame, origin: int, horizon: int) -> Dict[str, float]:
    hist = df[(df["time_hours"] <= origin) & (df["time_hours"] > origin - 24)]
    log = F._log_from(df[df["time_hours"] <= origin], origin)
    hsc = F.hours_since_cleaning_from_log([h for h, _ in log], origin + 1)
    w = 2.0 * math.pi * origin / 8760.0
    planned = F.dataset_planned_cleanings(df, origin, horizon)
    after = float(origin + horizon - max(p["hour"] for p in planned)) if planned else -1.0
    return {
        "U_med": float(hist["U_overall_meas"].median()),
        "dP_med": float(hist["delta_P_meas_kPa"].median()),
        "T_med": float(hist["T_cool_in_meas"].median()),
        "flow_med": float(hist["mass_flow_cool_meas"].median()),
        "hours_since_cleaning": float(hsc),
        "season_sin": math.sin(w),
        "season_cos": math.cos(w),
        "rf_est": F.persistence_rf(df, origin),
        "clean_in_horizon": 1.0 if planned else 0.0,
        "hours_after_clean_at_target": after,
    }


def feature_matrix(df: pd.DataFrame, origins: Sequence[int], horizon: int) -> np.ndarray:
    return np.array([[features_at(df, o, horizon)[k] for k in FEATURES] for o in origins])


def training_origins(df: pd.DataFrame, horizon: int, stride_h: int = 6) -> List[int]:
    lo, hi = D.split_ranges(df)["train"]
    return list(range(lo + 24, hi - horizon, stride_h))


class HorizonRegressors:
    """One regressor per horizon, trained on nominal-train origins."""

    def __init__(self, kind: str, horizons: Sequence[int]):
        self.kind, self.horizons, self.models = kind, list(horizons), {}

    def _make(self):
        if self.kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        if self.kind == "gbm":
            return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
        raise ValueError(self.kind)

    def fit(self, df: pd.DataFrame) -> "HorizonRegressors":
        rf_true = df.set_index("time_hours")["Rf_fouling_resistance"]
        for h in self.horizons:
            origins = training_origins(df, h)
            X = feature_matrix(df, origins, h)
            y = rf_true.loc[[o + h for o in origins]].to_numpy(float)
            self.models[h] = self._make().fit(X, y * 1e4)   # scale target to O(1)
        return self

    def predict(self, df: pd.DataFrame, origins: Sequence[int], horizon: int) -> np.ndarray:
        return np.clip(self.models[horizon].predict(feature_matrix(df, origins, horizon)) / 1e4, 0.0, None)
