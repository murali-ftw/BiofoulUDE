#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/preprocessor.py

Feature Engineering and Zero-Leakage Chronological Data Partitioning.
Ensures that all normalization statistics and model features are fit strictly
on the training slice, preventing lookahead data leakage in time-series evaluation.
"""

import math
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.65,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data chronologically into train, validation, and test subsets.
    Strictly enforces temporal causality: max(t_train) < min(t_val) < min(t_test).
    """
    if not (0.1 < train_frac < 0.9):
        raise ValueError(f"Invalid train_frac: {train_frac}")
    if not (0.05 < val_frac < 0.5):
        raise ValueError(f"Invalid val_frac: {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError("Sum of train_frac and val_frac must be less than 1.0.")

    n_total = len(df)
    train_end = int(n_total * train_frac)
    val_end = int(n_total * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    # Verification of chronological ordering
    time_col = "time_hours" if "time_hours" in df.columns else "time_h"
    if time_col in df.columns:
        if len(train_df) > 0 and len(val_df) > 0:
            assert train_df[time_col].max() < val_df[time_col].min(), "Train/Val chronological violation"
        if len(val_df) > 0 and len(test_df) > 0:
            assert val_df[time_col].max() < test_df[time_col].min(), "Val/Test chronological violation"

    return train_df, val_df, test_df


def train_val_test_split(df: pd.DataFrame, train_frac: float = 0.65, val_frac: float = 0.15):
    """Alias for chronological_split."""
    return chronological_split(df, train_frac=train_frac, val_frac=val_frac)


def chronological_split_conformal(
    df: pd.DataFrame,
    train_frac: float = 0.50,
    val_frac: float = 0.15,
    cal_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data chronologically into train, validation, calibration, and test subsets.
    Strictly enforces temporal causality: max(t_train) < min(t_val) < min(t_cal) < min(t_test).
    Calibration slice is strictly held out from training to calibrate split-conformal prediction intervals.
    """
    if train_frac + val_frac + cal_frac >= 1.0:
        raise ValueError("Sum of train, val, and cal fractions must be strictly less than 1.0.")
    n_total = len(df)
    train_end = int(n_total * train_frac)
    val_end = int(n_total * (train_frac + val_frac))
    cal_end = int(n_total * (train_frac + val_frac + cal_frac))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    cal_df = df.iloc[val_end:cal_end].copy().reset_index(drop=True)
    test_df = df.iloc[cal_end:].copy().reset_index(drop=True)

    time_col = "time_hours" if "time_hours" in df.columns else "time_h"
    if time_col in df.columns:
        if len(train_df) > 0 and len(val_df) > 0:
            assert train_df[time_col].max() < val_df[time_col].min(), "Train/Val chronological violation"
        if len(val_df) > 0 and len(cal_df) > 0:
            assert val_df[time_col].max() < cal_df[time_col].min(), "Val/Cal chronological violation"
        if len(cal_df) > 0 and len(test_df) > 0:
            assert cal_df[time_col].max() < test_df[time_col].min(), "Cal/Test chronological violation"

    return train_df, val_df, cal_df, test_df


class FeaturePipeline:
    """
    Stateful feature transformation pipeline maintaining fitted scaler state.
    Computes rolling averages, derivatives, and Fourier seasonality without lookahead.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, target_col: str = "Rf_fouling_resistance"):
        self.feature_cols = feature_cols or [
            "T_cool_in_meas",
            "delta_P_meas_kPa",
            "mass_flow_cool_meas",
            "U_overall_meas",
        ]
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.scaler_ = self.scaler  # Maintain both naming styles for compatibility
        self.fitted_ = False

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer domain-specific physical and temporal features."""
        feat_df = df.copy()

        # 1. Fourier seasonal harmonics (annual cycle)
        if "time_hours" in feat_df.columns:
            t = feat_df["time_hours"].values
            feat_df["sin_season"] = np.sin(2.0 * np.pi * t / 8760.0)
            feat_df["cos_season"] = np.cos(2.0 * np.pi * t / 8760.0)
            feat_df["sin_diurnal"] = np.sin(2.0 * np.pi * (t % 24) / 24.0)
            feat_df["cos_diurnal"] = np.cos(2.0 * np.pi * (t % 24) / 24.0)

        # 2. Rolling short-term statistics (causal rolling with min_periods=1)
        if "delta_P_meas_kPa" in feat_df.columns:
            feat_df["delta_P_roll24"] = feat_df["delta_P_meas_kPa"].rolling(24, min_periods=1).mean()
            feat_df["delta_P_diff24"] = feat_df["delta_P_meas_kPa"].diff(24).fillna(0.0)

        if "T_cool_in_meas" in feat_df.columns:
            feat_df["T_cool_roll24"] = feat_df["T_cool_in_meas"].rolling(24, min_periods=1).mean()

        if "U_overall_meas" in feat_df.columns:
            feat_df["U_roll24"] = feat_df["U_overall_meas"].rolling(24, min_periods=1).mean()

        return feat_df

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and align feature matrix X and target vector y."""
        feat_df = self.create_features(df)
        cols_to_use = [c for c in self.feature_cols if c in feat_df.columns]
        
        # Add engineered features
        engineered = ["sin_season", "cos_season", "delta_P_roll24", "delta_P_diff24"]
        for c in engineered:
            if c in feat_df.columns and c not in cols_to_use:
                cols_to_use.append(c)

        X = feat_df[cols_to_use].values
        if self.target_col in feat_df.columns:
            y = feat_df[self.target_col].values
        else:
            y = np.zeros(len(feat_df))

        return X, y

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """Fit scaler strictly on training DataFrame."""
        X_train, _ = self.get_feature_matrix(train_df)
        self.scaler.fit(X_train)
        self.scaler_ = self.scaler
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform features using previously fitted scaler."""
        if not self.fitted_:
            raise RuntimeError("Pipeline must be fitted on training data before calling transform.")
        X, y = self.get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)
        return X_scaled, y

    def fit_transform(self, train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on training data and return scaled features."""
        return self.fit(train_df).transform(train_df)
