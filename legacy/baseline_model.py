#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/baseline_model.py

Baseline Predictive Modeling for Heat-Exchanger Fouling Resistance.
Implements Ridge Linear Regression and Exponentially Weighted Moving Average (EWMA) baselines.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from legacy.preprocessor import FeaturePipeline


class BaselineFoulingModel:
    """
    Ridge Linear Regression baseline model for thermal fouling resistance forecasting.
    """

    def __init__(self, alpha: float = 1.0, target_col: str = "Rf_fouling_resistance"):
        self.alpha = float(alpha)
        self.target_col = target_col
        self.pipeline = FeaturePipeline(target_col=self.target_col)
        self.model = Ridge(alpha=self.alpha)
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, target_col: Optional[str] = None) -> "BaselineFoulingModel":
        """Fit the baseline model on the training slice."""
        if target_col:
            self.target_col = target_col
            self.pipeline.target_col = target_col

        X_train, y_train = self.pipeline.fit_transform(train_df)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for a given DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        X, _ = self.pipeline.transform(df)
        preds = self.model.predict(X)
        # Physical boundary: thermal fouling resistance cannot be negative
        preds = np.clip(preds, a_min=0.0, a_max=0.001)
        return preds

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Compute evaluation metrics: RMSE, MAE, R²."""
        preds = self.predict(test_df)
        y_true = test_df[self.target_col].values

        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, preds))
        r2 = float(r2_score(y_true, preds))

        return {
            "model": "Baseline_Ridge",
            "rmse": round(rmse, 7),
            "mae": round(mae, 7),
            "r2": round(r2, 4),
        }


# Compatibility alias
RidgeFoulingModel = BaselineFoulingModel
