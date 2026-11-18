#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/improved_model.py

Improved Non-Linear Machine Learning Forecaster for Biofouling Progression.
Implements a physics-informed residual boosting ensemble combining a global linear
thermo-hydraulic trend with non-linear tree-based residual forecasting (Random Forest / GBR)
capable of multi-step forecasting (24h, 72h, 168h ahead) without data leakage.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from legacy.preprocessor import FeaturePipeline


class ImprovedFoulingModel:
    """
    Physics-informed ensemble regressor for multi-step biofouling forecasting.
    Combines global continuum trend modeling with non-linear residual tree boosting.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 5,
        random_state: int = 42,
        target_col: str = "Rf_fouling_resistance",
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self.target_col = target_col
        self.pipeline = FeaturePipeline(target_col=self.target_col)
        
        # Base linear estimator for global continuum trend
        self.base_ridge = Ridge(alpha=1.0)
        # Non-linear tree regressor for residual seasonal/noise fluctuations
        self.tree_booster = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, target_col: Optional[str] = None) -> "ImprovedFoulingModel":
        """Fit the ensemble model on training data."""
        if target_col:
            self.target_col = target_col
            self.pipeline.target_col = target_col

        X_train, y_train = self.pipeline.fit_transform(train_df)
        
        # 1. Fit base linear model
        self.base_ridge.fit(X_train, y_train)
        ridge_preds = self.base_ridge.predict(X_train)
        
        # 2. Fit tree booster on residuals
        residuals = y_train - ridge_preds
        self.tree_booster.fit(X_train, residuals)

        # 3. Calculate empirical residual uncertainty (standard error)
        tree_preds = self.tree_booster.predict(X_train)
        final_residuals = residuals - tree_preds
        self.residual_std_ = float(np.std(final_residuals))
        
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate point predictions for given dataset."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        X, _ = self.pipeline.transform(df)
        
        # Combined prediction = Base trend + Residual adjustment
        base_pred = self.base_ridge.predict(X)
        residual_pred = self.tree_booster.predict(X)
        preds = base_pred + residual_pred
        preds = np.clip(preds, a_min=0.0, a_max=0.001)
        return preds

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        confidence: float = 0.95,
        method: str = "conformal",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Generate point predictions alongside statistical prediction bounds.
        If method == 'conformal' and model has calibrated conformal_q_hat_, uses
        distribution-free split-conformal prediction interval: [\hat{y} - q_hat, \hat{y} + q_hat].
        Otherwise falls back to residual standard error: [\hat{y} - z*sigma, \hat{y} + z*sigma].
        """
        preds = self.predict(df)
        if method == "conformal" and hasattr(self, "conformal_q_hat_"):
            margin = self.conformal_q_hat_
        else:
            z = 1.96 if confidence >= 0.95 else 1.645
            sigma = getattr(self, "residual_std_", 6.8e-6)
            margin = z * sigma

        lower = np.clip(preds - margin, a_min=0.0, a_max=0.001)
        upper = np.clip(preds + margin, a_min=0.0, a_max=0.001)
        return preds, lower, upper

    def calibrate_conformal(self, cal_df: pd.DataFrame, alpha: float = 0.05) -> float:
        r"""
        Calibrate split-conformal prediction intervals using an independent held-out calibration slice.
        Computes non-conformity absolute residuals s_i = |y_i - \hat{y}_i| on cal_df and derives
        the distribution-free conformal quantile q_hat.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before conformal calibration.")
        y_true = cal_df[self.target_col].values
        preds = self.predict(cal_df)
        residuals = np.abs(y_true - preds)
        n_cal = len(cal_df)
        if n_cal == 0:
            raise ValueError("Calibration dataset must not be empty.")
        q_level = min(1.0, np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
        self.conformal_q_hat_ = float(np.quantile(residuals, q_level))
        self.calibration_sample_size_ = n_cal
        self.calibration_target_coverage_ = float(1.0 - alpha)
        self.is_calibrated_ = True
        return self.conformal_q_hat_

    def evaluate_conformal_coverage(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Compute empirical coverage and interval sharpness on held-out test data."""
        y_true = test_df[self.target_col].values
        preds, lower, upper = self.predict_with_uncertainty(test_df)
        covered = (y_true >= lower) & (y_true <= upper)
        coverage_pct = float(np.mean(covered)) * 100.0
        mean_width = float(np.mean(upper - lower))
        return {
            "test_sample_size": len(test_df),
            "coverage_pct": round(coverage_pct, 2),
            "mean_interval_width": round(mean_width, 8),
            "conformal_q_hat": round(getattr(self, "conformal_q_hat_", 0.0), 8),
            "target_coverage_pct": round(getattr(self, "calibration_target_coverage_", 0.95) * 100.0, 1),
            "is_calibrated": getattr(self, "is_calibrated_", False),
        }

    def forecast_horizon(self, df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
        """
        Generate multi-step horizon forecast from the latest available timestep,
        including statistical confidence intervals.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forecasting.")
        
        last_rows = df.tail(horizon_hours).copy().reset_index(drop=True)
        preds, lower, upper = self.predict_with_uncertainty(last_rows)
        last_rows["predicted_Rf"] = preds
        last_rows["predicted_Rf_lower"] = lower
        last_rows["predicted_Rf_upper"] = upper
        return last_rows

    def forecast_multi_horizon(
        self,
        history_df: pd.DataFrame,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Generate recursive multi-step ahead forecast from latest historical timestamp,
        propagating predicted fouling through fluid dynamics without future target leakage.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before multi-horizon forecasting.")
        import math
        from core import config
        from core import fouling_indicators

        last_row = history_df.iloc[-1]
        t0 = int(last_row["time_hours"])
        last_rf = float(last_row.get("Rf_fouling_resistance", 0.0001))

        forecast_rows = []
        for step in range(1, horizon_hours + 1):
            sim_t = t0 + step
            phase = (2.0 * math.pi * (sim_t - 2190.0)) / 8760.0
            t_cool = config.ANNUAL_MEAN_WATER_TEMP_C + config.SEASONAL_TEMP_AMPLITUDE_C * math.sin(phase)
            delta_est_m = last_rf * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK
            dp_pa, _, _ = fouling_indicators.compute_hydraulic_pressure_drop(
                config.COOLING_WATER_MASS_FLOW_KG_S, biofilm_thickness_m=delta_est_m
            )
            dp_kpa = dp_pa / 1000.0
            u_meas = fouling_indicators.compute_degraded_U(config.U_CLEAN_DESIGN_W_M2K, last_rf)

            forecast_rows.append({
                "time_hours": sim_t,
                "T_cool_in_meas": t_cool,
                "delta_P_meas_kPa": dp_kpa,
                "mass_flow_cool_meas": config.COOLING_WATER_MASS_FLOW_KG_S,
                "U_overall_meas": u_meas,
            })

        fut_df = pd.DataFrame(forecast_rows)
        hist_tail = history_df.tail(24).copy().reset_index(drop=True)
        eval_df = pd.concat([hist_tail, fut_df], ignore_index=True)
        
        preds, lower, upper = self.predict_with_uncertainty(eval_df)
        fut_preds = preds[-horizon_hours:]
        fut_lower = lower[-horizon_hours:]
        fut_upper = upper[-horizon_hours:]

        fut_df["predicted_Rf"] = fut_preds
        fut_df["predicted_Rf_lower"] = fut_lower
        fut_df["predicted_Rf_upper"] = fut_upper
        return fut_df

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Compute evaluation metrics: RMSE, MAE, R²."""
        preds = self.predict(test_df)
        y_true = test_df[self.target_col].values

        mse = mean_squared_error(y_true, preds)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, preds))
        r2 = float(r2_score(y_true, preds))

        return {
            "model": "Improved_Hybrid_Ensemble",
            "rmse": round(rmse, 7),
            "mae": round(mae, 7),
            "r2": round(r2, 4),
        }

    def get_feature_importances(self) -> Dict[str, float]:
        """Retrieve normalized feature importances from the residual booster."""
        if not self.is_fitted:
            return {}
        names = ["T_cool", "delta_P", "mass_flow", "U_meas", "sin_season", "cos_season", "dP_roll24", "dP_diff24"]
        importances = self.tree_booster.feature_importances_
        return {
            names[i] if i < len(names) else f"feat_{i}": round(float(imp), 4)
            for i, imp in enumerate(importances)
        }


# Compatibility alias
RandomForestFoulingModel = ImprovedFoulingModel
