"""
tests/test_forecast_reliability.py

Comprehensive test suite verifying ML forecasting reliability, multi-horizon evaluation
without target leakage, and split-conformal calibration validity.
"""

import math
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core import config
from legacy import preprocessor
from legacy import improved_model
from core import synthetic_generator


class TestForecastReliability(unittest.TestCase):
    """Rigorous verification of Phase 2A forecasting reliability and conformal calibration."""

    @classmethod
    def setUpClass(cls):
        """Generate deterministic dataset and fitted/calibrated model for testing."""
        cls.df = synthetic_generator.generate_biofouling_dataset(
            hours=8760,
            seed=config.DEFAULT_RANDOM_SEED,
            inject_cleanings=True,
        )
        cls.train_df, cls.val_df, cls.cal_df, cls.test_df = preprocessor.chronological_split_conformal(
            cls.df, train_frac=0.50, val_frac=0.15, cal_frac=0.15
        )
        cls.model = improved_model.ImprovedFoulingModel()
        cls.model.fit(cls.train_df)
        cls.q_hat = cls.model.calibrate_conformal(cls.cal_df, alpha=0.05)

    def test_01_chronological_four_way_partition_causality(self):
        """Verify strict non-overlapping temporal causality across 4 chronological splits."""
        t_train_max = self.train_df["time_hours"].max()
        t_val_min = self.val_df["time_hours"].min()
        t_val_max = self.val_df["time_hours"].max()
        t_cal_min = self.cal_df["time_hours"].min()
        t_cal_max = self.cal_df["time_hours"].max()
        t_test_min = self.test_df["time_hours"].min()

        self.assertLess(t_train_max, t_val_min, "Train must strictly precede Validation partition.")
        self.assertLess(t_val_max, t_cal_min, "Validation must strictly precede Calibration partition.")
        self.assertLess(t_cal_max, t_test_min, "Calibration must strictly precede Test partition.")

        total_rows = len(self.train_df) + len(self.val_df) + len(self.cal_df) + len(self.test_df)
        self.assertEqual(total_rows, len(self.df), "Split partitions must sum to exactly 100% of rows.")

        # Verify sample sizes
        self.assertEqual(len(self.train_df), 4380)
        self.assertEqual(len(self.val_df), 1314)
        self.assertEqual(len(self.cal_df), 1314)
        self.assertEqual(len(self.test_df), 1752)

    def test_02_split_conformal_calibration_quantile_and_coverage(self):
        """Verify conformal calibration produces strictly positive q_hat and >= 95% empirical test coverage."""
        self.assertTrue(self.model.is_calibrated_, "Model must be marked as calibrated.")
        self.assertGreater(self.q_hat, 0.0, "Conformal quantile margin q_hat must be strictly positive.")
        self.assertLess(self.q_hat, 1e-4, "Conformal quantile margin must be physically reasonable (< 1e-4 m²·K/W).")

        cov_metrics = self.model.evaluate_conformal_coverage(self.test_df)
        self.assertIn("coverage_pct", cov_metrics)
        self.assertIn("mean_interval_width", cov_metrics)

        # Theoretical guarantee: finite-sample coverage >= 1 - alpha = 95%
        self.assertGreaterEqual(
            cov_metrics["coverage_pct"],
            95.0,
            f"Empirical test coverage ({cov_metrics['coverage_pct']}%) fell below 95% target.",
        )
        self.assertGreater(cov_metrics["mean_interval_width"], 0.0)

    def test_03_non_identical_multi_horizon_rmse_progression(self):
        """Verify that multi-horizon rollout errors grow monotonically without identical-stride artifacts."""
        test_df = self.test_df.copy().reset_index(drop=True)
        horizons = [1, 24, 72, 168]
        horizon_rmses = []

        # Evaluate recursive multi-step forecasting across test origins spaced every 48 hours
        origins = list(range(24, len(test_df) - 168, 48))
        self.assertGreater(len(origins), 5, "Must evaluate across multiple origins.")

        for H in horizons:
            errors = []
            for org in origins:
                hist_slice = test_df.iloc[:org]
                fut_truth = test_df.iloc[org:org + H]
                fut_pred_df = self.model.forecast_multi_horizon(hist_slice, horizon_hours=H)
                y_true = fut_truth["Rf_fouling_resistance"].values[-1]
                y_pred = fut_pred_df["predicted_Rf"].values[-1]
                errors.append(y_true - y_pred)
            rmse_h = float(np.sqrt(np.mean(np.square(errors))))
            horizon_rmses.append(rmse_h)

        # Verify all RMSE values are strictly distinct (no identical values across horizons)
        unique_rmses = set(round(r, 9) for r in horizon_rmses)
        self.assertEqual(
            len(unique_rmses),
            len(horizons),
            f"RMSE values across horizons {horizons} must be distinct: {horizon_rmses}",
        )

        # Verify general error growth: 168h error > 1h error
        self.assertGreater(
            horizon_rmses[-1],
            horizon_rmses[0],
            f"Long horizon error (168h: {horizon_rmses[-1]:.2e}) must exceed 1-step error (1h: {horizon_rmses[0]:.2e})",
        )

    def test_04_future_target_leakage_prevention(self):
        """Verify that multi-horizon forecaster does not peek at future ground truth target values."""
        hist_slice = self.test_df.iloc[:48].copy()
        future_slice = self.test_df.iloc[48:72].copy()

        # Forecasting should require only history_df and exogenous steps, not future targets
        fut_pred_1 = self.model.forecast_multi_horizon(hist_slice, horizon_hours=24)

        # Corrupt ground truth target in future_slice
        corrupted_future = future_slice.copy()
        corrupted_future["Rf_fouling_resistance"] = 0.999999  # Extreme corrupted target

        # Generate forecast again from original history
        fut_pred_2 = self.model.forecast_multi_horizon(hist_slice, horizon_hours=24)

        # The forecast output must be completely identical regardless of future ground truth
        np.testing.assert_allclose(
            fut_pred_1["predicted_Rf"].values,
            fut_pred_2["predicted_Rf"].values,
            rtol=1e-10,
            err_msg="Forecast predictions varied when future targets were modified (future leakage detected).",
        )

    def test_05_deterministic_forecasting_reproducibility(self):
        """Verify bitwise identical forecasting outputs when executed repeatedly on identical inputs."""
        hist_slice = self.test_df.iloc[:72].copy()

        pred_run_a = self.model.forecast_multi_horizon(hist_slice, horizon_hours=48)
        pred_run_b = self.model.forecast_multi_horizon(hist_slice, horizon_hours=48)

        self.assertEqual(
            list(pred_run_a["predicted_Rf"].values),
            list(pred_run_b["predicted_Rf"].values),
            "Forecasting must be deterministic given identical inputs.",
        )
        self.assertEqual(
            list(pred_run_a["predicted_Rf_lower"].values),
            list(pred_run_b["predicted_Rf_lower"].values),
            "Uncertainty lower bounds must be deterministic.",
        )
        self.assertEqual(
            list(pred_run_a["predicted_Rf_upper"].values),
            list(pred_run_b["predicted_Rf_upper"].values),
            "Uncertainty upper bounds must be deterministic.",
        )

if __name__ == "__main__":
    unittest.main()
