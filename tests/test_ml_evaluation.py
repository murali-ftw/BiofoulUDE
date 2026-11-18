"""Evaluation helpers on toy cases, and leak-freedom of the baseline feature builder."""

from pathlib import Path
import sys
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ml  # noqa: F401
from ml import baselines as B
from ml import data as D
from ml import evaluate as E


class TestEvaluationHelpers(unittest.TestCase):

    def test_regression_metrics_known_answer(self):
        m = E.regression_metrics([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 6.0])
        self.assertAlmostEqual(m["mae"], 1.0)
        self.assertAlmostEqual(m["rmse"], np.sqrt((1 + 0 + 1 + 4) / 4))
        self.assertAlmostEqual(m["r2"], 1.0 - 6.0 / 5.0)
        perfect = E.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual((perfect["mae"], perfect["rmse"], perfect["r2"]), (0.0, 0.0, 1.0))

    def test_interval_coverage(self):
        self.assertAlmostEqual(E.interval_coverage([1, 2, 3, 4], [0, 2.5, 2, 5], [2, 3, 4, 6]), 0.5)

    def test_first_crossing_hour(self):
        self.assertEqual(E.first_crossing_hour([1, 2, 5, 7], [10, 11, 12, 13], 4.0), 12)
        self.assertIsNone(E.first_crossing_hour([1, 2], [0, 1], 4.0))

    def test_linear_trend_crossing(self):
        hours = np.arange(0, 100)
        self.assertAlmostEqual(E.linear_trend_crossing(hours, 2.0 * hours + 1.0, 201.0, 99), 100.0)
        self.assertIsNone(E.linear_trend_crossing(hours, -hours, 5.0, 99))


class TestBaselineFeaturesLeakFree(unittest.TestCase):

    def test_features_ignore_future_rows(self):
        df = D.load("stress_long_interval").copy()
        origin, horizon = 3000, 168
        before = B.features_at(df, origin, horizon)
        future = df["time_hours"] > origin
        for col in ("U_overall_meas", "delta_P_meas_kPa", "T_cool_in_meas", "mass_flow_cool_meas", "Rf_fouling_resistance"):
            df.loc[future, col] = -999.0
        self.assertEqual(B.features_at(df, origin, horizon), before)

    def test_features_window_is_24h(self):
        df = D.load("nominal").copy()
        origin = 4000
        before = B.features_at(df, origin, 24)
        df.loc[df["time_hours"] <= origin - 24, ["U_overall_meas", "delta_P_meas_kPa"]] = -1.0
        self.assertEqual(B.features_at(df, origin, 24), before)

    def test_training_origins_stay_in_train_split(self):
        df = D.load("nominal")
        lo, hi = D.split_ranges(df)["train"]
        for h in (1, 168):
            origins = B.training_origins(df, h)
            self.assertGreaterEqual(min(origins) - 23, lo)
            self.assertLess(max(origins) + h, hi)


if __name__ == "__main__":
    unittest.main()
