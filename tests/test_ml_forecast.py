"""Forecast API on the trained artifacts: contract shape, determinism, risk monotonicity."""

from pathlib import Path
import sys
import unittest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ml  # noqa: F401
from ml import data as D
from ml import forecast as F
from ml import train as T

ARTIFACTS = T.ODE_PATH.exists() and all(T.ude_path(s).exists() for s in T.ENSEMBLE_SEEDS)


@unittest.skipUnless(ARTIFACTS, "ml artifacts missing: run python -m ml.train")
class TestForecastAPI(unittest.TestCase):

    def test_forecast_shape_and_bands(self):
        hist = D.load("nominal").iloc[:3001]
        for model in ("ude", "ode", "persistence"):
            with self.subTest(model=model):
                f = F.forecast(hist, 72, planned_cleanings=[{"hour": 3040, "type": "mechanical"}], model=model)
                self.assertEqual(f["hours"], list(range(3001, 3073)))
                for q in ("Rf", "U", "dP_kpa"):
                    for k in ("mean", "lo", "hi"):
                        self.assertEqual(len(f[q][k]), 72)
                    self.assertTrue(all(lo <= m + 1e-15 and m <= hi + 1e-15 for lo, m, hi in zip(f[q]["lo"], f[q]["mean"], f[q]["hi"])))
                self.assertEqual(set(f["time_to_limit"]), {"rf", "dp"})
                self.assertIsNone(f["truth"]["Rf"])  # history only: no truth available

    def test_planned_cleaning_resets_forecast(self):
        hist = D.load("nominal").iloc[:3001]
        f = F.forecast(hist, 72, planned_cleanings=[{"hour": 3040, "type": "chemical"}], model="ude")
        i = f["hours"].index(3040)
        self.assertLess(f["Rf"]["mean"][i], 0.2 * f["Rf"]["mean"][i - 1])

    def test_deterministic(self):
        hist = D.load("mismatch_stress").iloc[:4001]
        self.assertEqual(F.forecast(hist, 168, model="ude"), F.forecast(hist, 168, model="ude"))

    def test_user_supplied_exog(self):
        hist = D.load("nominal").iloc[:2001]
        warm = F.forecast(hist, 168, exog={"T": [27.0] * 168, "flow": [220.0] * 168})
        cold = F.forecast(hist, 168, exog={"T": [8.0] * 168, "flow": [220.0] * 168})
        self.assertGreater(warm["Rf"]["mean"][-1], cold["Rf"]["mean"][-1])

    def test_risk_by_interval_monotone(self):
        df = D.load("stress_no_clean")
        r = F.risk_by_interval(df, 2800, 1, 60)
        p = [row["p_any"] for row in r["intervals"]]
        self.assertEqual(len(p), 60)
        self.assertTrue(all(a <= b for a, b in zip(p, p[1:])))
        self.assertEqual(p[0], 0.0)
        self.assertEqual(p[-1], 1.0)   # Rf truly crosses at h 3305, ~21 days after the origin

    def test_correction_slices(self):
        c = F.correction_slices(n=11)
        self.assertEqual(c["ensemble_size"], len(T.ENSEMBLE_SEEDS))
        for s in c["slices"]:
            self.assertEqual(len(s["x"]), 11)
            self.assertEqual(len(s["g"]), 11)
            self.assertTrue(all(g > 0 for g in s["g"]))


if __name__ == "__main__":
    unittest.main()
