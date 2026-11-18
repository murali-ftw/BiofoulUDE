"""Data windows, truth-free state initialisation, training and leak-safety of forecasts."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ml  # noqa: F401
import torch

from ml import data as D
from ml import forecast as F
from ml import physics_torch as pt
from ml import train as T
from ml.ude import FoulingUDE

DT = torch.float64


def small_batches(n=8, days=3):
    cfg = T.TrainConfig(window_days=days, eval_every=5)
    df = D.load("nominal")
    r = D.split_ranges(df)
    L = days * 24
    train = D.make_windows(df, D.window_starts(df, r["train"], L)[:n], L)
    val = D.make_windows(df, D.window_starts(df, r["val"], L)[:4], L)
    return cfg, train, val


class TestDataAndTraining(unittest.TestCase):

    def test_split_is_by_whole_cycles(self):
        df = D.load("nominal")
        r = D.split_ranges(df)
        starts = set(D.cycle_starts(df))
        self.assertEqual(r["train"][0], 0)
        self.assertEqual(r["train"][1], r["val"][0])
        self.assertEqual(r["val"][1], r["test"][0])
        self.assertEqual(r["test"][1], len(df))
        self.assertIn(r["val"][0], starts)
        self.assertIn(r["test"][0], starts)

    def test_windows_have_no_cleaning_in_burn_in(self):
        df = D.load("nominal")
        clean = {h for h, _ in D.cleaning_log(df)}
        for s in D.window_starts(df, D.split_ranges(df)["train"], 336):
            self.assertFalse(any(h in clean for h in range(s, s + D.BURN_IN_HOURS)))

    def test_delta0_inversion_noise_free(self):
        for delta in (0.0, 20e-6, 80e-6, 150e-6, 300e-6):
            d = torch.full((12,), delta, dtype=DT)
            flow = torch.full((12,), 214.0, dtype=DT)
            _, u, dp = pt.observe(d, flow)
            with self.subTest(delta=delta):
                self.assertAlmostEqual(D.estimate_delta(u.numpy(), dp.numpy(), flow.numpy()), delta, delta=1e-10)

    def test_tiny_training_reduces_loss(self):
        cfg, train, val = small_batches()
        model = FoulingUDE(k_dep=0.5e-7, use_correction=False).double()  # start far from the fit
        before = float(T.huber_loss(train, model, cfg)[0])
        cfg.stage1_iters = 30
        with mock.patch.object(T, "FoulingUDE", lambda use_correction: FoulingUDE(k_dep=0.5e-7, use_correction=use_correction)):
            fitted = T.fit_stage1(train, val, cfg, "cpu", DT, [])
        self.assertLess(float(T.huber_loss(train, fitted, cfg)[0]), before)

    def test_cpu_determinism(self):
        cfg, train, val = small_batches()
        cfg.stage1_iters, cfg.stage2_max_iters = 5, 10
        runs = []
        for _ in range(2):
            ode = T.fit_stage1(train, val, cfg, "cpu", DT, [])
            ude, _, _ = T.fit_stage2(ode, train, val, cfg, 11, "cpu", DT, [])
            runs.append({k: v.clone() for k, v in ude.state_dict().items()})
        for k in runs[0]:
            with self.subTest(param=k):
                torch.testing.assert_close(runs[0][k], runs[1][k], rtol=0, atol=0)


class TestForecastLeakage(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(3)
        model = FoulingUDE().double()
        with torch.no_grad():
            model.net[-1].weight.normal_(0, 0.3)
        self.patch = mock.patch.object(F, "load_ensemble", lambda kind: (model,))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_corrupting_future_observations_does_not_change_forecast(self):
        df = D.load("stress_no_clean").copy()
        origin, horizon = 2500, 168
        base = {ex: F.forecast_at(df, origin, horizon, "ude", exog=ex, with_truth=False) for ex in ("climatology", "oracle")}
        bad = df.copy()
        future = bad["time_hours"] > origin
        for col in ("U_overall_meas", "delta_P_meas_kPa", "Rf_fouling_resistance", "biofilm_thickness_um", "bsi_score"):
            bad.loc[future, col] = bad.loc[future, col] * 3.0 + 1.0
        for ex in ("climatology", "oracle"):
            with self.subTest(exog=ex):
                self.assertEqual(F.forecast_at(bad, origin, horizon, "ude", exog=ex, with_truth=False), base[ex])
        # climatology also ignores future exogenous columns
        bad.loc[future, "T_cool_in_meas"] += 10.0
        self.assertEqual(F.forecast_at(bad, origin, horizon, "ude", exog="climatology", with_truth=False), base["climatology"])

    def test_persistence_ignores_future(self):
        df = D.load("nominal").copy()
        a = F.forecast_at(df, 3000, 72, "persistence", with_truth=False)
        df.loc[df["time_hours"] > 3000, "U_overall_meas"] = 900.0
        self.assertEqual(F.forecast_at(df, 3000, 72, "persistence", with_truth=False), a)


if __name__ == "__main__":
    unittest.main()
