"""FoulingUDE: reproduces the generator's known physics, starts as the ODE, stays bounded."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ml  # noqa: F401
import torch

import cleaning_optimizer as co
import config
import synthetic_generator as sg
from ml import physics_torch as pt
from ml.ude import CHEMICAL, MECHANICAL, NONE, FoulingUDE, cum_clean_hours_from_log, hours_since_cleaning_from_log

DT = torch.float64


def true_exogenous(hours: int, seed: int):
    """
    Exact true water temperature (closed form, as in the generator) and true mass flow, replayed
    from the generator's RNG: each hour draws flow, T-sensor, dP-sensor, flow-sensor,
    T_out and U noise, in that order, independently of the state.
    """
    rng = np.random.default_rng(seed)
    T = np.empty(hours)
    flow = np.empty(hours)
    for h in range(hours):
        phase = (2.0 * math.pi * (h - 2190.0)) / 8760.0
        diurnal = (2.0 * math.pi * (h % 24)) / 24.0
        T[h] = (config.ANNUAL_MEAN_WATER_TEMP_C + config.SEASONAL_TEMP_AMPLITUDE_C * math.sin(phase)
                + config.DIURNAL_TEMP_AMPLITUDE_C * math.sin(diurnal))
        flow[h] = config.COOLING_WATER_MASS_FLOW_KG_S * (1.0 + rng.normal(0.0, 0.012))
        for s in (0.22, 0.015, 0.010, 0.25, 25.0):
            rng.normal(0.0, s)
    return T, flow


def nutrient_factor(hours):
    phase = 2.0 * np.pi * (np.arange(hours) - 2190.0) / 8760.0
    return 1.0 + 0.3 * np.sin(phase)


class TestUDE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hours = 1600  # first cleaning at 719, second at 1439: two full cycles
        cls.gen = co.simulate_policy(sg.default_cleaning_policy, hours=cls.hours, seed=42)
        T, flow = true_exogenous(cls.hours, 42)
        codes = np.zeros(cls.hours, dtype=np.int64)
        for h, typ in zip(cls.gen.df.index[cls.gen.df["cleaning_event"] == 1], cls.gen.cleaning_types):
            codes[h] = MECHANICAL if typ == sg.MECHANICAL else CHEMICAL
        cls.exog = {"T": torch.tensor(T, dtype=DT)[None], "flow": torch.tensor(flow, dtype=DT)[None],
                    "hour": torch.arange(cls.hours, dtype=DT)[None]}
        cls.cleaning = torch.tensor(codes)[None]
        cls.zero = torch.zeros(1, dtype=DT)

    def test_known_physics_reproduces_generator(self):
        """With g = the generator's nutrient factor, true inputs and true delta0 = 0, the trajectory is exact."""
        model = FoulingUDE().double()
        g = torch.tensor(nutrient_factor(self.hours), dtype=DT)[None]
        with torch.no_grad():
            delta = model.rollout(self.zero, self.exog, self.cleaning, self.zero, self.zero, g_override=g)[0].numpy()
        np.testing.assert_allclose(delta, self.gen.delta_m, rtol=1e-9, atol=1e-15)

    def test_g_equal_one_misses_only_the_nutrient_factor(self):
        """g = 1 (pure physics) deviates: the generator's seasonal nutrient factor is what g must learn."""
        model = FoulingUDE(use_correction=False).double()
        with torch.no_grad():
            delta = model.rollout(self.zero, self.exog, self.cleaning, self.zero, self.zero)[0].numpy()
        self.assertGreater(np.max(np.abs(delta - self.gen.delta_m)), 1e-6)

    def test_ude_equals_ode_at_init(self):
        ude, ode = FoulingUDE().double(), FoulingUDE(use_correction=False).double()
        with torch.no_grad():
            a = ude.rollout(self.zero, self.exog, self.cleaning, self.zero, self.zero)
            b = ode.rollout(self.zero, self.exog, self.cleaning, self.zero, self.zero)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_state_bounded(self):
        model = FoulingUDE().double()
        with torch.no_grad():
            model.net[-1].bias.fill_(3.0)  # g = e^3: very fast growth
            B, L = 4, 3000
            exog = {"T": torch.full((B, L), 28.0, dtype=DT), "flow": torch.full((B, L), 220.0, dtype=DT),
                    "hour": torch.arange(L, dtype=DT).expand(B, L)}
            d = model.rollout(torch.tensor([0.0, 1e-4, 3e-4, 3.5e-4], dtype=DT), exog,
                              torch.zeros(B, L, dtype=torch.long), torch.full((B,), 500.0, dtype=DT), torch.zeros(B, dtype=DT))
        self.assertGreaterEqual(float(d.min()), 0.0)
        self.assertLessEqual(float(d.max()), pt.DELTA_MAX)

    def test_cleaning_reset_both_types(self):
        model = FoulingUDE().double()
        delta = torch.tensor([100e-6, 100e-6, 100e-6], dtype=DT)
        cleaning = torch.tensor([NONE, MECHANICAL, CHEMICAL])
        T = torch.full((3,), 5.0, dtype=DT)  # below T_min: no growth
        d, cum, hsc, _ = model.step(delta, torch.full((3,), 400.0, dtype=DT), torch.full((3,), 50.0, dtype=DT),
                                    T, torch.full((3,), 220.0, dtype=DT), torch.zeros(3, dtype=DT), cleaning)
        d = d.detach()
        no_clean = float(d[0])
        self.assertAlmostEqual(float(d[1]), (1 - config.MECHANICAL_CLEANING_EFFICIENCY) * no_clean, places=15)
        self.assertAlmostEqual(float(d[2]), (1 - config.CHEMICAL_BIOCIDE_EFFICIENCY) * no_clean, places=15)
        self.assertEqual(hsc.tolist(), [51.0, 0.0, 0.0])
        self.assertEqual(cum.tolist(), [401.0, 401.0 * 0.2, 401.0 * 0.2])

    def test_gradients_reach_all_parameters(self):
        model = FoulingUDE().double()
        with torch.no_grad():
            model.net[-1].weight.normal_(0, 0.1)  # nonzero so net weights before the last layer get gradient
        L = 200
        exog = {k: v[:, :L] for k, v in self.exog.items()}
        d = model.rollout(torch.tensor([20e-6], dtype=DT), exog, self.cleaning[:, :L],
                          torch.tensor([300.0], dtype=DT), torch.zeros(1, dtype=DT))
        _, u, dp = model.observe(d, exog["flow"])
        (u.mean() + dp.mean()).backward()
        for name, p in model.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(p.grad)
                self.assertGreater(float(p.grad.abs().sum()), 0.0)

    def test_log_bookkeeping_matches_generator(self):
        """Lag-phase and hours-since-cleaning bookkeeping replayed from the cleaning log equals the generator state."""
        from datetime import datetime
        log = self.gen.df.index[self.gen.df["cleaning_event"] == 1].tolist()
        for upto in (500, 719, 720, 1000, 1440, 1599):
            state, rng = sg.FoulingState(), np.random.default_rng(42)
            for h in range(upto):
                state, _ = sg.step(state, h, sg.default_cleaning_policy, rng, datetime(2026, 1, 1), 7.91)
            with self.subTest(upto=upto):
                self.assertAlmostEqual(cum_clean_hours_from_log(log, upto), state.cum_clean_hours, places=9)
                self.assertEqual(hours_since_cleaning_from_log(log, upto), state.time_since_last_clean)


if __name__ == "__main__":
    unittest.main()
