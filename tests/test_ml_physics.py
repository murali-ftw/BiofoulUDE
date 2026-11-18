"""Torch physics port: parity with the numpy implementation and gradcheck (CPU, float64)."""

from pathlib import Path
import sys
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ml  # noqa: F401
import torch

from core import config
from core import fouling_indicators as fi
from core import synthetic_generator as sg
from ml import physics_torch as pt

DT = torch.float64


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))


class TestPhysicsParity(unittest.TestCase):

    def test_ratkowsky(self):
        temps = np.linspace(-5.0, 55.0, 1201)
        ref = np.array([sg.ratkowsky_growth_factor(t) for t in temps])
        got = pt.ratkowsky_growth_factor(torch.tensor(temps, dtype=DT)).numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-12)

    def test_hydraulics_and_shear(self):
        deltas = np.linspace(0.0, config.BIOFILM_MAX_THICKNESS_M, 71)
        flows = np.linspace(180.0, 260.0, 9)
        D, F = np.meshgrid(deltas, flows)
        ref = np.array([fi.compute_hydraulic_pressure_drop(f, biofilm_thickness_m=d) for d, f in zip(D.ravel(), F.ravel())])
        dp, u, f, tau = pt.hydraulics(torch.tensor(F.ravel(), dtype=DT), torch.tensor(D.ravel(), dtype=DT))
        self.assertLess(rel_err(dp.numpy(), ref[:, 0]), 1e-5)
        self.assertLess(rel_err(u.numpy(), ref[:, 1]), 1e-5)
        self.assertLess(rel_err(f.numpy(), ref[:, 2]), 1e-5)
        tau_ref = (ref[:, 2] / 8.0) * config.WATER_DENSITY_KG_M3 * ref[:, 1] ** 2  # as in the generator
        self.assertLess(rel_err(tau.numpy(), tau_ref), 1e-5)

    def test_observation_model(self):
        deltas = np.linspace(0.0, config.BIOFILM_MAX_THICKNESS_M, 101)
        rf_ref = np.array([fi.compute_thermal_resistance(d) for d in deltas])
        u_ref = np.array([fi.compute_degraded_U(config.U_CLEAN_DESIGN_W_M2K, r) for r in rf_ref])
        dp_ref = np.array([fi.compute_hydraulic_pressure_drop(220.0, biofilm_thickness_m=d)[0] / 1000.0 for d in deltas])
        rf, u, dp = pt.observe(torch.tensor(deltas, dtype=DT), torch.full((len(deltas),), 220.0, dtype=DT))
        np.testing.assert_allclose(rf.numpy(), rf_ref, rtol=1e-5, atol=1e-15)
        self.assertLess(rel_err(u.numpy(), u_ref), 1e-5)
        self.assertLess(rel_err(dp.numpy(), dp_ref), 1e-5)

    def test_inversions(self):
        deltas = torch.linspace(0.0, 300e-6, 31, dtype=DT)
        flow = torch.full_like(deltas, 215.0)
        _, u, dp = pt.observe(deltas, flow)
        np.testing.assert_allclose(pt.delta_from_u(u).numpy(), deltas.numpy(), atol=1e-12)
        np.testing.assert_allclose(pt.delta_from_dp(dp, flow).numpy(), deltas.numpy(), atol=1e-10)

    def test_gradcheck_observation_model(self):
        delta = torch.linspace(5e-6, 300e-6, 6, dtype=DT).requires_grad_(True)
        flow = torch.linspace(200.0, 240.0, 6, dtype=DT).requires_grad_(True)

        def fn(d, m):
            rf, u, dp = pt.observe(d, m)
            return rf * 1e4, u / 1e3, dp

        self.assertTrue(torch.autograd.gradcheck(fn, (delta, flow), eps=1e-9, atol=1e-6, rtol=1e-4))

    def test_gradcheck_ratkowsky_and_shear(self):
        T = torch.linspace(8.0, 40.0, 7, dtype=DT).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(pt.ratkowsky_growth_factor, (T,), eps=1e-6, atol=1e-7))
        delta = torch.linspace(5e-6, 300e-6, 5, dtype=DT).requires_grad_(True)
        flow = torch.full((5,), 220.0, dtype=DT)
        self.assertTrue(torch.autograd.gradcheck(lambda d: pt.hydraulics(flow, d)[3], (delta,), eps=1e-9, atol=1e-6, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
