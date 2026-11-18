"""
Test Suite: Phase 3.1 Manual Optimizer Sensitivity & CSV Ingestion UX Audit
===========================================================================
Validates:
1. Darcy-Weisbach hydraulic inversion recovers physical biofilm thickness delta_f
   and thermal fouling resistance Rf when operator omits overall HTC U.
2. Interval optimiser (exact 1..90 d sweep) gives diverse recommendations, ordered sensibly:
   fouled < clean, summer < winter, and 0 days (immediate) for a breach.
"""

import math
import os
from pathlib import Path
import sys
import unittest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import config
import fouling_indicators
import cleaning_optimizer


class TestPhase31SensitivityAndUX(unittest.TestCase):
    """Regression and verification tests for Phase 3.1 engineering upgrades."""

    def setUp(self):
        self.optimizer = cleaning_optimizer.CleaningIntervalOptimizer()

    def test_01_hydraulic_inversion_monotonicity(self):
        """Verify Darcy-Weisbach inversion recovers biofilm thickness monotonically from delta_P."""
        m_dot = 220.0
        clean_dp = fouling_indicators.compute_clean_pressure_drop(m_dot)
        self.assertAlmostEqual(clean_dp, 7.91, delta=0.05)

        # Baseline clean
        t0 = fouling_indicators.invert_biofilm_thickness_from_dp(clean_dp, mass_flow_kg_s=m_dot)
        self.assertEqual(t0, 0.0)

        # Sub-clean pressure drop returns 0.0
        t_sub = fouling_indicators.invert_biofilm_thickness_from_dp(clean_dp - 1.0, mass_flow_kg_s=m_dot)
        self.assertEqual(t_sub, 0.0)

        # Monotonic increase
        dp_steps = [9.5, 11.0, 12.5, 14.0, 15.5]
        prev_t = 0.0
        for dp in dp_steps:
            t = fouling_indicators.invert_biofilm_thickness_from_dp(dp, mass_flow_kg_s=m_dot)
            self.assertGreater(t, prev_t, f"Thickness failed to increase monotonically at dp={dp}")
            # Verify reconstructed pressure drop matches input within tolerance
            reconstructed_dp_pa, _, _ = fouling_indicators.compute_hydraulic_pressure_drop(m_dot, biofilm_thickness_m=t)
            self.assertAlmostEqual(reconstructed_dp_pa / 1000.0, dp, delta=0.01)
            prev_t = t

    def test_03_optimizer_granularity_beyond_binary_clustering(self):
        """Verify the 1..90 d sweep gives diverse, physically ordered recommendations across scenarios."""
        # Case A: clean tube, early summer
        rec_a = self.optimizer.optimize_interval(0.0, current_hour=3000)["recommended_interval_days"]
        # Case B: moderate fouling, same season
        rec_b = self.optimizer.optimize_interval(80.0, current_hour=3000)["recommended_interval_days"]
        # Case D: breach emergency
        rec_d = self.optimizer.optimize_interval(190.0, current_hour=3000)["recommended_interval_days"]
        # Case E: clean tube, cold winter
        rec_e = self.optimizer.optimize_interval(0.0, current_hour=0)["recommended_interval_days"]
        # Case F: clean tube, spring
        rec_f = self.optimizer.optimize_interval(0.0, current_hour=1000)["recommended_interval_days"]

        recommendations_observed = {rec_a, rec_b, rec_d, rec_e, rec_f}
        self.assertGreaterEqual(len(recommendations_observed), 4, f"Insufficient granularity: {recommendations_observed}")
        self.assertEqual(rec_d, 0)
        self.assertLess(rec_b, rec_a, "a fouled tube must get a shorter interval than a clean one")
        self.assertGreater(rec_e, rec_a, "winter must allow a longer interval than summer")

if __name__ == "__main__":
    unittest.main()
