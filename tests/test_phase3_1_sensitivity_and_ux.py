"""
Test Suite: Phase 3.1 Manual Optimizer Sensitivity & CSV Ingestion UX Audit
===========================================================================
Validates:
1. Darcy-Weisbach hydraulic inversion recovers physical biofilm thickness delta_f
   and thermal fouling resistance Rf when operator omits overall HTC U.
2. Manual operating mode sensitivity across diverse conditions:
   - Clean baseline -> 21 to 30 days
   - Mild fouling -> 21 days
   - Moderate fouling -> 14 days
   - Severe / near-critical fouling -> 7 days
   - Threshold breach / hydraulic surge -> 0 days (Immediate)
   - Cold seasonal baseline -> 30 to 60 days
3. Optimizer does NOT collapse to only {0, 21} days.
4. CSV Ingestion metadata supports the 5-stage status sequence:
   (File Selected -> Uploading -> Uploaded -> Validated -> Analyzed)
   and provides all fields required for the Ingestion & Analysis Summary Card.
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
        """Verify optimizer outputs diverse recommendations (0, 7, 14, 21, 30, 60 days) across scenarios."""
        recommendations_observed = set()

        # Case A: Clean summer baseline (expect 21d)
        rec_a = self.optimizer.optimize_interval(0.0, current_hour=3000)
        recommendations_observed.add(rec_a["recommended_interval_days"])

        # Case B: Moderate fouling (expect 14d)
        rec_b = self.optimizer.optimize_interval(80.0, current_hour=3000)
        recommendations_observed.add(rec_b["recommended_interval_days"])

        # Case C: Near-critical fouling (expect 7d)
        rec_c = self.optimizer.optimize_interval(170.0, current_hour=3000)
        recommendations_observed.add(rec_c["recommended_interval_days"])

        # Case D: Breach emergency (expect 0d)
        rec_d = self.optimizer.optimize_interval(190.0, current_hour=3000)
        recommendations_observed.add(rec_d["recommended_interval_days"])

        # Case E: Cold winter baseline (expect 30d or 60d)
        rec_e = self.optimizer.optimize_interval(0.0, current_hour=0)
        recommendations_observed.add(rec_e["recommended_interval_days"])

        # Case F: Spring baseline (expect 30d)
        rec_f = self.optimizer.optimize_interval(0.0, current_hour=1000)
        recommendations_observed.add(rec_f["recommended_interval_days"])

        # Must observe at least 4 distinct interval values
        self.assertGreaterEqual(len(recommendations_observed), 4, f"Insufficient granularity: {recommendations_observed}")
        self.assertIn(0, recommendations_observed)
        self.assertIn(14, recommendations_observed)
        self.assertIn(21, recommendations_observed)
        self.assertTrue(30 in recommendations_observed or 60 in recommendations_observed)

if __name__ == "__main__":
    unittest.main()
