#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/tests/test_cleaning_interval_optimizer.py

Focused Unit Test Suite for Predictive Cleaning Interval Optimizer.
Tests:
1. Candidate interval cost evaluations (forward re-simulation)
2. Status-quo threshold crossing calculation
3. Lowest-cost feasible candidate selection
4. Infeasible condition detection and emergency intervention reporting
5. Cost sensitivity (cleaning cost vs thermal/electricity penalty)
6. Cleaning recovery dynamics & physical reset
7. Deterministic reproducibility
8. Statistical uncertainty bounds
"""

from pathlib import Path
import sys
import unittest
import numpy as np

# Add application directory to sys.path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
import fouling_indicators
import synthetic_generator
import cleaning_optimizer
import improved_model


class TestCleaningIntervalOptimizer(unittest.TestCase):
    """Test suite for the predictive cleaning interval optimizer."""

    def setUp(self):
        self.optimizer = cleaning_optimizer.CleaningIntervalOptimizer()

    def test_01_candidate_scenario_evaluation(self):
        """Verify scenario evaluation calculates all required cost and physical components."""
        scenario = self.optimizer.evaluate_scenario(
            current_biofilm_um=100.0,
            current_hour=3000,
            candidate_interval_days=14,
            planning_horizon_days=60,
        )
        self.assertIn("candidate_interval_days", scenario)
        self.assertIn("candidate_interval_hours", scenario)
        self.assertIn("is_feasible", scenario)
        self.assertIn("intervention_count", scenario)
        self.assertIn("cleaning_cost", scenario)
        self.assertIn("downtime_cost", scenario)
        self.assertIn("pump_energy_kwh", scenario)
        self.assertIn("pump_energy_cost", scenario)
        self.assertIn("thermal_deficit_mwh", scenario)
        self.assertIn("thermal_penalty_cost", scenario)
        self.assertIn("total_projected_cost", scenario)
        self.assertIn("max_rf", scenario)
        self.assertIn("min_thermal_efficiency", scenario)

        # Total cost must equal sum of the 4 cost components
        expected_total = round(
            scenario["cleaning_cost"] + scenario["downtime_cost"] + scenario["pump_energy_cost"] + scenario["thermal_penalty_cost"],
            2
        )
        self.assertAlmostEqual(scenario["total_projected_cost"], expected_total, places=1)
        self.assertEqual(scenario["candidate_interval_hours"], 14 * 24)
        self.assertTrue(scenario["is_feasible"])

    def test_02_threshold_crossing_detection(self):
        """Verify status quo threshold crossing accurately flags breached metric and time."""
        crossing = self.optimizer.find_uncleaned_threshold_crossing(
            current_biofilm_um=100.0,
            current_hour=3000,
            max_lookahead_hours=2160,
        )
        self.assertTrue(crossing["breached"])
        self.assertIsNotNone(crossing["breach_hour_relative"])
        self.assertIsNotNone(crossing["breach_days_relative"])
        self.assertIn(crossing["limiting_parameter"], ["Rf_fouling_resistance", "delta_P_meas_kPa", "bsi_score"])
        self.assertGreater(crossing["breach_days_relative"], 10.0)
        self.assertLess(crossing["breach_days_relative"], 60.0)

    def test_03_lowest_cost_feasible_selection(self):
        """Verify optimizer picks the lowest-cost feasible candidate rather than simply the lowest cleaning cost."""
        res = self.optimizer.optimize_interval(
            current_biofilm_um=100.0,
            current_hour=3000,
            planning_horizon_days=60,
            candidates=[0, 7, 14, 21, 30, 45, 60],
        )
        self.assertEqual(res["status"], "OPTIMAL_FEASIBLE")
        self.assertTrue(res["is_feasible"])
        rec_days = res["recommended_interval_days"]
        self.assertTrue(rec_days in [7, 14, 21, 30])

        best_cost = res["best_scenario"]["total_projected_cost"]
        for s in res["all_scenarios"]:
            if s["is_feasible"]:
                self.assertLessEqual(best_cost, s["total_projected_cost"])

    def test_04_infeasible_condition_handling(self):
        """Verify optimizer flags infeasible state when fouling breaches constraints before cleaning."""
        # Critical initial fouling of 185 um (Rf = 185e-6/0.60 = 0.000308, above 0.00030 limit)
        res = self.optimizer.optimize_interval(
            current_biofilm_um=185.0,
            current_hour=3000,
            planning_horizon_days=60,
            candidates=[7, 14, 21, 30],
        )
        self.assertFalse(res["is_feasible"])
        self.assertIn("INFEASIBLE", res["status"])
        self.assertEqual(res["recommended_interval_days"], 0)
        self.assertIn("Immediate", res["recommended_window_days"])
        self.assertIn("Immediate", res["primary_reason"])

    def test_05_cost_sensitivity(self):
        """Verify recommendation shifts based on cleaning cost vs operational penalty."""
        # High cleaning cost ($9,000) makes frequent cleanings uneconomical -> shifts to longer interval
        opt_expensive = cleaning_optimizer.CleaningIntervalOptimizer(cleaning_cost=9000.0)
        res_exp = opt_expensive.optimize_interval(current_biofilm_um=100.0, current_hour=3000, planning_horizon_days=60)

        # Low cleaning cost ($800) makes frequent cleanings economical -> shifts to shorter interval
        opt_cheap = cleaning_optimizer.CleaningIntervalOptimizer(cleaning_cost=800.0)
        res_cheap = opt_cheap.optimize_interval(current_biofilm_um=100.0, current_hour=3000, planning_horizon_days=60)

        self.assertGreater(res_exp["recommended_interval_days"], res_cheap["recommended_interval_days"])

    def test_06_cleaning_recovery_reset(self):
        """Verify cleaning events reset biofilm thickness according to physical efficiency."""
        initial_thickness = 150e-6  # 150 um
        # Mechanical cleaning (90% removal efficiency)
        residual_mech = synthetic_generator.apply_cleaning_reset(initial_thickness, "Mechanical_Balls")
        self.assertAlmostEqual(residual_mech, 15e-6, delta=1e-6)

        # Chemical shock (95% removal efficiency)
        residual_chem = synthetic_generator.apply_cleaning_reset(initial_thickness, "Chemical_Biocide")
        self.assertAlmostEqual(residual_chem, 7.5e-6, delta=1e-6)

    def test_07_deterministic_output(self):
        """Verify optimizer produces deterministic bitwise identical output given identical inputs."""
        res1 = self.optimizer.optimize_interval(current_biofilm_um=110.0, current_hour=2500, planning_horizon_days=60)
        res2 = self.optimizer.optimize_interval(current_biofilm_um=110.0, current_hour=2500, planning_horizon_days=60)

        self.assertEqual(res1["recommended_interval_days"], res2["recommended_interval_days"])
        self.assertEqual(res1["best_scenario"]["total_projected_cost"], res2["best_scenario"]["total_projected_cost"])
        self.assertEqual(res1["primary_reason"], res2["primary_reason"])

    def test_08_uncertainty_bounds(self):
        """Verify ML model calculates statistical uncertainty bounds without negative bounds."""
        model = improved_model.ImprovedFoulingModel(n_estimators=20, random_state=42)
        df = synthetic_generator.generate_biofouling_dataset(hours=500, seed=42)
        model.fit(df)

        preds, lower, upper = model.predict_with_uncertainty(df.tail(24))
        self.assertEqual(len(preds), 24)
        self.assertTrue(np.all(lower <= preds))
        self.assertTrue(np.all(preds <= upper))
        self.assertTrue(np.all(lower >= 0.0))

if __name__ == "__main__":
    unittest.main()
