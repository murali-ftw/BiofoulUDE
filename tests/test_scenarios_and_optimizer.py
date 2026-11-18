#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/tests/test_scenarios_and_optimizer.py

Automated unit test suite validating the 10 controlled scenarios (A through J)
and critical optimizer sensitivity behavior.
"""

from pathlib import Path
import sys
import unittest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
import data_ingestion
import cleaning_optimizer
import fouling_indicators


class TestControlledScenariosAndOptimizer(unittest.TestCase):
    """Verify physical and optimizer behavior across Scenarios A through J."""

    def setUp(self):
        self.optimizer = cleaning_optimizer.CleaningIntervalOptimizer()

    def test_scenario_a_clean_baseline(self):
        """Scenario A: Clean heat exchanger at nominal design."""
        inp = {
            "T_cool_in_meas": 18.5,
            "T_cool_out_meas": 26.2,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 7.91,
            "time_hours": 100.0,
            "U_overall_meas": 2200.0,
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["derived_indicators"]["Rf_fouling_resistance"], 0.0)
        self.assertEqual(res["derived_indicators"]["bsi_score"], 0.0)

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            100,
            mass_flow_kg_s=220.0,
            operating_temp_c=18.5,
            current_rf=0.0,
            current_dp_kpa=7.91,
        )
        self.assertEqual(rec["status"], "OPTIMAL_FEASIBLE")
        self.assertFalse(rec["threshold_crossing"]["breached"])
        self.assertGreaterEqual(rec["recommended_interval_days"], 21)

    def test_scenario_b_mild_fouling(self):
        """Scenario B: Early mild fouling accumulation."""
        inp = {
            "T_cool_in_meas": 19.0,
            "T_cool_out_meas": 26.4,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 10.2,
            "time_hours": 800.0,
            "U_overall_meas": 1980.0,
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])
        self.assertGreater(res["derived_indicators"]["Rf_fouling_resistance"], 0.0)
        self.assertLess(res["derived_indicators"]["Rf_fouling_resistance"], 1e-4)

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            800,
            mass_flow_kg_s=220.0,
            operating_temp_c=19.0,
            current_rf=res["derived_indicators"]["Rf_fouling_resistance"],
            current_dp_kpa=10.2,
        )
        self.assertEqual(rec["status"], "OPTIMAL_FEASIBLE")
        self.assertTrue(rec["is_feasible"])

    def test_scenario_c_moderate_fouling_tightened_interval(self):
        """Scenario C: Moderate fouling approaches alert, causing interval tightening."""
        inp = {
            "T_cool_in_meas": 20.0,
            "T_cool_out_meas": 26.8,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 14.5,
            "time_hours": 2400.0,
            "U_overall_meas": 1650.0,
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])
        self.assertGreater(res["derived_indicators"]["Rf_fouling_resistance"], 1.0e-4)

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            2400,
            mass_flow_kg_s=220.0,
            operating_temp_c=20.0,
            current_rf=res["derived_indicators"]["Rf_fouling_resistance"],
            current_dp_kpa=14.5,
        )
        self.assertEqual(rec["status"], "OPTIMAL_FEASIBLE")
        # Recommended interval should tighten to <= 21 days
        self.assertLessEqual(rec["recommended_interval_days"], 21)
        self.assertTrue(rec["threshold_crossing"]["breached"])

    def test_scenario_d_severe_fouling_immediate_cleaning(self):
        """Scenario D: Severe fouling breaching thresholds triggers immediate cleaning (0 days)."""
        inp = {
            "T_cool_in_meas": 21.0,
            "T_cool_out_meas": 27.0,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 22.0,  # > 16.0 critical
            "time_hours": 4200.0,
            "U_overall_meas": 1350.0,
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            4200,
            mass_flow_kg_s=220.0,
            operating_temp_c=21.0,
            current_rf=res["derived_indicators"]["Rf_fouling_resistance"],
            current_dp_kpa=22.0,
        )
        self.assertEqual(rec["recommended_interval_days"], 0)
        self.assertIn("Immediate", rec["recommended_window_days"])
        self.assertEqual(rec["threshold_crossing"]["breach_days_relative"], 0.0)

    def test_scenario_e_hydraulic_surge_anomaly(self):
        """Scenario E: Isolated high delta-P hydraulic surge triggers immediate cleaning."""
        inp = {
            "T_cool_in_meas": 18.5,
            "T_cool_out_meas": 26.0,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 19.8,  # > 16.0 critical
            "time_hours": 1500.0,
            "U_overall_meas": 2050.0,
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            1500,
            mass_flow_kg_s=220.0,
            operating_temp_c=18.5,
            current_rf=res["derived_indicators"]["Rf_fouling_resistance"],
            current_dp_kpa=19.8,
        )
        self.assertEqual(rec["recommended_interval_days"], 0)
        self.assertEqual(rec["threshold_crossing"]["limiting_parameter"], "delta_P_meas_kPa")

    def test_scenario_f_severe_thermal_degradation(self):
        """Scenario F: Low HTC / high Rf triggers thermal limiting breach."""
        inp = {
            "T_cool_in_meas": 18.5,
            "T_cool_out_meas": 23.5,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 15.0,
            "time_hours": 3200.0,
            "U_overall_meas": 1200.0,  # Rf = 0.0003788 > 0.00030
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])
        rf = res["derived_indicators"]["Rf_fouling_resistance"]
        self.assertGreater(rf, config.CRITICAL_RF_M2K_W)

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            3200,
            mass_flow_kg_s=220.0,
            operating_temp_c=18.5,
            current_rf=rf,
            current_dp_kpa=15.0,
        )
        self.assertEqual(rec["recommended_interval_days"], 0)
        self.assertEqual(rec["threshold_crossing"]["limiting_parameter"], "Rf_fouling_resistance")

    def test_scenario_g_high_coolant_flow_rate(self):
        """Scenario G: Coolant flow rate +45% higher (320 kg/s) increases baseline pressure drop and shear."""
        inp = {
            "T_cool_in_meas": 18.5,
            "T_cool_out_meas": 24.2,
            "mass_flow_cool_meas": 320.0,
            "delta_P_meas_kPa": 17.5,
            "time_hours": 1200.0,
            "U_overall_meas": 2150.0,
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["validated_condition"]["mass_flow_cool_meas"], 320.0)

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            1200,
            mass_flow_kg_s=320.0,
            operating_temp_c=18.5,
            current_rf=res["derived_indicators"]["Rf_fouling_resistance"],
            current_dp_kpa=17.5,
        )
        self.assertEqual(rec["status"], "OPTIMAL_FEASIBLE")
        self.assertTrue(rec["is_feasible"])

    def test_scenario_h_recent_cleaning_reset(self):
        """Scenario H: Cleaning event resets induction delay and maintains low BSI."""
        inp = {
            "T_cool_in_meas": 18.5,
            "T_cool_out_meas": 26.1,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 8.35,
            "time_hours": 200.0,
            "U_overall_meas": 2120.0,
            "cleaning_event": 1,
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])
        self.assertLess(res["derived_indicators"]["bsi_score"], 15.0)

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            200,
            mass_flow_kg_s=220.0,
            operating_temp_c=18.5,
            current_rf=res["derived_indicators"]["Rf_fouling_resistance"],
            current_dp_kpa=8.35,
        )
        self.assertEqual(rec["status"], "OPTIMAL_FEASIBLE")
        self.assertFalse(rec["threshold_crossing"]["breached"])

    def test_scenario_i_missing_measurements_derivation(self):
        """Scenario I: Missing heat duty and U are derived from first-principles enthalpy balance."""
        inp = {
            "T_cool_in_meas": 18.5,
            "T_cool_out_meas": 26.2,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 10.5,
            "time_hours": 600.0,
        }
        res = data_ingestion.validate_manual_snapshot(inp)
        self.assertTrue(res["is_valid"])
        self.assertIn("heat_duty_MW", res["derived_indicators"])
        self.assertIn("U_overall_meas", res["derived_indicators"])
        self.assertEqual(res["provenance"]["heat_duty_MW"], "derived_calculation")
        self.assertEqual(res["provenance"]["U_overall_meas"], "derived_calculation")

        rec = self.optimizer.optimize_interval(
            res["derived_indicators"]["biofilm_thickness_um"],
            600,
            mass_flow_kg_s=220.0,
            operating_temp_c=18.5,
            current_rf=res["derived_indicators"]["Rf_fouling_resistance"],
            current_dp_kpa=10.5,
        )
        self.assertEqual(rec["status"], "OPTIMAL_FEASIBLE")

    def test_scenario_j_physical_violations_rejected(self):
        """Scenario J: Rejects thermodynamic violations (Tout <= Tin) and negative flow."""
        # J1: Thermodynamic violation
        bad_j1 = {
            "T_cool_in_meas": 28.0,
            "T_cool_out_meas": 22.0,
            "mass_flow_cool_meas": 220.0,
            "delta_P_meas_kPa": 8.5,
            "time_hours": 500.0,
        }
        res_j1 = data_ingestion.validate_manual_snapshot(bad_j1)
        self.assertFalse(res_j1["is_valid"])
        self.assertTrue(any("Thermodynamic violation" in e for e in res_j1["errors"]))

        # J2: Negative flow
        bad_j2 = {
            "T_cool_in_meas": 18.5,
            "T_cool_out_meas": 26.2,
            "mass_flow_cool_meas": -45.0,
            "delta_P_meas_kPa": 7.91,
            "time_hours": 500.0,
        }
        res_j2 = data_ingestion.validate_manual_snapshot(bad_j2)
        self.assertFalse(res_j2["is_valid"])
        self.assertTrue(any("mass_flow" in e.lower() for e in res_j2["errors"]))


if __name__ == "__main__":
    unittest.main()
