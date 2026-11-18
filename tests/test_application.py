#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/tests/test_application.py

Internal Unit Test Suite for Repaired Biofouling AI Engineering Application.
Covers physical calculations, simulation reproducibility, ML forecasting,
cleaning optimization, sustainability metrics, and HTTP dashboard endpoints.
"""

import http.server
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
import urllib.request
import numpy as np
import pandas as pd

# Add application directory to sys.path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
import fouling_indicators
import synthetic_generator
import preprocessor
import baseline_model
import improved_model
import cleaning_optimizer
import sustainability_metrics
import dashboard_server


class TestRepairedBiofoulingApplication(unittest.TestCase):
    """Complete application unit test suite."""

    def test_01_config_constants(self):
        """Test geometric and physical configuration constants."""
        self.assertEqual(config.TUBE_INNER_DIAMETER_M, 0.021)
        self.assertEqual(config.NUMBER_OF_TUBES, 450)
        self.assertEqual(config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK, 0.60)
        self.assertEqual(config.U_CLEAN_DESIGN_W_M2K, 2200.0)
        self.assertTrue(config.TOTAL_HEAT_TRANSFER_AREA_M2 > 100.0)

    def test_02_thermal_resistance(self):
        """Test thermal resistance calculation Rf = delta_f / k_biofilm."""
        delta_f = 120e-6  # 120 um
        k_bio = 0.60
        rf = fouling_indicators.compute_thermal_resistance(delta_f, k_bio)
        self.assertAlmostEqual(rf, 0.00020, places=6)

        u = fouling_indicators.compute_degraded_U(config.U_CLEAN_DESIGN_W_M2K, rf)
        expected_u = 2200.0 / (1.0 + 2200.0 * 0.00020)
        self.assertAlmostEqual(u, expected_u, delta=0.5)

    def test_03_hydraulic_pressure_drop(self):
        """Test continuum Darcy-Weisbach + Haaland hydraulic calculations."""
        # Clean state
        dp_clean, u_clean, f_clean = fouling_indicators.compute_hydraulic_pressure_drop(
            mass_flow_kg_s=config.COOLING_WATER_MASS_FLOW_KG_S,
            biofilm_thickness_m=0.0
        )
        self.assertTrue(1.0 <= u_clean <= 2.5)
        self.assertTrue(0.015 <= f_clean <= 0.035)

        # Fouled state (200 um)
        dp_fouled, u_fouled, f_fouled = fouling_indicators.compute_hydraulic_pressure_drop(
            mass_flow_kg_s=config.COOLING_WATER_MASS_FLOW_KG_S,
            biofilm_thickness_m=200e-6
        )
        self.assertGreater(u_fouled, u_clean)
        self.assertGreater(f_fouled, f_clean)
        self.assertGreater(dp_fouled, dp_clean * 1.15)

    def test_04_bsi_composite_score(self):
        """Test Biofouling Severity Index normalization."""
        bsi_clean = fouling_indicators.compute_biofouling_severity_index(0.0, 1.0)
        bsi_severe = fouling_indicators.compute_biofouling_severity_index(0.0005, 1.8)

        self.assertTrue(0.0 <= bsi_clean <= 5.0)
        self.assertTrue(70.0 <= bsi_severe <= 100.0)

    def test_05_ratkowsky_kinetics(self):
        """Test cardinal temperature growth multiplier."""
        mu_dormant = synthetic_generator.ratkowsky_growth_factor(5.0)
        mu_opt = synthetic_generator.ratkowsky_growth_factor(28.0)
        mu_dead = synthetic_generator.ratkowsky_growth_factor(48.0)

        self.assertEqual(mu_dormant, 0.0)
        self.assertGreater(mu_opt, 0.80)
        self.assertEqual(mu_dead, 0.0)

    def test_06_synthetic_generation_and_reproducibility(self):
        """Test dataset shape, columns, and seed locking."""
        df1 = synthetic_generator.generate_biofouling_dataset(hours=200, seed=123)
        df2 = synthetic_generator.generate_biofouling_dataset(hours=200, seed=123)

        self.assertEqual(len(df1), 200)
        self.assertIn("Rf_fouling_resistance", df1.columns)
        self.assertIn("delta_P_meas_kPa", df1.columns)
        self.assertIn("U_overall_meas", df1.columns)

        # Bitwise identical numeric comparison
        num_diff = np.max(np.abs(df1["Rf_fouling_resistance"].values - df2["Rf_fouling_resistance"].values))
        self.assertEqual(num_diff, 0.0)

    def test_07_chronological_splitting(self):
        """Test strict non-overlapping temporal split."""
        df = synthetic_generator.generate_biofouling_dataset(hours=1000, seed=42)
        train_df, val_df, test_df = preprocessor.chronological_split(df, 0.6, 0.2)

        self.assertLess(train_df["time_hours"].max(), val_df["time_hours"].min())
        self.assertLess(val_df["time_hours"].max(), test_df["time_hours"].min())

    def test_08_model_forecasting(self):
        """Test baseline vs improved model training and metrics."""
        df = synthetic_generator.generate_biofouling_dataset(hours=1500, seed=42)
        train_df, _, test_df = preprocessor.chronological_split(df, 0.7, 0.1)

        base = baseline_model.BaselineFoulingModel()
        base.fit(train_df)
        base_eval = base.evaluate(test_df)
        self.assertTrue(base_eval["rmse"] < 0.0004)

        ml = improved_model.ImprovedFoulingModel(n_estimators=30, random_state=42)
        ml.fit(train_df)
        ml_eval = ml.evaluate(test_df)
        self.assertLessEqual(ml_eval["rmse"], base_eval["rmse"] * 1.05)

    def test_09_cleaning_optimization(self):
        """Test Condition-Based Optimization cost reduction."""
        df = synthetic_generator.generate_biofouling_dataset(hours=2000, seed=42, inject_cleanings=False)
        opt = cleaning_optimizer.CleaningScheduleOptimizer()

        res_fixed = opt.evaluate_fixed_calendar_schedule(df, interval_days=30)
        res_cbo = opt.evaluate_condition_based_schedule(df)

        self.assertIn("total_cost", res_fixed)
        self.assertIn("total_cost", res_cbo)
        self.assertLessEqual(res_cbo["total_cost"], res_fixed["total_cost"] * 1.05)

    def test_10_sustainability_metrics(self):
        """Test parasitic pumping power and CO2 calculations."""
        kw = sustainability_metrics.compute_parasitic_pumping_power_kw(0.22, 30.0, 0.75)
        self.assertAlmostEqual(kw, 8.8, delta=0.2)

        co2 = sustainability_metrics.compute_carbon_emissions_kg(10000.0, 0.45)
        self.assertEqual(co2, 4500.0)

        biocide = sustainability_metrics.track_biocide_discharge(5, 12.0, 100.0)
        self.assertEqual(biocide["total_biocide_kg"], 60.0)
        self.assertTrue(biocide["within_permit_limit"])

    def test_11_dashboard_server_endpoints(self):
        """Test local HTTP server status and KPI endpoints."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = http.server.HTTPServer(("127.0.0.1", port), dashboard_server.BiofoulingDashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.3)

        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                html = resp.read().decode("utf-8")
                self.assertIn("<svg", html.lower())

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data.get("status"), "ONLINE")

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/kpis", timeout=3) as resp:
                self.assertEqual(resp.status, 200)
                kpis = json.loads(resp.read().decode("utf-8"))
                self.assertIn("Rf_fouling_resistance", kpis)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
