#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/tests/test_phase1_engineering.py

Phase 1 Engineering Verification Test Suite:
1. Pressure-drop baseline and continuum scaling consistency (7.91 kPa).
2. Parasitic pumping power, energy, and carbon emissions non-zero accounting.
3. Economic cost parameters disambiguation ($3,500/event vs $35/MWh).
4. Model ablation and zero-leakage chronological split integrity.
"""

from pathlib import Path
import sys
import unittest

# Ensure application root is in sys.path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
import fouling_indicators
import sustainability_metrics
import cleaning_optimizer
import synthetic_generator
import preprocessor
import improved_model
import baseline_model


class TestPhase1EngineeringUpgrade(unittest.TestCase):
    """Test suite validating Phase 1 engineering corrections."""

    # -------------------------------------------------------------------------
    # Task 1: Pressure-Drop Reference & Dependent Calculations
    # -------------------------------------------------------------------------

    def test_01_clean_pressure_drop_darcy_weisbach(self):
        """Verify continuum Darcy-Weisbach clean tube bundle pressure drop is ~7.91 kPa."""
        calc_clean_dp = fouling_indicators.compute_clean_pressure_drop(
            mass_flow_kg_s=config.COOLING_WATER_MASS_FLOW_KG_S
        )
        self.assertAlmostEqual(calc_clean_dp, 7.91, places=1,
                               msg=f"Expected ~7.91 kPa clean pressure drop, got {calc_clean_dp:.3f} kPa")
        self.assertEqual(config.CLEAN_PRESSURE_DROP_KPA, 7.91,
                         msg="config.CLEAN_PRESSURE_DROP_KPA must equal 7.91 kPa")

    def test_02_fouled_vs_clean_pressure_drop_scaling(self):
        """Verify fouled pressure drop strictly exceeds clean baseline under identical conditions."""
        clean_dp, u_clean, _ = fouling_indicators.compute_hydraulic_pressure_drop(
            mass_flow_kg_s=config.COOLING_WATER_MASS_FLOW_KG_S,
            biofilm_thickness_m=0.0
        )
        fouled_dp, u_fouled, _ = fouling_indicators.compute_hydraulic_pressure_drop(
            mass_flow_kg_s=config.COOLING_WATER_MASS_FLOW_KG_S,
            biofilm_thickness_m=100e-6  # 100 micrometers biofilm
        )
        self.assertGreater(fouled_dp, clean_dp, "Fouled dP must exceed clean dP")
        self.assertGreater(u_fouled, u_clean, "Fouled velocity must increase due to cross-sectional constriction")
        self.assertGreater(fouled_dp / 1000.0, 9.0, "100 um biofilm should generate > 9.0 kPa dP")

    def test_03_parasitic_pumping_power_non_zero_when_fouled(self):
        """Verify parasitic pumping power and CO2 emissions are non-zero when fouled."""
        vol_flow = config.COOLING_WATER_MASS_FLOW_KG_S / config.WATER_DENSITY_KG_M3
        
        # When clean (excess = 0.0)
        p_clean = sustainability_metrics.compute_parasitic_pumping_power_kw(vol_flow, delta_P_excess_kPa=0.0)
        self.assertEqual(p_clean, 0.0)

        # When fouled (dP = 10.5 kPa -> excess = 2.59 kPa above 7.91 kPa baseline)
        excess_dp = 10.5 - config.CLEAN_PRESSURE_DROP_KPA
        p_fouled = sustainability_metrics.compute_parasitic_pumping_power_kw(vol_flow, delta_P_excess_kPa=excess_dp)
        self.assertGreater(p_fouled, 0.5, "Excess pumping power must be > 0.5 kW for 2.59 kPa surge")

        # Carbon emissions for 1,000 kWh excess
        co2_kg = sustainability_metrics.compute_carbon_emissions_kg(1000.0)
        self.assertAlmostEqual(co2_kg, 450.0, places=1)

    def test_04_annual_sustainability_accounting_realistic(self):
        """Verify annual sustainability accounting yields authentic non-zero energy and emissions."""
        df = synthetic_generator.generate_biofouling_dataset(hours=2000, seed=42, inject_cleanings=True)
        sust = sustainability_metrics.compute_total_sustainability_impact(df)
        
        self.assertGreater(sust["excess_pumping_kwh"], 100.0, "Excess pumping energy must be > 100 kWh over 2000h")
        self.assertGreater(sust["equivalent_co2_kg"], 40.0, "Carbon emissions must be > 40 kg CO2")
        self.assertTrue(sust["within_biocide_permit"], "Biocide should remain within permit limits")

    # -------------------------------------------------------------------------
    # Task 2: Cleaning Cost Mismatch Resolution
    # -------------------------------------------------------------------------

    def test_05_economic_constants_definitions(self):
        """Verify economic constants have distinct, explicitly documented values and units."""
        self.assertEqual(config.MECHANICAL_CLEANING_COST, 3500.0, "Mechanical cleaning cost must be $3,500.00/event")
        self.assertEqual(config.CHEMICAL_CLEANING_COST, 6500.0, "Chemical cleaning cost must be $6,500.00/event")
        self.assertEqual(config.THERMAL_PENALTY_COST_PER_MWH_LOSS, 35.0, "Thermal deficit penalty must be $35.00/MWh")
        self.assertEqual(config.ELECTRICITY_COST_PER_KWH, 0.12, "Electricity cost must be $0.12/kWh")
        self.assertEqual(config.DOWNTIME_COST_PER_HOUR, 250.0, "Downtime cost must be $250.00/h")

    # -------------------------------------------------------------------------
    # Task 3: Trained ML Model Integration
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Task 4: Model Validation and Ablation
    # -------------------------------------------------------------------------

    def test_10_chronological_split_integrity(self):
        """Verify temporal causality: max(t_train) < min(t_val) < min(t_test)."""
        df = synthetic_generator.generate_biofouling_dataset(hours=500, seed=42)
        train_df, val_df, test_df = preprocessor.chronological_split(df, 0.60, 0.20)
        
        self.assertLess(train_df["time_hours"].max(), val_df["time_hours"].min())
        self.assertLess(val_df["time_hours"].max(), test_df["time_hours"].min())

    def test_11_ablation_behavior(self):
        """Verify model evaluates successfully with and without overall HTC feature."""
        df = synthetic_generator.generate_biofouling_dataset(hours=2000, seed=42)
        train_df, val_df, test_df = preprocessor.chronological_split(df, 0.65, 0.15)
        
        # Default with U_overall_meas
        m_full = improved_model.ImprovedFoulingModel(n_estimators=30, max_depth=4)
        m_full.fit(train_df)
        res_full = m_full.evaluate(test_df)
        self.assertGreater(res_full["r2"], 0.85)

        # Without U_overall_meas
        m_ablated = improved_model.ImprovedFoulingModel(n_estimators=30, max_depth=4)
        m_ablated.pipeline.feature_cols = ["T_cool_in_meas", "delta_P_meas_kPa", "mass_flow_cool_meas"]
        m_ablated.fit(train_df)
        res_ablated = m_ablated.evaluate(test_df)
        self.assertGreater(res_ablated["r2"], 0.85)
        self.assertLess(res_ablated["rmse"], 0.0001)


if __name__ == "__main__":
    unittest.main()
