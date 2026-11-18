#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/tests/test_data_ingestion.py

Comprehensive Test Suite for Phase 2B External Data Ingestion & Condition-Based Cleaning:
1. Manual operating snapshot validation, physical indicator derivation, and physical range checks.
2. Rejection of thermodynamic violations (Tout <= Tin), negative flow, and out-of-range pressures.
3. CSV delimiter detection (comma, semicolon, tab) and column alias mapping.
4. Missing data identification without silent fabrication.
5. Duplicate and out-of-order timestamp detection.
6. Multi-factor Data Quality Score computation and rating classification (EXCELLENT, GOOD, DEGRADED, REJECTED).
7. Out-of-distribution condition handling and ML lag feature requirement warnings (< 25 rows / single snapshot).
8. Single snapshot historical rate caveat warnings.
9. Historical telemetry analysis (growth rates, cleaning discontinuities, time-to-threshold).
10. Dashboard API integration for modes, ingestion endpoints, provenance tracking, and dynamic recalculation.
"""

import json
from pathlib import Path
import sys
import threading
import time
import unittest
import urllib.request
import urllib.error

# Ensure application root is in sys.path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
import fouling_indicators
import data_ingestion
from data_ingestion import validate_manual_snapshot, CSVDataIngestor, PHYSICAL_RANGES
import historical_analyzer
import dashboard_server


class TestManualSnapshotIngestion(unittest.TestCase):
    """Test manual operating measurement snapshot validation and indicator derivation."""

    def test_valid_complete_manual_snapshot(self):
        """Test a fully specified, physically valid manual operating snapshot."""
        payload = {
            "T_cool_in_meas": 20.0,
            "T_cool_out_meas": 28.5,
            "mass_flow_kg_s": 420.0,
            "delta_P_meas_kPa": 18.5,
            "time_hours": 150.0,
            "U_overall_meas": 1750.0,
            "heat_duty_MW": 3.74,
            "timestamp": "2026-05-10T12:00:00",
        }
        res = validate_manual_snapshot(payload)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["errors"]), 0)
        self.assertIn("indicators", res)
        self.assertIn("provenance", res)

        ind = res["indicators"]
        self.assertGreater(ind["Rf_fouling_resistance"], 0.0)
        self.assertGreater(ind["U_overall_W_m2K"], 500.0)
        self.assertLess(ind["U_overall_W_m2K"], config.U_CLEAN_DESIGN_W_M2K + 100.0)
        self.assertGreater(ind["delta_P_ratio"], 1.0)
        self.assertGreaterEqual(ind["bsi_score"], 0.0)
        self.assertLessEqual(ind["bsi_score"], 100.0)
        self.assertGreater(ind["parasitic_pump_kw"], 0.0)

        # Provenance verification
        prov = res["provenance"]
        self.assertEqual(prov["T_cool_in_meas"], "direct_user_measurement")
        self.assertEqual(prov["mass_flow_kg_s"], "direct_user_measurement")
        self.assertEqual(prov["Rf_fouling_resistance"], "derived_calculation")

        # Explicit warning that single snapshot cannot establish transient growth rate
        self.assertIn("warnings", res)
        self.assertTrue(any("growth rate" in w.lower() or "transient" in w.lower() for w in res["warnings"]))

    def test_valid_minimal_manual_snapshot(self):
        """Test minimal valid manual input where heat duty is derived from enthalpy balance."""
        payload = {
            "T_cool_in_meas": 18.0,
            "T_cool_out_meas": 27.0,
            "mass_flow_kg_s": 400.0,
            "delta_P_meas_kPa": 12.0,
        }
        res = validate_manual_snapshot(payload)
        self.assertTrue(res["valid"])
        ind = res["indicators"]
        # Q = m_dot * Cp * delta_T = 400 * 4184 * 9 = 15.0624 MW
        expected_q = 400.0 * config.WATER_SPECIFIC_HEAT_J_KG_K * 9.0 / 1.0e6
        self.assertAlmostEqual(ind["heat_duty_MW"], expected_q, places=2)

    def test_thermodynamic_violation_rejection(self):
        """Reject cases where Tout <= Tin (cooling water cooling instead of heating)."""
        payload_equal = {
            "T_cool_in_meas": 25.0,
            "T_cool_out_meas": 25.0,
            "mass_flow_kg_s": 400.0,
            "delta_P_meas_kPa": 10.0,
        }
        res_eq = validate_manual_snapshot(payload_equal)
        self.assertFalse(res_eq["valid"])
        self.assertTrue(any("thermodynamic" in e.lower() or "t_cool_out" in e.lower() for e in res_eq["errors"]))

        payload_invert = {
            "T_cool_in_meas": 30.0,
            "T_cool_out_meas": 22.0,
            "mass_flow_kg_s": 400.0,
            "delta_P_meas_kPa": 10.0,
        }
        res_inv = validate_manual_snapshot(payload_invert)
        self.assertFalse(res_inv["valid"])

    def test_negative_mass_flow_rejection(self):
        """Reject non-positive mass flow rate."""
        payload = {
            "T_cool_in_meas": 20.0,
            "T_cool_out_meas": 28.0,
            "mass_flow_kg_s": -50.0,
            "delta_P_meas_kPa": 10.0,
        }
        res = validate_manual_snapshot(payload)
        self.assertFalse(res["valid"])
        self.assertTrue(any("mass_flow" in e for e in res["errors"]))

    def test_out_of_range_temperatures_and_pressures(self):
        """Reject extreme temperatures and pressures outside physical domain."""
        # Unrealistic high intake temperature
        res_temp = validate_manual_snapshot({
            "T_cool_in_meas": 65.0,  # Max is 45.0
            "T_cool_out_meas": 75.0,
            "mass_flow_kg_s": 400.0,
            "delta_P_meas_kPa": 15.0,
        })
        self.assertFalse(res_temp["valid"])

        # Unrealistic high pressure drop
        res_dp = validate_manual_snapshot({
            "T_cool_in_meas": 20.0,
            "T_cool_out_meas": 28.0,
            "mass_flow_kg_s": 400.0,
            "delta_P_meas_kPa": 450.0,  # Max is 200.0
        })
        self.assertFalse(res_dp["valid"])


class TestCSVDataIngestor(unittest.TestCase):
    """Test CSV ingestion, delimiter detection, alias mapping, and quality scoring."""

    def test_delimiter_detection(self):
        """Verify automatic sniffing of comma, semicolon, and tab delimiters."""
        comma_text = "time_hours,T_cool_in,T_cool_out\n1,20.0,28.0\n2,20.1,28.2\n"
        semi_text = "time_hours;T_cool_in;T_cool_out\n1;20.0;28.0\n2;20.1;28.2\n"
        tab_text = "time_hours\tT_cool_in\tT_cool_out\n1\t20.0\t28.0\n2\t20.1\t28.2\n"

        self.assertEqual(CSVDataIngestor.detect_delimiter(comma_text), ",")
        self.assertEqual(CSVDataIngestor.detect_delimiter(semi_text), ";")
        self.assertEqual(CSVDataIngestor.detect_delimiter(tab_text), "\t")

    def test_csv_preview_and_column_mapping(self):
        """Test preview generation and mapping of field aliases."""
        csv_content = (
            "hour;temp_in;temp_out;flow;dp\n"
            "0;18.5;26.8;410.0;8.5\n"
            "1;18.6;26.9;410.0;8.6\n"
            "2;18.7;27.0;412.0;8.7\n"
        )
        preview = CSVDataIngestor.generate_preview(csv_content, max_rows=2)
        self.assertTrue(preview["success"])
        self.assertEqual(preview["delimiter"], ";")
        self.assertEqual(preview["row_count"], 3)
        self.assertEqual(len(preview["preview_rows"]), 2)

        mapping = preview["suggested_mapping"]
        self.assertEqual(mapping.get("temp_in"), "T_cool_in_meas")
        self.assertEqual(mapping.get("temp_out"), "T_cool_out_meas")
        self.assertEqual(mapping.get("flow"), "mass_flow_cool_meas")
        self.assertEqual(mapping.get("dp"), "delta_P_meas_kPa")
        self.assertEqual(mapping.get("hour"), "time_hours")

    def test_missing_data_identification_without_fabrication(self):
        """Verify that missing observations are tallied, penalized, and NOT silently filled."""
        csv_content = (
            "time_hours,T_cool_in_meas,T_cool_out_meas,mass_flow_kg_s,delta_P_meas_kPa\n"
            "0,18.0,26.0,400.0,8.0\n"
            "1,,26.1,400.0,8.1\n"  # Missing T_in
            "2,18.2,,400.0,8.2\n"  # Missing T_out
            "3,18.3,26.3,,8.3\n"    # Missing flow
            "4,18.4,26.4,400.0,8.4\n"
        )
        res = CSVDataIngestor.ingest_and_validate(csv_content)
        self.assertTrue(res["success"])
        quality = res["quality_assessment"]
        self.assertGreater(quality["missing_data_count"], 0)
        self.assertLess(quality["completeness_score"], 100.0)
        self.assertIn("T_cool_in_meas", quality["missing_summary"])

    def test_duplicate_and_out_of_order_timestamps(self):
        """Verify detection of duplicate timestamps and non-monotonic ordering."""
        csv_content = (
            "time_hours,T_cool_in_meas,T_cool_out_meas,mass_flow_kg_s,delta_P_meas_kPa\n"
            "0,18.0,26.0,400.0,8.0\n"
            "1,18.1,26.1,400.0,8.1\n"
            "1,18.1,26.1,400.0,8.1\n"  # Duplicate hour 1
            "3,18.3,26.3,400.0,8.3\n"
            "2,18.2,26.2,400.0,8.2\n"  # Out of order: 2 after 3
        )
        res = CSVDataIngestor.ingest_and_validate(csv_content)
        self.assertTrue(res["success"])
        quality = res["quality_assessment"]
        self.assertEqual(quality["duplicate_timestamp_count"], 1)
        self.assertTrue(quality["temporal_anomalies_detected"])

    def test_quality_score_and_rating_classification(self):
        """Test calculation of Data Quality Score across tiers (EXCELLENT, GOOD, DEGRADED, REJECTED)."""
        # 1. Excellent dataset (50 regular valid rows)
        rows = ["time_hours,T_cool_in_meas,T_cool_out_meas,mass_flow_kg_s,delta_P_meas_kPa"]
        for h in range(50):
            rows.append(f"{h},20.0,28.5,420.0,{8.0 + 0.05*h:.2f}")
        csv_excellent = "\n".join(rows)

        res_exc = CSVDataIngestor.ingest_and_validate(csv_excellent)
        self.assertTrue(res_exc["success"])
        self.assertGreaterEqual(res_exc["quality_assessment"]["quality_score"], 90.0)
        self.assertEqual(res_exc["quality_assessment"]["quality_rating"], "EXCELLENT")
        self.assertTrue(res_exc["ml_sufficient_history"])

        # 2. Rejected dataset (< 3 valid rows or severe corruption)
        csv_corrupt = (
            "time_hours,T_cool_in_meas,T_cool_out_meas,mass_flow_kg_s,delta_P_meas_kPa\n"
            "0,100.0,10.0,-50.0,-20.0\n"
            "1,110.0,12.0,-50.0,-20.0\n"
        )
        res_rej = CSVDataIngestor.ingest_and_validate(csv_corrupt)
        self.assertFalse(res_rej["success"])
        self.assertEqual(res_rej["quality_assessment"]["quality_rating"], "REJECTED")

    def test_short_dataset_ml_lag_feature_warning(self):
        """Verify that datasets with < 25 rows set ml_sufficient_history=False with explicit explanation."""
        rows = ["time_hours,T_cool_in_meas,T_cool_out_meas,mass_flow_kg_s,delta_P_meas_kPa"]
        for h in range(10):  # Only 10 hours (< 24h lag requirement)
            rows.append(f"{h},20.0,28.5,420.0,9.0")
        csv_short = "\n".join(rows)

        res = CSVDataIngestor.ingest_and_validate(csv_short)
        self.assertTrue(res["success"])
        self.assertFalse(res["ml_sufficient_history"])
        self.assertTrue(any("24h lag" in w or "ml ensemble" in w.lower() for w in res["warnings"]))


class TestHistoricalAnalyzer(unittest.TestCase):
    """Test historical telemetry diagnostics, rate calculations, and cleaning discontinuity detection."""

    def test_analyzer_empty_dataframe(self):
        res = historical_analyzer.analyze_historical_telemetry(None)
        self.assertEqual(res["status"], "INSUFFICIENT_DATA")

    def test_growth_rates_and_cleaning_discontinuity(self):
        """Verify calculation of linear slopes and detection of cleaning step drops."""
        # Create 100-hour telemetry with a cleaning event at hour 50
        records = []
        clean_dp = config.CLEAN_PRESSURE_DROP_KPA
        u_clean = config.U_CLEAN_DESIGN_W_M2K

        for h in range(100):
            if h < 50:
                rf = 0.00005 + 0.000003 * h
                dp = clean_dp + 0.1 * h
            else:
                # Cleaning occurs at hour 50: Rf drops back down, dp drops back down
                rf = 0.00005 + 0.000003 * (h - 50)
                dp = clean_dp + 0.1 * (h - 50)

            u = 1.0 / (1.0 / u_clean + rf)
            records.append({
                "time_hours": float(h),
                "T_cool_in_meas": 20.0,
                "T_cool_out_meas": 28.0,
                "mass_flow_kg_s": 420.0,
                "delta_P_meas_kPa": dp,
                "Rf_fouling_resistance": rf,
                "U_overall_W_m2K": u,
                "delta_P_ratio": dp / clean_dp,
                "bsi_score": min(100.0, rf / 0.0005 * 100.0),
            })

        import pandas as pd
        df = pd.DataFrame(records)
        analysis = historical_analyzer.analyze_historical_telemetry(df)

        self.assertEqual(analysis["status"], "SUCCESS")
        self.assertEqual(analysis["total_records"], 100)
        self.assertGreater(analysis["progression_rates"]["dRf_dt_per_hour"], 0.0)
        self.assertGreater(analysis["progression_rates"]["dDeltaP_dt_kPa_per_day"], 0.0)

        # Cleaning discontinuity detection
        self.assertGreaterEqual(analysis["cleaning_history"]["detected_cleanings_count"], 1)
        event = analysis["cleaning_history"]["events"][0]
        self.assertEqual(event["event_hour"], 50.0)
        self.assertLess(event["delta_Rf"], 0.0)
        self.assertLess(event["delta_P_drop_kPa"], 0.0)
        self.assertGreater(event["thermal_recovery_pct"], 0.0)


import http.server

class TestDashboardAPIModesAndIngestion(unittest.TestCase):
    """Test dashboard server HTTP endpoints for multi-mode ingestion, provenance, and dynamic recalculation."""

    @classmethod
    def setUpClass(cls):
        """Start a dedicated test instance of the dashboard server on an ephemeral port."""
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), dashboard_server.BiofoulingDashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.server_thread.start()
        time.sleep(0.5)
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        """Reset dashboard to simulation mode before each test."""
        req = urllib.request.Request(
            f"{self.base_url}/api/mode/reset",
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            self.assertEqual(res.status, 200)

    def test_get_and_post_operating_mode(self):
        """Verify GET /api/mode and POST /api/mode endpoints."""
        # 1. Initial mode should be simulation
        with urllib.request.urlopen(f"{self.base_url}/api/mode", timeout=5) as res:
            data = json.loads(res.read().decode("utf-8"))
            self.assertEqual(data["mode"], "simulation")
            self.assertEqual(data["provenance_tag"], "synthetic_simulation")

        # 2. Switch mode to manual
        req = urllib.request.Request(
            f"{self.base_url}/api/mode",
            data=json.dumps({"mode": "manual"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read().decode("utf-8"))
            self.assertEqual(data["mode"], "manual")

    def test_post_manual_ingestion_updates_kpis_and_provenance(self):
        """Test POST /api/ingest/manual updates KPIs and sets provenance to direct_user_measurement."""
        payload = {
            "T_cool_in_meas": 22.5,
            "T_cool_out_meas": 31.0,
            "mass_flow_kg_s": 435.0,
            "delta_P_meas_kPa": 16.4,
            "time_hours": 320.0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/ingest/manual",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertEqual(data["mode"], "manual")
            self.assertIn("indicators", data)

        # Verify /api/kpis now reflects the manual measurements and provenance
        with urllib.request.urlopen(f"{self.base_url}/api/kpis", timeout=5) as res:
            kpis = json.loads(res.read().decode("utf-8"))
            self.assertEqual(kpis["operating_mode"], "manual")
            self.assertAlmostEqual(kpis["T_cool_in_meas"], 22.5, places=2)
            self.assertAlmostEqual(kpis["mass_flow_kg_s"], 435.0, places=2)
            self.assertAlmostEqual(kpis["delta_P_meas_kPa"], 16.4, places=2)

            prov = kpis.get("provenance_summary", {})
            self.assertEqual(prov.get("T_cool_in_meas"), "direct_user_measurement")
            self.assertEqual(prov.get("mass_flow_kg_s"), "direct_user_measurement")
            self.assertEqual(prov.get("Rf_fouling_resistance"), "derived_calculation")

    def test_post_manual_ingestion_invalid_rejected(self):
        """Reject invalid manual measurements with HTTP 400."""
        payload = {
            "T_cool_in_meas": 25.0,
            "T_cool_out_meas": 20.0,  # Inverted Tout < Tin
            "mass_flow_kg_s": 400.0,
            "delta_P_meas_kPa": 10.0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/ingest/manual",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_csv_preview_and_ingestion_workflow(self):
        """Test complete workflow: preview CSV, confirm ingestion, verify quality and historical analysis."""
        sample_csv = dashboard_server.generate_sample_industrial_csv()

        # 1. Test Preview endpoint
        req_prev = urllib.request.Request(
            f"{self.base_url}/api/ingest/csv/preview",
            data=json.dumps({"csv_text": sample_csv}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_prev, timeout=5) as res:
            prev_data = json.loads(res.read().decode("utf-8"))
            self.assertTrue(prev_data["success"])
            self.assertEqual(prev_data["row_count"], 120)

        # 2. Test Ingestion endpoint
        req_ing = urllib.request.Request(
            f"{self.base_url}/api/ingest/csv",
            data=json.dumps({"csv_text": sample_csv, "filename": "test_plant_telemetry.csv"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ing, timeout=5) as res:
            ing_data = json.loads(res.read().decode("utf-8"))
            self.assertTrue(ing_data["success"])
            self.assertEqual(ing_data["mode"], "csv")
            self.assertGreaterEqual(ing_data["quality_score"], 90.0)

        # 3. Test Quality endpoint
        with urllib.request.urlopen(f"{self.base_url}/api/ingest/quality", timeout=5) as res:
            q_data = json.loads(res.read().decode("utf-8"))
            self.assertEqual(q_data["mode"], "csv")
            self.assertEqual(q_data["quality"]["quality_rating"], "EXCELLENT")

        # 4. Test Historical Analysis endpoint
        with urllib.request.urlopen(f"{self.base_url}/api/historical/analysis", timeout=5) as res:
            hist_data = json.loads(res.read().decode("utf-8"))
            self.assertEqual(hist_data["status"], "SUCCESS")
            self.assertEqual(hist_data["total_records"], 120)
            self.assertIn("progression_rates", hist_data)
            self.assertIn("cleaning_history", hist_data)

    def test_sample_csv_download_endpoint(self):
        """Test GET /api/sample/csv returns downloadable CSV with content headers."""
        with urllib.request.urlopen(f"{self.base_url}/api/sample/csv", timeout=5) as res:
            self.assertEqual(res.status, 200)
            content_type = res.headers.get("Content-Type", "")
            self.assertIn("text/csv", content_type)
            content = res.read().decode("utf-8")
            self.assertIn("time_hours", content)
            self.assertIn("delta_P_meas_kPa", content)

    def test_condition_based_recommendation_in_manual_mode(self):
        """Verify POST /api/recommendation computes interval based on manual operating state."""
        # Ingest elevated fouling snapshot: Rf = 0.00015 m2K/W, delta_P = 18.0 kPa
        payload = {
            "T_cool_in_meas": 20.0,
            "T_cool_out_meas": 27.5,
            "mass_flow_kg_s": 420.0,
            "delta_P_meas_kPa": 18.0,
            "time_hours": 300.0,
        }
        req_man = urllib.request.Request(
            f"{self.base_url}/api/ingest/manual",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_man, timeout=5) as res:
            self.assertTrue(json.loads(res.read().decode("utf-8"))["success"])

        # Fetch recommendation
        req_rec = urllib.request.Request(
            f"{self.base_url}/api/recommendation",
            data=json.dumps({"cleaning_cost": 3500.0, "electricity_cost": 0.12}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_rec, timeout=5) as res:
            rec = json.loads(res.read().decode("utf-8"))
            self.assertEqual(rec["operating_mode"], "manual")
            self.assertEqual(rec["decision_engine"], "physics_based_optimizer")
            self.assertGreater(rec["recommended_interval_days"], 0)
            self.assertIn("candidate_evaluations", rec)
            self.assertIn("provenance", rec)
            self.assertEqual(rec["provenance"]["initial_condition"], "direct_user_measurement")


if __name__ == "__main__":
    unittest.main()
