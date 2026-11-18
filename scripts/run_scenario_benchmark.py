#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/scripts/run_scenario_benchmark.py

Controlled Deterministic Benchmark for Scenarios A through J:
  Scenario A: Clean baseline (nominal operation)
  Scenario B: Mild fouling (early accumulation)
  Scenario C: Moderate fouling (approaching alert)
  Scenario D: Severe fouling (critical threshold breached)
  Scenario E: High ΔP (hydraulic anomaly / excessive flow resistance)
  Scenario F: Reduced U (severe thermal degradation)
  Scenario G: High flow (elevated cooling water mass flow rate)
  Scenario H: Recent cleaning (post-wash baseline with residual fouling)
  Scenario I: Missing measurements (boundary/incomplete data handling)
  Scenario J1: Thermodynamic violation (Tout <= Tin)
  Scenario J2: Negative mass flow rate

Outputs:
  - workspace/biofouling_ai_repaired/scenario_results.csv
  - Formatted terminal Markdown / ASCII comparative table
"""

import csv
import json
import math
from pathlib import Path
import sys

# Ensure repaired package is on import path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
import data_ingestion
import cleaning_optimizer
import fouling_indicators


def run_benchmark():
    optimizer = cleaning_optimizer.CleaningIntervalOptimizer()
    
    scenarios_def = [
        {
            "id": "A",
            "name": "Clean Baseline (Nominal Operation)",
            "description": "Clean heat exchanger at nominal design conditions",
            "inputs": {
                "T_cool_in_meas": 18.5,
                "T_cool_out_meas": 26.2,
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 7.91,
                "time_hours": 100.0,
                "U_overall_meas": 2200.0,
            },
            "is_valid_expected": True,
        },
        {
            "id": "B",
            "name": "Mild Fouling (Early Accumulation)",
            "description": "Early thin biofilm accumulation with slight HTC drop",
            "inputs": {
                "T_cool_in_meas": 19.0,
                "T_cool_out_meas": 26.4,
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 10.2,
                "time_hours": 800.0,
                "U_overall_meas": 1980.0,
            },
            "is_valid_expected": True,
        },
        {
            "id": "C",
            "name": "Moderate Fouling (Approaching Alert)",
            "description": "Mature biofilm approaching operational advisory thresholds",
            "inputs": {
                "T_cool_in_meas": 20.0,
                "T_cool_out_meas": 26.8,
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 14.5,
                "time_hours": 2400.0,
                "U_overall_meas": 1650.0,
            },
            "is_valid_expected": True,
        },
        {
            "id": "D",
            "name": "Severe Fouling (Critical Breach)",
            "description": "Critical biofilm growth breaching thermal and hydraulic limits",
            "inputs": {
                "T_cool_in_meas": 21.0,
                "T_cool_out_meas": 27.0,
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 22.0,
                "time_hours": 4200.0,
                "U_overall_meas": 1350.0,
            },
            "is_valid_expected": True,
        },
        {
            "id": "E",
            "name": "High ΔP Anomaly (Hydraulic Surge)",
            "description": "Isolated tube clogging / hydraulic flow restriction",
            "inputs": {
                "T_cool_in_meas": 18.5,
                "T_cool_out_meas": 26.0,
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 19.8,  # > 16.0 kPa critical
                "time_hours": 1500.0,
                "U_overall_meas": 2050.0,
            },
            "is_valid_expected": True,
        },
        {
            "id": "F",
            "name": "Reduced U (Severe Thermal Degradation)",
            "description": "Severe thermal deficit penalty with low overall HTC",
            "inputs": {
                "T_cool_in_meas": 18.5,
                "T_cool_out_meas": 23.5,
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 15.0,
                "time_hours": 3200.0,
                "U_overall_meas": 1200.0,  # Rf = 0.0003788 > 0.00030 crit
            },
            "is_valid_expected": True,
        },
        {
            "id": "G",
            "name": "High Flow Rate (+45% Surge)",
            "description": "Elevated coolant velocity (320 kg/s) increasing shear and baseline dP",
            "inputs": {
                "T_cool_in_meas": 18.5,
                "T_cool_out_meas": 24.2,
                "mass_flow_cool_meas": 320.0,
                "delta_P_meas_kPa": 17.5,
                "time_hours": 1200.0,
                "U_overall_meas": 2150.0,
            },
            "is_valid_expected": True,
        },
        {
            "id": "H",
            "name": "Recent Cleaning (Post-Wash Reset)",
            "description": "Post-mechanical wash condition with residual lag phase induction",
            "inputs": {
                "T_cool_in_meas": 18.5,
                "T_cool_out_meas": 26.1,
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 8.35,
                "time_hours": 200.0,
                "U_overall_meas": 2120.0,
                "cleaning_event": 1,
            },
            "is_valid_expected": True,
        },
        {
            "id": "I",
            "name": "Missing Measurements (Secondary Recovery)",
            "description": "Missing heat duty and U, derived from first-principles enthalpy balance",
            "inputs": {
                "T_cool_in_meas": 18.5,
                "T_cool_out_meas": 26.2,
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 10.5,
                "time_hours": 600.0,
                # heat_duty_MW and U_overall_meas omitted
            },
            "is_valid_expected": True,
        },
        {
            "id": "J1",
            "name": "Thermodynamic Violation (Tout <= Tin)",
            "description": "Cooling water temperature decreases across heated exchanger",
            "inputs": {
                "T_cool_in_meas": 28.0,
                "T_cool_out_meas": 22.0,  # Physical violation
                "mass_flow_cool_meas": 220.0,
                "delta_P_meas_kPa": 8.5,
                "time_hours": 500.0,
            },
            "is_valid_expected": False,
        },
        {
            "id": "J2",
            "name": "Negative Mass Flow Rate",
            "description": "Negative flow rate violating non-negativity constraint",
            "inputs": {
                "T_cool_in_meas": 18.5,
                "T_cool_out_meas": 26.2,
                "mass_flow_cool_meas": -45.0,  # Physical violation
                "delta_P_meas_kPa": 7.91,
                "time_hours": 500.0,
            },
            "is_valid_expected": False,
        },
    ]

    results = []

    for sc in scenarios_def:
        sc_id = sc["id"]
        sc_name = sc["name"]
        inp = sc["inputs"]
        
        # Step 1: Input validation via data ingestion engine
        val_res = data_ingestion.validate_manual_snapshot(inp)
        is_valid = val_res.get("is_valid", False)
        errors = val_res.get("errors", [])
        warnings = val_res.get("warnings", [])
        
        row = {
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "validation_status": "VALID" if is_valid else "REJECTED",
            "T_cool_in_c": inp.get("T_cool_in_meas"),
            "T_cool_out_c": inp.get("T_cool_out_meas"),
            "mass_flow_kg_s": inp.get("mass_flow_cool_meas"),
            "delta_P_meas_kPa": inp.get("delta_P_meas_kPa"),
            "U_overall_W_m2K": None,
            "Rf_m2K_W": None,
            "biofilm_um": None,
            "bsi_score": None,
            "error_or_warning": "; ".join(errors) if errors else ("; ".join(warnings) if warnings else "None"),
            "optimizer_status": "N/A (Rejected)",
            "recommended_interval_days": "N/A",
            "recommended_window": "N/A",
            "limiting_parameter": "N/A",
            "uncleaned_breach_days": "N/A",
            "total_projected_cost_usd": "N/A",
            "pump_excess_cost_usd": "N/A",
            "thermal_penalty_cost_usd": "N/A",
            "primary_rationale": "Snapshot rejected: " + "; ".join(errors) if errors else "None",
        }

        if is_valid:
            der = val_res.get("derived_indicators", {})
            val_cond = val_res.get("validated_condition", {})
            
            rf = der.get("Rf_fouling_resistance", 0.0)
            u_meas = der.get("U_overall_meas", config.U_CLEAN_DESIGN_W_M2K)
            biofilm_um = der.get("biofilm_thickness_um", 0.0)
            bsi = der.get("bsi_score", 0.0)
            h_op = int(val_cond.get("time_hours", 1000))
            flow = float(val_cond.get("mass_flow_cool_meas", 220.0))
            t_in = float(val_cond.get("T_cool_in_meas", 18.5))
            dp = float(val_cond.get("delta_P_meas_kPa", 7.91))
            
            row["U_overall_W_m2K"] = u_meas
            row["Rf_m2K_W"] = f"{rf:.7f}"
            row["biofilm_um"] = biofilm_um
            row["bsi_score"] = bsi

            # Step 2: Dynamic optimization run
            rec = optimizer.optimize_interval(
                current_biofilm_um=biofilm_um,
                current_hour=h_op,
                planning_horizon_days=60,
                mass_flow_kg_s=flow,
                operating_temp_c=t_in,
                current_rf=rf,
                current_dp_kpa=dp,
            )

            crossing = rec.get("threshold_crossing", {})
            best = rec.get("best_scenario", {})

            row["optimizer_status"] = rec.get("status", "UNKNOWN")
            row["recommended_interval_days"] = rec.get("recommended_interval_days")
            row["recommended_window"] = rec.get("recommended_window_days")
            row["limiting_parameter"] = crossing.get("limiting_parameter", "None")
            row["uncleaned_breach_days"] = crossing.get("breach_days_relative", "None")
            row["total_projected_cost_usd"] = best.get("total_projected_cost", 0.0)
            row["pump_excess_cost_usd"] = best.get("pump_energy_cost", 0.0)
            row["thermal_penalty_cost_usd"] = best.get("thermal_penalty_cost", 0.0)
            row["primary_rationale"] = rec.get("primary_reason", "")

        results.append(row)

    # Save to CSV
    csv_out_path = APP_DIR / "scenario_results.csv"
    fieldnames = [
        "scenario_id", "scenario_name", "validation_status", "T_cool_in_c", "T_cool_out_c",
        "mass_flow_kg_s", "delta_P_meas_kPa", "U_overall_W_m2K", "Rf_m2K_W", "biofilm_um",
        "bsi_score", "error_or_warning", "optimizer_status", "recommended_interval_days",
        "recommended_window", "limiting_parameter", "uncleaned_breach_days",
        "total_projected_cost_usd", "pump_excess_cost_usd", "thermal_penalty_cost_usd",
        "primary_rationale"
    ]
    with open(csv_out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Successfully generated scenario benchmark results at:\n  {csv_out_path}\n")

    # Print Formatted Markdown Comparative Table
    print("| ID | Scenario Name | Status | Rf (m²K/W) | ΔP (kPa) | BSI | Rec Days | Breach Limit | Uncleaned Breach | Total Cost ($) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        rf_str = r["Rf_m2K_W"] or "N/A"
        dp_str = f"{r['delta_P_meas_kPa']:.1f}" if r["delta_P_meas_kPa"] is not None else "N/A"
        bsi_str = f"{r['bsi_score']:.1f}" if r["bsi_score"] is not None else "N/A"
        breach_d = f"{r['uncleaned_breach_days']}d" if r['uncleaned_breach_days'] not in [None, "None", "N/A"] else "None"
        cost_str = f"${r['total_projected_cost_usd']:,}" if isinstance(r['total_projected_cost_usd'], (int, float)) else str(r['total_projected_cost_usd'])
        print(f"| **{r['scenario_id']}** | {r['scenario_name']} | {r['validation_status']} | {rf_str} | {dp_str} | {bsi_str} | **{r['recommended_interval_days']}** | {r['limiting_parameter']} | {breach_d} | {cost_str} |")

    return results


if __name__ == "__main__":
    run_benchmark()
