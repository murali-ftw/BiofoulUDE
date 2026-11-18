#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_manual_sensitivity_experiment.py
============================================
Executes controlled manual sensitivity benchmark across Cases 1 to 8 and boundary checks.
Exports results to manual_sensitivity_results.csv and prints markdown table.
"""

import csv
import os
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import config
import data_ingestion
import cleaning_optimizer
import fouling_indicators


def run_experiment():
    opt = cleaning_optimizer.CleaningIntervalOptimizer()

    test_matrix = [
        # Case 1: Clean Baseline
        {
            "case_id": "1",
            "group": "Standard Cases",
            "name": "Clean Baseline (Nominal Design)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 7.91, "time_hours": 100.0, "U_overall_meas": 2200.0},
            "horizon": 60,
        },
        # Case 2: Mild Fouling
        {
            "case_id": "2",
            "group": "Standard Cases",
            "name": "Mild Fouling (Early Accumulation)",
            "input": {"T_cool_in_meas": 19.0, "T_cool_out_meas": 26.4, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 10.2, "time_hours": 800.0, "U_overall_meas": 1980.0},
            "horizon": 60,
        },
        # Case 3: Moderate Fouling
        {
            "case_id": "3",
            "group": "Standard Cases",
            "name": "Moderate Fouling (Approaching Alert)",
            "input": {"T_cool_in_meas": 20.0, "T_cool_out_meas": 26.8, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 14.5, "time_hours": 2400.0, "U_overall_meas": 1650.0},
            "horizon": 60,
        },
        # Case 4: Severe Fouling (Breach)
        {
            "case_id": "4",
            "group": "Standard Cases",
            "name": "Severe Fouling (Critical Breach)",
            "input": {"T_cool_in_meas": 21.0, "T_cool_out_meas": 27.0, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 22.0, "time_hours": 4200.0, "U_overall_meas": 1350.0},
            "horizon": 60,
        },
        # Case 5: Hydraulic Gradient (U omitted, Darcy-Weisbach inversion)
        {
            "case_id": "5A",
            "group": "Hydraulic Gradient (U Omitted)",
            "name": "ΔP = 7.91 kPa (Clean Baseline)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 7.91, "time_hours": 1500.0},
            "horizon": 60,
        },
        {
            "case_id": "5B",
            "group": "Hydraulic Gradient (U Omitted)",
            "name": "ΔP = 9.50 kPa (Early Hydraulic Rise)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 9.50, "time_hours": 1500.0},
            "horizon": 60,
        },
        {
            "case_id": "5C",
            "group": "Hydraulic Gradient (U Omitted)",
            "name": "ΔP = 11.50 kPa (Moderate Hydraulic Loss)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 11.50, "time_hours": 1500.0},
            "horizon": 60,
        },
        {
            "case_id": "5D",
            "group": "Hydraulic Gradient (U Omitted)",
            "name": "ΔP = 14.50 kPa (High Resistance)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 14.50, "time_hours": 1500.0},
            "horizon": 60,
        },
        {
            "case_id": "5E",
            "group": "Hydraulic Gradient (U Omitted)",
            "name": "ΔP = 15.80 kPa (Near-Critical Hydraulic Limit)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 15.80, "time_hours": 1500.0},
            "horizon": 60,
        },
        {
            "case_id": "5F",
            "group": "Hydraulic Gradient (U Omitted)",
            "name": "ΔP = 17.50 kPa (Critical Surge Breach)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 17.50, "time_hours": 1500.0},
            "horizon": 60,
        },
        # Case 6: Thermal Gradient (U Specified)
        {
            "case_id": "6A",
            "group": "Thermal Gradient (U Specified)",
            "name": "U = 2200 W/m²K (Design Clean)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 8.0, "time_hours": 1500.0, "U_overall_meas": 2200.0},
            "horizon": 60,
        },
        {
            "case_id": "6B",
            "group": "Thermal Gradient (U Specified)",
            "name": "U = 1950 W/m²K (Mild Thermal Loss)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 9.5, "time_hours": 1500.0, "U_overall_meas": 1950.0},
            "horizon": 60,
        },
        {
            "case_id": "6C",
            "group": "Thermal Gradient (U Specified)",
            "name": "U = 1650 W/m²K (Moderate Thermal Loss)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 11.5, "time_hours": 1500.0, "U_overall_meas": 1650.0},
            "horizon": 60,
        },
        {
            "case_id": "6D",
            "group": "Thermal Gradient (U Specified)",
            "name": "U = 1380 W/m²K (Near-Critical Summer Fouling)",
            "input": {"T_cool_in_meas": 23.5, "T_cool_out_meas": 31.0, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 13.0, "time_hours": 3000.0, "U_overall_meas": 1380.0},
            "horizon": 60,
        },
        {
            "case_id": "6E",
            "group": "Thermal Gradient (U Specified)",
            "name": "U = 1300 W/m²K (Severe Thermal Loss)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 13.5, "time_hours": 1500.0, "U_overall_meas": 1300.0},
            "horizon": 60,
        },
        {
            "case_id": "6F",
            "group": "Thermal Gradient (U Specified)",
            "name": "U = 1100 W/m²K (Near-Critical Thermal Limit)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 14.5, "time_hours": 1500.0, "U_overall_meas": 1100.0},
            "horizon": 60,
        },
        {
            "case_id": "6G",
            "group": "Thermal Gradient (U Specified)",
            "name": "U = 950 W/m²K (Critical Thermal Breach)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 15.0, "time_hours": 1500.0, "U_overall_meas": 950.0},
            "horizon": 60,
        },
        # Case 7: Flow Variation
        {
            "case_id": "7A",
            "group": "Flow Variation",
            "name": "Flow = 160 kg/s (Low Throughput)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 27.5, "mass_flow_cool_meas": 160.0, "delta_P_meas_kPa": 5.8, "time_hours": 1000.0},
            "horizon": 60,
        },
        {
            "case_id": "7B",
            "group": "Flow Variation",
            "name": "Flow = 220 kg/s (Nominal Design)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 7.91, "time_hours": 1000.0},
            "horizon": 60,
        },
        {
            "case_id": "7C",
            "group": "Flow Variation",
            "name": "Flow = 320 kg/s (High Flow Surge)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 24.2, "mass_flow_cool_meas": 320.0, "delta_P_meas_kPa": 16.5, "time_hours": 1000.0},
            "horizon": 60,
        },
        # Case 8: Recent Cleaning / Post-Wash Reset
        {
            "case_id": "8",
            "group": "Standard Cases",
            "name": "Recent Cleaning (Post-Wash Reset)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 8.10, "time_hours": 48.0, "U_overall_meas": 2160.0},
            "horizon": 60,
        },
        # Boundary Checks: Demonstrating 30d and 60d
        {
            "case_id": "B1",
            "group": "Boundary Checks",
            "name": "Winter Cold Water (Hour 0, T=10°C, Low Kinetics)",
            "input": {"T_cool_in_meas": 10.0, "T_cool_out_meas": 17.5, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 7.91, "time_hours": 0.0, "U_overall_meas": 2200.0},
            "horizon": 60,
        },
        {
            "case_id": "B2",
            "group": "Boundary Checks",
            "name": "Spring Transition (Hour 1000, T=14°C)",
            "input": {"T_cool_in_meas": 14.0, "T_cool_out_meas": 21.7, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 7.91, "time_hours": 1000.0, "U_overall_meas": 2200.0},
            "horizon": 60,
        },
        {
            "case_id": "B3",
            "group": "Boundary Checks",
            "name": "Late Autumn (Hour 7000, T=13°C)",
            "input": {"T_cool_in_meas": 13.0, "T_cool_out_meas": 20.7, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 7.91, "time_hours": 7000.0, "U_overall_meas": 2200.0},
            "horizon": 60,
        },
        {
            "case_id": "B4",
            "group": "Boundary Checks",
            "name": "Elevated Cleaning Cost ($6,500/event)",
            "input": {"T_cool_in_meas": 18.5, "T_cool_out_meas": 26.2, "mass_flow_cool_meas": 220.0, "delta_P_meas_kPa": 7.91, "time_hours": 6000.0, "U_overall_meas": 2200.0},
            "horizon": 60,
            "cleaning_cost": 6500.0,
        },
    ]

    records = []
    print("\n" + "=" * 105)
    print("PHASE 3.1: MANUAL OPERATING SENSITIVITY BENCHMARK EXPERIMENT")
    print("=" * 105)
    print(f"{'Case':<5} | {'Group':<22} | {'Name':<35} | {'Rf (m²K/W)':<11} | {'ΔP (kPa)':<8} | {'Biofilm':<7} | {'Rec Days':<8} | {'Status':<12}")
    print("-" * 105)

    for item in test_matrix:
        cid = item["case_id"]
        group = item["group"]
        name = item["name"]
        inp = item["input"]
        horizon = item.get("horizon", 60)
        c_clean = item.get("cleaning_cost", config.MECHANICAL_CLEANING_COST)

        # 1. Validate manual snapshot
        val = data_ingestion.validate_manual_snapshot(inp)
        ind = val["derived_indicators"]

        m_dot = float(inp.get("mass_flow_cool_meas", inp.get("mass_flow_kg_s", 220.0)))
        t_in = float(inp.get("T_cool_in_meas", 18.5))
        hour = int(inp.get("time_hours", 1000))
        rf = ind["Rf_fouling_resistance"]
        dp = inp["delta_P_meas_kPa"]
        bf_um = ind["biofilm_thickness_um"]
        u_val = ind["U_overall_meas"]

        # 2. Run optimizer
        if c_clean != config.MECHANICAL_CLEANING_COST:
            custom_opt = cleaning_optimizer.CleaningIntervalOptimizer(cleaning_cost=c_clean)
            rec = custom_opt.optimize_interval(
                bf_um, hour, planning_horizon_days=horizon, mass_flow_kg_s=m_dot,
                operating_temp_c=t_in, current_rf=rf, current_dp_kpa=dp
            )
        else:
            rec = opt.optimize_interval(
                bf_um, hour, planning_horizon_days=horizon, mass_flow_kg_s=m_dot,
                operating_temp_c=t_in, current_rf=rf, current_dp_kpa=dp
            )

        rec_days = rec["recommended_interval_days"]
        status = rec["status"]
        best_cost = rec.get("best_scenario", {}).get("total_projected_cost", 0.0)
        crossing = rec.get("threshold_crossing", {})
        breach_days = crossing.get("breach_days_relative", "N/A")
        limiting_param = crossing.get("limiting_parameter", "None")

        rec_row = {
            "case_id": cid,
            "group": group,
            "name": name,
            "Tin_C": t_in,
            "Tout_C": inp.get("T_cool_out_meas", 26.2),
            "mass_flow_kg_s": m_dot,
            "delta_P_kPa": dp,
            "U_W_m2K": u_val,
            "Rf_m2K_W": rf,
            "biofilm_um": bf_um,
            "recommended_interval_days": rec_days,
            "status": status,
            "total_projected_cost": best_cost,
            "uncleaned_breach_days": breach_days,
            "limiting_parameter": limiting_param,
            "provenance_U": val["provenance"].get("U_overall_meas", "derived_calculation"),
            "provenance_Rf": val["provenance"].get("Rf_fouling_resistance", "derived_calculation"),
        }
        records.append(rec_row)

        print(f"{cid:<5} | {group:<22} | {name:<35} | {rf:<11.7f} | {dp:<8.2f} | {bf_um:<7.1f} | {rec_days:<8} | {status:<12}")

    print("=" * 105)

    # Save to CSV
    csv_path = APP_DIR / "manual_sensitivity_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[OK] Successfully wrote {len(records)} test rows to: {csv_path}")

    # Summary of distinct recommendations
    rec_counts = {}
    for r in records:
        d = r["recommended_interval_days"]
        rec_counts[d] = rec_counts.get(d, 0) + 1
    print("\nRecommendation Frequency Distribution:")
    for d, cnt in sorted(rec_counts.items()):
        print(f"  - {d:>2} days: {cnt:>2} cases")

    return records


if __name__ == "__main__":
    run_experiment()
