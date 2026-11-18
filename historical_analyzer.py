#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/historical_analyzer.py

Historical Fouling Diagnostics & Trend Analysis Engine.
Analyzes ingested time-series operating data:
  - Pressure-drop growth rate (kPa/day, kPa/100h)
  - Thermal resistance fouling rate (dRf/dt in m²·K/W per day)
  - Thermal efficiency degradation trend (% / 30 days)
  - Identification of past cleaning interventions and recovery efficiency
  - Comprehensive diagnostic summary metrics for condition-based monitoring
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

import config
import fouling_indicators


def analyze_historical_telemetry(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform deep diagnostic analysis on an ingested historical time-series dataset.
    Computes fouling progression rates, thermal degradation slopes,
    detected historical cleaning events, and condition summary metrics.
    """
    if df is None or len(df) == 0:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "Dataset is empty.",
            "data_points": 0,
        }

    total_points = len(df)
    clean_dp = config.CLEAN_PRESSURE_DROP_KPA
    u_clean = config.U_CLEAN_DESIGN_W_M2K

    # Check temporal span
    if "time_hours" in df.columns:
        h_start = float(df["time_hours"].iloc[0])
        h_end = float(df["time_hours"].iloc[-1])
        span_hours = max(1.0, h_end - h_start)
    else:
        span_hours = float(total_points)

    span_days = span_hours / 24.0

    # Ensure required fields exist or compute fallback
    rf_series = df.get("Rf_fouling_resistance")
    if rf_series is None or rf_series.isna().all():
        rf_arr = np.zeros(total_points)
    else:
        rf_arr = pd.to_numeric(rf_series, errors="coerce").fillna(0.0).values

    dp_series = df.get("delta_P_meas_kPa")
    if dp_series is None or dp_series.isna().all():
        dp_arr = np.full(total_points, clean_dp)
    else:
        dp_arr = pd.to_numeric(dp_series, errors="coerce").fillna(clean_dp).values

    eff_series = df.get("thermal_efficiency_ratio")
    if eff_series is None or eff_series.isna().all():
        eff_arr = np.ones(total_points)
    else:
        eff_arr = pd.to_numeric(eff_series, errors="coerce").fillna(1.0).values

    # 1. Fouling Resistance Progression Rate (dRf/dt)
    # Use robust linear regression over the time span
    x_hours = np.linspace(0.0, span_hours, total_points)
    if total_points >= 5:
        # Linear slope of Rf vs hours
        slope_rf_per_h, intercept_rf = np.polyfit(x_hours, rf_arr, 1)
        slope_rf_per_day = slope_rf_per_h * 24.0

        # Linear slope of delta_P vs hours
        slope_dp_per_h, intercept_dp = np.polyfit(x_hours, dp_arr, 1)
        slope_dp_per_day = slope_dp_per_h * 24.0
        slope_dp_per_100h = slope_dp_per_h * 100.0

        # Efficiency decay (% per 30 days)
        slope_eff_per_h, _ = np.polyfit(x_hours, eff_arr, 1)
        eff_decay_pct_30d = slope_eff_per_h * 24.0 * 30.0 * 100.0
    else:
        slope_rf_per_day = 0.0
        slope_dp_per_day = 0.0
        slope_dp_per_100h = 0.0
        eff_decay_pct_30d = 0.0

    # 2. Historical Extremes & Latest Condition
    rf_current = float(rf_arr[-1])
    rf_max = float(np.max(rf_arr))
    rf_mean = float(np.mean(rf_arr))

    dp_current = float(dp_arr[-1])
    dp_max = float(np.max(dp_arr))
    dp_min = float(np.min(dp_arr))
    dp_mean = float(np.mean(dp_arr))

    eff_current = float(eff_arr[-1])
    eff_min = float(np.min(eff_arr))

    # 3. Detect and Analyze Historical Cleaning Events
    # A cleaning event exhibits a sharp downward discontinuity in Rf or delta_P
    cleaning_events: List[Dict[str, Any]] = []
    if total_points >= 3:
        diff_rf = np.diff(rf_arr)
        diff_dp = np.diff(dp_arr)

        for i in range(len(diff_rf)):
            # Drop threshold: Rf drops by > 2e-5 m²K/W or dP drops by > 1.2 kPa in one step
            if diff_rf[i] < -2.0e-5 or diff_dp[i] < -1.2:
                step_idx = i + 1
                row_before = df.iloc[i]
                row_after = df.iloc[step_idx]

                h_event = int(row_after.get("time_hours", step_idx))
                rf_before = float(rf_arr[i])
                rf_after = float(rf_arr[step_idx])
                dp_before = float(dp_arr[i])
                dp_after = float(dp_arr[step_idx])

                # Recovery efficiency = (Rf_before - Rf_after) / Rf_before
                recov_eff = ((rf_before - rf_after) / max(1e-6, rf_before)) * 100.0 if rf_before > 1e-5 else 0.0

                cleaning_events.append({
                    "event_index": step_idx,
                    "event_hour": float(h_event),
                    "time_hours": h_event,
                    "timestamp": str(row_after.get("timestamp", f"Hour {h_event}")),
                    "rf_drop_m2k_w": round(rf_before - rf_after, 7),
                    "delta_Rf": round(rf_after - rf_before, 7),
                    "dp_drop_kPa": round(dp_before - dp_after, 2),
                    "delta_P_drop_kPa": round(dp_after - dp_before, 2),
                    "rf_recovery_efficiency_pct": round(min(100.0, max(0.0, recov_eff)), 1),
                    "thermal_recovery_pct": round(min(100.0, max(0.0, recov_eff)), 1),
                    "estimated_method": "Mechanical_Cleaning" if recov_eff < 92.0 else "Chemical_Biocide_Shock",
                })

    # 4. Projected Days to Operational Safety Breach (Linear extrapolation)
    days_to_rf_breach = None
    days_to_dp_breach = None

    if slope_rf_per_day > 1e-8:
        remaining_rf = config.CRITICAL_RF_M2K_W - rf_current
        if remaining_rf > 0:
            days_to_rf_breach = round(remaining_rf / slope_rf_per_day, 1)
        else:
            days_to_rf_breach = 0.0

    if slope_dp_per_day > 0.01:
        remaining_dp = config.CRITICAL_DELTA_P_KPA - dp_current
        if remaining_dp > 0:
            days_to_dp_breach = round(remaining_dp / slope_dp_per_day, 1)
        else:
            days_to_dp_breach = 0.0

    # 5. Diagnostic Severity Assessment
    if rf_current >= config.CRITICAL_RF_M2K_W or dp_current >= config.CRITICAL_DELTA_P_KPA:
        overall_severity = "CRITICAL_ACTION_REQUIRED"
        severity_color = "#EF4444"
    elif rf_current >= config.CRITICAL_RF_M2K_W * 0.70 or dp_current >= config.CRITICAL_DELTA_P_KPA * 0.75:
        overall_severity = "WARNING_APPROACHING_LIMITS"
        severity_color = "#F59E0B"
    else:
        overall_severity = "NOMINAL_OPERATING"
        severity_color = "#22C55E"

    return {
        "status": "SUCCESS",
        "total_records": total_points,
        "data_summary": {
            "total_points": total_points,
            "span_hours": round(span_hours, 1),
            "span_days": round(span_days, 1),
            "data_frequency_hours": round(span_hours / max(1, total_points - 1), 2) if total_points > 1 else 1.0,
        },
        "trends": {
            "rf_growth_rate_m2k_w_per_day": round(slope_rf_per_day, 8),
            "dp_growth_rate_kPa_per_day": round(slope_dp_per_day, 4),
            "dp_growth_rate_kPa_per_100h": round(slope_dp_per_100h, 3),
            "thermal_efficiency_decay_pct_30d": round(eff_decay_pct_30d, 2),
        },
        "progression_rates": {
            "dRf_dt_per_hour": round(slope_rf_per_day / 24.0, 9),
            "dRf_dt_per_day": round(slope_rf_per_day, 8),
            "dDeltaP_dt_kPa_per_day": round(slope_dp_per_day, 4),
            "thermal_efficiency_decay_pct_30d": round(eff_decay_pct_30d, 2),
        },
        "cleaning_history": {
            "detected_cleanings_count": len(cleaning_events),
            "events": cleaning_events,
        },
        "extrapolated_breach_estimates": {
            "days_to_rf_breach": days_to_rf_breach,
            "days_to_dp_breach": days_to_dp_breach,
            "limiting_parameter": "Rf_fouling_resistance" if (days_to_rf_breach or 999) < (days_to_dp_breach or 999) else "delta_P_meas_kPa",
        },
        "current_state": {
            "Rf_fouling_resistance": round(rf_current, 7),
            "delta_P_meas_kPa": round(dp_current, 2),
            "thermal_efficiency_ratio": round(eff_current, 4),
            "clean_pressure_drop_kPa": clean_dp,
            "pressure_surge_pct": round(((dp_current - clean_dp) / clean_dp) * 100.0, 1),
            "severity_assessment": overall_severity,
            "severity_color": severity_color,
        },
        "historical_extremes": {
            "Rf_max": round(rf_max, 7),
            "Rf_mean": round(rf_mean, 7),
            "delta_P_max_kPa": round(dp_max, 2),
            "delta_P_min_kPa": round(dp_min, 2),
            "delta_P_mean_kPa": round(dp_mean, 2),
            "thermal_efficiency_min": round(eff_min, 4),
        },
        "detected_cleaning_events": cleaning_events,
        "cleaning_event_count": len(cleaning_events),
        "provenance": {
            "measured_fields": ["T_cool_in_meas", "T_cool_out_meas", "mass_flow_cool_meas", "delta_P_meas_kPa"],
            "derived_fields": ["Rf_fouling_resistance", "thermal_efficiency_ratio", "bsi_score", "parasitic_pump_kw"],
            "analytical_methods": ["Linear least-squares regression", "Continuum Darcy-Weisbach hydraulics", "Discontinuity detection"],
        },
        "nuclear_safety_notice": "Academic Demonstration Prototype — Diagnostic Support Only",
    }
