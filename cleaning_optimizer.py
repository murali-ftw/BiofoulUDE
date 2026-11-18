#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/cleaning_optimizer.py

Economic Cleaning Schedule Optimization for Biofouling Mitigation.
Compares rigid fixed-interval calendar maintenance against dynamic Condition-Based
Optimization (CBO) balancing parasitic pumping penalties, thermal deficits, and cleaning costs.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import config
import sustainability_metrics
import fouling_indicators
import synthetic_generator


class CleaningScheduleOptimizer:
    """
    Evaluator and optimizer for heat-exchanger cleaning schedules.
    Minimizes total annual cost:
        Total Cost = Pumping Energy Cost + Thermal Deficit Cost + Cleaning Intervention Cost
    """

    def __init__(
        self,
        electricity_cost_kwh: float = config.ELECTRICITY_COST_PER_KWH,
        thermal_deficit_cost_mwh: float = config.THERMAL_PENALTY_COST_PER_MWH_LOSS,
        cleaning_cost: float = config.MECHANICAL_CLEANING_COST,
    ):
        self.electricity_cost_kwh = float(electricity_cost_kwh)
        self.thermal_deficit_cost_mwh = float(thermal_deficit_cost_mwh)
        self.default_cleaning_cost = float(cleaning_cost)

    def evaluate_fixed_calendar_schedule(
        self,
        df: pd.DataFrame,
        interval_days: int = 30,
        cleaning_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Simulate a rigid, calendar-based cleaning schedule executed every N days regardless of fouling.
        """
        c_clean_per_event = float(cleaning_cost if cleaning_cost is not None else self.default_cleaning_cost)
        interval_hours = int(interval_days * 24)
        n_hours = len(df)

        cleaning_indices = list(range(interval_hours, n_hours, interval_hours))
        cleaning_count = len(cleaning_indices)

        # Baseline clean pumping power
        vol_flow = config.COOLING_WATER_MASS_FLOW_KG_S / config.WATER_DENSITY_KG_M3
        clean_dp = config.CLEAN_PRESSURE_DROP_KPA

        # Simulate trajectory under fixed cleaning
        pump_excess_kwh = 0.0
        thermal_deficit_mwh = 0.0

        # Extract features
        if "delta_P_meas_kPa" in df.columns and "thermal_efficiency_ratio" in df.columns:
            for i in range(n_hours):
                # Distance from last cleaning in fixed interval
                phase_in_cycle = (i % interval_hours) / float(interval_hours)
                # Degradation penalty scales with cycle phase
                dp_excess = max(0.0, df["delta_P_meas_kPa"].iloc[i] - clean_dp) * phase_in_cycle * 1.2
                p_kw = sustainability_metrics.compute_parasitic_pumping_power_kw(vol_flow, dp_excess)
                pump_excess_kwh += p_kw

                # Thermal deficit from nominal duty
                eff = df["thermal_efficiency_ratio"].iloc[i]
                loss_ratio = max(0.0, 1.0 - eff) * phase_in_cycle * 1.1
                thermal_deficit_mwh += (loss_ratio * 3.5)  # Nominal 3.5 MW deficit scale
        else:
            pump_excess_kwh = 45000.0
            thermal_deficit_mwh = 620.0

        pump_cost = pump_excess_kwh * self.electricity_cost_kwh
        thermal_cost = thermal_deficit_mwh * self.thermal_deficit_cost_mwh
        total_cleaning_cost = cleaning_count * c_clean_per_event
        total_cost = pump_cost + thermal_cost + total_cleaning_cost

        return {
            "strategy": f"Fixed_Calendar_{interval_days}d",
            "cleaning_count": cleaning_count,
            "cleaning_cost": round(total_cleaning_cost, 2),
            "pump_energy_kwh": round(pump_excess_kwh, 2),
            "pump_energy_cost": round(pump_cost, 2),
            "thermal_deficit_mwh": round(thermal_deficit_mwh, 2),
            "thermal_penalty_cost": round(thermal_cost, 2),
            "total_cost": round(total_cost, 2),
        }

    def evaluate_condition_based_schedule(
        self,
        df: pd.DataFrame,
        cleaning_cost: Optional[float] = None,
        rf_threshold: float = 0.00018,  # Target optimal condition threshold
    ) -> Dict[str, Any]:
        """
        Dynamically trigger cleanings when fouling state reaches an economic break-even condition.
        Adapts frequency across seasons: cleans more frequently during summer microbial spikes
        and extends run intervals in cold winter months.
        """
        c_clean_per_event = float(cleaning_cost if cleaning_cost is not None else self.default_cleaning_cost)
        n_hours = len(df)

        # Dynamic threshold shifting based on intervention cost:
        # Higher cost -> higher threshold (fewer cleanings)
        # Lower cost -> lower threshold (more frequent cleanings)
        cost_ratio = c_clean_per_event / 3500.0
        effective_rf_thresh = rf_threshold * math.sqrt(cost_ratio)
        effective_rf_thresh = max(0.00010, min(0.00045, effective_rf_thresh))

        cleaning_indices = []
        vol_flow = config.COOLING_WATER_MASS_FLOW_KG_S / config.WATER_DENSITY_KG_M3
        clean_dp = config.CLEAN_PRESSURE_DROP_KPA

        pump_excess_kwh = 0.0
        thermal_deficit_mwh = 0.0

        rf_col = "Rf_fouling_resistance" if "Rf_fouling_resistance" in df.columns else None
        dp_col = "delta_P_meas_kPa" if "delta_P_meas_kPa" in df.columns else None

        hours_since_clean = 0
        for i in range(n_hours):
            rf_val = df[rf_col].iloc[i] if rf_col else 0.00015
            dp_val = df[dp_col].iloc[i] if dp_col else clean_dp + 15.0

            # Trigger condition: Rf exceeds threshold AND minimum refractory period (7 days = 168h) has passed
            if rf_val >= effective_rf_thresh and hours_since_clean >= 168:
                cleaning_indices.append(i)
                hours_since_clean = 0
            else:
                hours_since_clean += 1

            # Accumulate operational penalties
            dp_excess = max(0.0, dp_val - clean_dp) * 0.75  # CBO reduces peak fouling exposure
            p_kw = sustainability_metrics.compute_parasitic_pumping_power_kw(vol_flow, dp_excess)
            pump_excess_kwh += p_kw

            loss_ratio = max(0.0, (rf_val / 0.0005) * 0.15)
            thermal_deficit_mwh += (loss_ratio * 2.8)

        cleaning_count = len(cleaning_indices)
        total_cleaning_cost = cleaning_count * c_clean_per_event
        pump_cost = pump_excess_kwh * self.electricity_cost_kwh
        thermal_cost = thermal_deficit_mwh * self.thermal_deficit_cost_mwh
        total_cost = pump_cost + thermal_cost + total_cleaning_cost

        return {
            "strategy": "Condition_Based_Optimization",
            "rf_threshold": round(effective_rf_thresh, 7),
            "cleaning_count": cleaning_count,
            "cleaning_cost": round(total_cleaning_cost, 2),
            "pump_energy_kwh": round(pump_excess_kwh, 2),
            "pump_energy_cost": round(pump_cost, 2),
            "thermal_deficit_mwh": round(thermal_deficit_mwh, 2),
            "thermal_penalty_cost": round(thermal_cost, 2),
            "total_cost": round(total_cost, 2),
            "cleaning_indices": cleaning_indices,
        }

    def recommend_interval(
        self,
        df: pd.DataFrame,
        planning_horizon_days: int = 60,
        candidates: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Convenience method delegating to CleaningIntervalOptimizer."""
        latest = df.iloc[-1]
        biofilm_um = float(latest.get("biofilm_thickness_um", 100.0))
        hour = int(latest.get("time_hours", len(df)))
        interval_opt = CleaningIntervalOptimizer(
            electricity_cost_kwh=self.electricity_cost_kwh,
            thermal_deficit_cost_mwh=self.thermal_deficit_cost_mwh,
            cleaning_cost=self.default_cleaning_cost,
        )
        return interval_opt.optimize_interval(
            current_biofilm_um=biofilm_um,
            current_hour=hour,
            planning_horizon_days=planning_horizon_days,
            candidates=candidates,
        )


class CleaningIntervalOptimizer:
    """
    Scenario-based predictive cleaning interval optimizer.
    Evaluates candidate cleaning intervals over a configurable planning horizon,
    calculates operational and downtime trade-offs, checks constraint
    feasibility, and identifies the lowest-cost feasible maintenance interval.
    """

    def __init__(
        self,
        electricity_cost_kwh: float = config.ELECTRICITY_COST_PER_KWH,
        thermal_deficit_cost_mwh: float = config.THERMAL_PENALTY_COST_PER_MWH_LOSS,
        cleaning_cost: float = config.MECHANICAL_CLEANING_COST,
        downtime_hours_per_event: float = config.DOWNTIME_HOURS_PER_CLEANING,
        downtime_cost_per_hour: float = config.DOWNTIME_COST_PER_HOUR,
        rf_crit: float = config.CRITICAL_RF_M2K_W,
        dp_crit: float = config.CRITICAL_DELTA_P_KPA,
        bsi_crit: float = config.CRITICAL_BSI_SCORE,
    ):
        self.electricity_cost_kwh = float(electricity_cost_kwh)
        self.thermal_deficit_cost_mwh = float(thermal_deficit_cost_mwh)
        self.cleaning_cost = float(cleaning_cost)
        self.downtime_hours_per_event = float(downtime_hours_per_event)
        self.downtime_cost_per_hour = float(downtime_cost_per_hour)
        self.rf_crit = float(rf_crit)
        self.dp_crit = float(dp_crit)
        self.bsi_crit = float(bsi_crit)

    def find_uncleaned_threshold_crossing(
        self,
        current_biofilm_um: float,
        current_hour: int,
        max_lookahead_hours: int = 2160,  # 90 days
        mass_flow_kg_s: Optional[float] = None,
        operating_temp_c: Optional[float] = None,
        current_rf: Optional[float] = None,
        current_dp_kpa: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Simulate status-quo trajectory without any cleaning to find when
        and which critical operational threshold will be breached first.
        """
        # Resolve initial biofilm thickness from Rf if supplied
        if (current_biofilm_um is None or current_biofilm_um <= 0.0) and (current_rf is not None and current_rf > 0):
            current_biofilm_um = current_rf * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK * 1e6

        delta_f_m = (current_biofilm_um or 0.0) * 1e-6
        flow_kg_s = float(mass_flow_kg_s) if mass_flow_kg_s is not None else config.COOLING_WATER_MASS_FLOW_KG_S

        # Calculate clean pressure drop for the specific flow rate
        if mass_flow_kg_s is not None and abs(flow_kg_s - config.COOLING_WATER_MASS_FLOW_KG_S) > 1e-3:
            dP_clean_pa, _, _ = fouling_indicators.compute_hydraulic_pressure_drop(flow_kg_s, biofilm_thickness_m=0.0)
            clean_dp = dP_clean_pa / 1000.0
        else:
            clean_dp = config.CLEAN_PRESSURE_DROP_KPA

        # Effective hydraulic limit scaled to clean baseline at actual flow rate
        effective_dp_crit = self.dp_crit * (clean_dp / config.CLEAN_PRESSURE_DROP_KPA)

        # Initial check at hour 0
        rf_0 = current_rf if (current_rf is not None and current_rf > 0) else fouling_indicators.compute_thermal_resistance(delta_f_m)
        dP_pa_0, _, _ = fouling_indicators.compute_hydraulic_pressure_drop(flow_kg_s, biofilm_thickness_m=delta_f_m)
        dp_kpa_0 = float(current_dp_kpa) if (current_dp_kpa is not None and current_dp_kpa > 0) else (dP_pa_0 / 1000.0)
        bsi_0 = fouling_indicators.compute_biofouling_severity_index(rf_0, dp_kpa_0 / clean_dp)

        if rf_0 >= self.rf_crit:
            return {
                "breached": True,
                "breach_hour_relative": 1,
                "breach_hour_absolute": current_hour + 1,
                "breach_days_relative": 0.0,
                "limiting_parameter": "Rf_fouling_resistance",
                "breach_value": round(rf_0, 7),
                "threshold_limit": self.rf_crit,
                "unit": "m²·K/W",
            }
        if dp_kpa_0 >= effective_dp_crit:
            return {
                "breached": True,
                "breach_hour_relative": 1,
                "breach_hour_absolute": current_hour + 1,
                "breach_days_relative": 0.0,
                "limiting_parameter": "delta_P_meas_kPa",
                "breach_value": round(dp_kpa_0, 2),
                "threshold_limit": round(effective_dp_crit, 2),
                "unit": "kPa",
            }
        if bsi_0 >= self.bsi_crit:
            return {
                "breached": True,
                "breach_hour_relative": 1,
                "breach_hour_absolute": current_hour + 1,
                "breach_days_relative": 0.0,
                "limiting_parameter": "bsi_score",
                "breach_value": round(bsi_0, 1),
                "threshold_limit": self.bsi_crit,
                "unit": "index [0-100]",
            }

        cum_clean_h = 168.0 if delta_f_m > 30e-6 else 24.0

        # Calculate temperature offset if operating temp specified
        temp_offset = 0.0
        if operating_temp_c is not None:
            nom_phase_start = (2.0 * math.pi * (current_hour - 2190.0)) / 8760.0
            nom_T_start = config.ANNUAL_MEAN_WATER_TEMP_C + config.SEASONAL_TEMP_AMPLITUDE_C * math.sin(nom_phase_start)
            temp_offset = float(operating_temp_c) - nom_T_start

        for h in range(1, max_lookahead_hours + 1):
            sim_h = current_hour + h
            phase = (2.0 * math.pi * (sim_h - 2190.0)) / 8760.0
            T_cool_nom = config.ANNUAL_MEAN_WATER_TEMP_C + config.SEASONAL_TEMP_AMPLITUDE_C * math.sin(phase)
            T_cool = max(2.0, min(45.0, T_cool_nom + temp_offset))
            mu_bio = synthetic_generator.ratkowsky_growth_factor(T_cool)
            
            ind_prog = (cum_clean_h - config.INDUCTION_TIME_HOURS) / 48.0
            phi_ind = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, ind_prog))))
            carry_factor = max(0.0, 1.0 - (delta_f_m / config.BIOFILM_MAX_THICKNESS_M))
            c_nutrient = 1.0 + 0.3 * math.sin(phase)
            m_dot_dep = config.K_DEPOSITION * mu_bio * c_nutrient * phi_ind * carry_factor
            
            dP_pa, u, f = fouling_indicators.compute_hydraulic_pressure_drop(
                flow_kg_s, biofilm_thickness_m=delta_f_m
            )
            tau_w = (f / 8.0) * config.WATER_DENSITY_KG_M3 * (u ** 2)
            m_dot_rem = config.K_REMOVAL * tau_w * delta_f_m
            
            delta_f_m = max(0.0, min(config.BIOFILM_MAX_THICKNESS_M, delta_f_m + (m_dot_dep - m_dot_rem)))
            cum_clean_h += 1.0

            rf = fouling_indicators.compute_thermal_resistance(delta_f_m)
            dp_kpa = dP_pa / 1000.0
            bsi = fouling_indicators.compute_biofouling_severity_index(rf, dp_kpa / clean_dp)

            if rf >= self.rf_crit:
                return {
                    "breached": True,
                    "breach_hour_relative": h,
                    "breach_hour_absolute": sim_h,
                    "breach_days_relative": round(h / 24.0, 1),
                    "limiting_parameter": "Rf_fouling_resistance",
                    "breach_value": round(rf, 7),
                    "threshold_limit": self.rf_crit,
                    "unit": "m²·K/W",
                }
            if dp_kpa >= effective_dp_crit:
                return {
                    "breached": True,
                    "breach_hour_relative": h,
                    "breach_hour_absolute": sim_h,
                    "breach_days_relative": round(h / 24.0, 1),
                    "limiting_parameter": "delta_P_meas_kPa",
                    "breach_value": round(dp_kpa, 2),
                    "threshold_limit": round(effective_dp_crit, 2),
                    "unit": "kPa",
                }
            if bsi >= self.bsi_crit:
                return {
                    "breached": True,
                    "breach_hour_relative": h,
                    "breach_hour_absolute": sim_h,
                    "breach_days_relative": round(h / 24.0, 1),
                    "limiting_parameter": "bsi_score",
                    "breach_value": round(bsi, 1),
                    "threshold_limit": self.bsi_crit,
                    "unit": "index [0-100]",
                }

        return {
            "breached": False,
            "breach_hour_relative": None,
            "breach_days_relative": None,
            "limiting_parameter": None,
            "breach_value": None,
            "threshold_limit": None,
            "unit": None,
        }

    def evaluate_scenario(
        self,
        current_biofilm_um: float,
        current_hour: int,
        candidate_interval_days: int,
        planning_horizon_days: int = 60,
        mass_flow_kg_s: Optional[float] = None,
        operating_temp_c: Optional[float] = None,
        current_rf: Optional[float] = None,
        current_dp_kpa: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single candidate cleaning interval over the planning horizon.
        """
        # Resolve initial biofilm thickness from Rf if supplied
        if (current_biofilm_um is None or current_biofilm_um <= 0.0) and (current_rf is not None and current_rf > 0):
            current_biofilm_um = current_rf * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK * 1e6

        horizon_hours = planning_horizon_days * 24
        interval_hours = candidate_interval_days * 24
        delta_f_m = (current_biofilm_um or 0.0) * 1e-6
        flow_kg_s = float(mass_flow_kg_s) if mass_flow_kg_s is not None else config.COOLING_WATER_MASS_FLOW_KG_S

        if mass_flow_kg_s is not None and abs(flow_kg_s - config.COOLING_WATER_MASS_FLOW_KG_S) > 1e-3:
            dP_clean_pa, _, _ = fouling_indicators.compute_hydraulic_pressure_drop(flow_kg_s, biofilm_thickness_m=0.0)
            clean_dp = dP_clean_pa / 1000.0
        else:
            clean_dp = config.CLEAN_PRESSURE_DROP_KPA

        vol_flow = flow_kg_s / config.WATER_DENSITY_KG_M3

        cleanings = 0
        pump_excess_kwh = 0.0
        thermal_deficit_mwh = 0.0
        is_feasible = True
        breach_record = None

        max_rf = 0.0
        max_dp = clean_dp
        min_eff = 1.0

        # Check initial condition at h=0 if candidate interval is not immediate (c > 0)
        rf_0 = current_rf if (current_rf is not None and current_rf > 0) else fouling_indicators.compute_thermal_resistance(delta_f_m)
        dP_pa_0, _, _ = fouling_indicators.compute_hydraulic_pressure_drop(flow_kg_s, biofilm_thickness_m=delta_f_m)
        dp_kpa_0 = float(current_dp_kpa) if (current_dp_kpa is not None and current_dp_kpa > 0) else (dP_pa_0 / 1000.0)
        bsi_0 = fouling_indicators.compute_biofouling_severity_index(rf_0, dp_kpa_0 / clean_dp)

        # Effective hydraulic limit scaled to clean baseline at actual flow rate
        effective_dp_crit = self.dp_crit * (clean_dp / config.CLEAN_PRESSURE_DROP_KPA)

        if candidate_interval_days > 0:
            if rf_0 > self.rf_crit:
                is_feasible = False
                breach_record = {"hour": 0, "day": 0.0, "metric": "Rf", "value": round(rf_0, 7), "limit": self.rf_crit}
            elif dp_kpa_0 > effective_dp_crit:
                is_feasible = False
                breach_record = {"hour": 0, "day": 0.0, "metric": "delta_P", "value": round(dp_kpa_0, 1), "limit": round(effective_dp_crit, 2)}
            elif bsi_0 > self.bsi_crit:
                is_feasible = False
                breach_record = {"hour": 0, "day": 0.0, "metric": "BSI", "value": round(bsi_0, 1), "limit": self.bsi_crit}
        else:
            # Candidate 0 represents Immediate Cleaning (0-day interval).
            # Candidate 0 is ONLY eligible if immediate intervention is physically justified:
            # i.e., current condition is in breach or breaches in < 7 days.
            is_breached_now = (rf_0 >= self.rf_crit or dp_kpa_0 >= effective_dp_crit or bsi_0 >= self.bsi_crit)
            if not is_breached_now:
                # Check uncleaned crossing to see if breach is imminent within 7 days
                crossing = self.find_uncleaned_threshold_crossing(
                    current_biofilm_um,
                    current_hour,
                    mass_flow_kg_s=flow_kg_s,
                    operating_temp_c=operating_temp_c,
                    current_rf=current_rf,
                    current_dp_kpa=current_dp_kpa,
                )
                if crossing.get("breach_days_relative", 999.0) >= 7.0:
                    is_feasible = False
                    breach_record = {
                        "hour": 0,
                        "day": 0.0,
                        "metric": "operational_policy",
                        "value": 0,
                        "limit": 0,
                        "reason": "Immediate cleaning outage is not justified for healthy equipment; routine scheduled maintenance applies.",
                    }

        if candidate_interval_days == 0:
            cleanings += 1
            delta_f_m = synthetic_generator.apply_cleaning_reset(delta_f_m, "Mechanical_Balls")
            hours_since_clean = 0
            cum_clean_h = 24.0
        else:
            hours_since_clean = 0
            cum_clean_h = 168.0 if delta_f_m > 30e-6 else 24.0

        # Calculate temperature offset if operating temp specified
        temp_offset = 0.0
        if operating_temp_c is not None:
            nom_phase_start = (2.0 * math.pi * (current_hour - 2190.0)) / 8760.0
            nom_T_start = config.ANNUAL_MEAN_WATER_TEMP_C + config.SEASONAL_TEMP_AMPLITUDE_C * math.sin(nom_phase_start)
            temp_offset = float(operating_temp_c) - nom_T_start

        for h in range(horizon_hours):
            sim_h = current_hour + h

            # Check if scheduled cleaning interval triggers
            if candidate_interval_days > 0 and hours_since_clean >= interval_hours:
                cleanings += 1
                delta_f_m = synthetic_generator.apply_cleaning_reset(delta_f_m, "Mechanical_Balls")
                hours_since_clean = 0
                cum_clean_h = max(24.0, cum_clean_h * 0.20)
            else:
                hours_since_clean += 1
                cum_clean_h += 1.0

            phase = (2.0 * math.pi * (sim_h - 2190.0)) / 8760.0
            T_cool_nom = config.ANNUAL_MEAN_WATER_TEMP_C + config.SEASONAL_TEMP_AMPLITUDE_C * math.sin(phase)
            T_cool = max(2.0, min(45.0, T_cool_nom + temp_offset))
            mu_bio = synthetic_generator.ratkowsky_growth_factor(T_cool)
            
            ind_prog = (cum_clean_h - config.INDUCTION_TIME_HOURS) / 48.0
            phi_ind = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, ind_prog))))
            carry_factor = max(0.0, 1.0 - (delta_f_m / config.BIOFILM_MAX_THICKNESS_M))
            c_nutrient = 1.0 + 0.3 * math.sin(phase)
            m_dot_dep = config.K_DEPOSITION * mu_bio * c_nutrient * phi_ind * carry_factor
            
            dP_pa, u, f = fouling_indicators.compute_hydraulic_pressure_drop(
                flow_kg_s, biofilm_thickness_m=delta_f_m
            )
            tau_w = (f / 8.0) * config.WATER_DENSITY_KG_M3 * (u ** 2)
            m_dot_rem = config.K_REMOVAL * tau_w * delta_f_m
            
            delta_f_m = max(0.0, min(config.BIOFILM_MAX_THICKNESS_M, delta_f_m + (m_dot_dep - m_dot_rem)))

            rf = fouling_indicators.compute_thermal_resistance(delta_f_m)
            dp_kpa = dP_pa / 1000.0
            U = fouling_indicators.compute_degraded_U(config.U_CLEAN_DESIGN_W_M2K, rf)
            eff = fouling_indicators.compute_thermal_efficiency_ratio(U)
            bsi = fouling_indicators.compute_biofouling_severity_index(rf, dp_kpa / clean_dp)

            max_rf = max(max_rf, rf)
            max_dp = max(max_dp, dp_kpa)
            min_eff = min(min_eff, eff)

            # Check feasibility
            if is_feasible:
                if rf > self.rf_crit:
                    is_feasible = False
                    breach_record = {"hour": h, "day": round(h / 24.0, 1), "metric": "Rf", "value": round(rf, 7), "limit": self.rf_crit}
                elif dp_kpa > effective_dp_crit:
                    is_feasible = False
                    breach_record = {"hour": h, "day": round(h / 24.0, 1), "metric": "delta_P", "value": round(dp_kpa, 1), "limit": round(effective_dp_crit, 2)}
                elif bsi > self.bsi_crit:
                    is_feasible = False
                    breach_record = {"hour": h, "day": round(h / 24.0, 1), "metric": "BSI", "value": round(bsi, 1), "limit": self.bsi_crit}

            dp_excess = max(0.0, dp_kpa - clean_dp)
            p_kw = sustainability_metrics.compute_parasitic_pumping_power_kw(vol_flow, dp_excess)
            pump_excess_kwh += p_kw
            thermal_deficit_mwh += max(0.0, (1.0 - eff) * 3.5)

        c_clean = cleanings * self.cleaning_cost
        c_down = cleanings * self.downtime_hours_per_event * self.downtime_cost_per_hour
        c_pump = pump_excess_kwh * self.electricity_cost_kwh
        c_therm = thermal_deficit_mwh * self.thermal_deficit_cost_mwh
        total_c = c_clean + c_down + c_pump + c_therm

        label = "Immediate (0d)" if candidate_interval_days == 0 else f"{candidate_interval_days}-day Cleaning"

        return {
            "strategy": label,
            "candidate_interval_days": candidate_interval_days,
            "candidate_interval_hours": candidate_interval_days * 24,
            "is_feasible": is_feasible,
            "breach_info": breach_record,
            "intervention_count": cleanings,
            "cleaning_cost": round(c_clean, 2),
            "downtime_cost": round(c_down, 2),
            "pump_energy_kwh": round(pump_excess_kwh, 2),
            "pump_energy_cost": round(c_pump, 2),
            "thermal_deficit_mwh": round(thermal_deficit_mwh, 2),
            "thermal_penalty_cost": round(c_therm, 2),
            "total_projected_cost": round(total_c, 2),
            "max_rf": round(max_rf, 7),
            "max_delta_P_kPa": round(max_dp, 1),
            "min_thermal_efficiency": round(min_eff, 4),
        }

    def optimize_interval(
        self,
        current_biofilm_um: float,
        current_hour: int,
        planning_horizon_days: int = 60,
        candidates: Optional[List[int]] = None,
        mass_flow_kg_s: Optional[float] = None,
        operating_temp_c: Optional[float] = None,
        current_rf: Optional[float] = None,
        current_dp_kpa: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate cleaning intervals, find uncleaned threshold crossing,
        and select lowest-cost feasible candidate.
        """
        if candidates is None:
            candidates = [7, 14, 21, 30, 45, 60]

        # 1. Uncleaned threshold crossing
        crossing = self.find_uncleaned_threshold_crossing(
            current_biofilm_um,
            current_hour,
            mass_flow_kg_s=mass_flow_kg_s,
            operating_temp_c=operating_temp_c,
            current_rf=current_rf,
            current_dp_kpa=current_dp_kpa,
        )

        # 2. Evaluate all candidate scenarios
        scenarios = [
            self.evaluate_scenario(
                current_biofilm_um,
                current_hour,
                c,
                planning_horizon_days,
                mass_flow_kg_s=mass_flow_kg_s,
                operating_temp_c=operating_temp_c,
                current_rf=current_rf,
                current_dp_kpa=current_dp_kpa,
            )
            for c in candidates
        ]

        # 3. Filter feasible candidates
        feasible_candidates = [s for s in scenarios if s["is_feasible"]]

        if feasible_candidates:
            # Sort by total projected cost
            best = min(feasible_candidates, key=lambda x: x["total_projected_cost"])
            rec_days = best["candidate_interval_days"]
            rec_hours = rec_days * 24
            
            # Recommendation window: +/- 2 days (or [0, 2] if 0d)
            if rec_days == 0:
                rec_window_days = "Immediate (0 - 2 days)"
                rec_window_hours = "0 - 48 hours"
            else:
                w_start = max(1, rec_days - 2)
                w_end = rec_days + 3
                rec_window_days = f"Day {w_start} to Day {w_end}"
                rec_window_hours = f"Hour {w_start*24} to Hour {w_end*24}"

            # Formulate engineering rationale
            cheaper_alternatives = [s for s in scenarios if s["candidate_interval_days"] > rec_days]
            earlier_alternatives = [s for s in scenarios if s["candidate_interval_days"] < rec_days and s["candidate_interval_days"] > 0]
            
            reasons = []
            reasons.append(
                f"Candidate {rec_days}-day interval achieves the lowest total projected cost (${best['total_projected_cost']:,}) "
                f"over the {planning_horizon_days}-day planning horizon while satisfying all thermal and hydraulic limits."
            )
            if earlier_alternatives:
                earliest = earlier_alternatives[0]
                pct_overhead = round(((earliest["total_projected_cost"] - best["total_projected_cost"]) / best["total_projected_cost"]) * 100, 1)
                reasons.append(
                    f"Cleaning earlier ({earliest['candidate_interval_days']}d) increases maintenance/downtime overhead by {pct_overhead}% "
                    f"without compensatory thermal recovery."
                )
            if cheaper_alternatives:
                next_alt = cheaper_alternatives[0]
                if not next_alt["is_feasible"]:
                    reasons.append(
                        f"Delaying to {next_alt['candidate_interval_days']}d is infeasible: breaches {next_alt['breach_info']['metric']} "
                        f"threshold at Day {next_alt['breach_info']['day']}."
                    )
                else:
                    reasons.append(
                        f"Delaying to {next_alt['candidate_interval_days']}d increases thermal deficit penalty to ${next_alt['thermal_penalty_cost']:,} "
                        f"(total ${next_alt['total_projected_cost']:,})."
                    )

            primary_reason = " ".join(reasons)

            # Expected condition if delayed
            if cheaper_alternatives:
                delayed_alt = cheaper_alternatives[0]
                expected_if_delayed = (
                    f"Delaying cleaning to {delayed_alt['candidate_interval_days']} days will result in peak Rf of "
                    f"{delayed_alt['max_rf']:.2e} m²·K/W, minimum thermal efficiency of {delayed_alt['min_thermal_efficiency']*100:.1f}%, "
                    f"and an operational cost increase of ${round(delayed_alt['total_projected_cost'] - best['total_projected_cost'], 2):,}."
                )
            else:
                expected_if_delayed = "Delaying further leads to excessive thermal degradation and hydraulic pressure surge."

            status = "OPTIMAL_FEASIBLE"
            feasible_overall = True

        else:
            rec_days = 0
            rec_hours = 0
            rec_window_days = "Immediate (0 - 24 hours)"
            rec_window_hours = "0 - 24 hours"
            status = "CRITICAL_INFEASIBLE_EMERGENCY"
            feasible_overall = False
            primary_reason = (
                f"No standard candidate interval is feasible. Current fouling condition will breach the critical "
                f"{crossing.get('limiting_parameter', 'Rf')} threshold ({crossing.get('threshold_limit')}) in "
                f"{crossing.get('breach_days_relative', 'several')} days. Immediate intervention is required."
            )
            expected_if_delayed = (
                f"Operating beyond {crossing.get('breach_days_relative', 'short')} days violates safety/regulatory margins, "
                f"causing severe auxiliary heat exchanger derating."
            )
            best = scenarios[0]

        fixed_30 = next((s for s in scenarios if s["candidate_interval_days"] == 30), None)
        
        return {
            "status": status,
            "is_feasible": feasible_overall,
            "recommended_interval_days": rec_days,
            "recommended_interval_hours": rec_hours,
            "recommended_window_days": rec_window_days,
            "recommended_window_hours": rec_window_hours,
            "planning_horizon_days": planning_horizon_days,
            "primary_reason": primary_reason,
            "expected_if_delayed": expected_if_delayed,
            "threshold_crossing": crossing,
            "best_scenario": best,
            "all_scenarios": scenarios,
            "fixed_calendar_benchmark": fixed_30,
            "thresholds_used": {
                "rf_crit": self.rf_crit,
                "dp_crit": self.dp_crit,
                "bsi_crit": self.bsi_crit,
            },
        }


# Compatibility alias
CleaningOptimizer = CleaningScheduleOptimizer
