#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/sustainability_metrics.py

Environmental and Energy Sustainability Accounting Module.
Calculates parasitic pumping power surges, equivalent grid carbon emissions,
auxiliary thermal replacement penalties, and biocide discharge permit margins.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

import config


def compute_parasitic_pumping_power_kw(
    volumetric_flow_m3_s: float,
    delta_P_excess_kPa: float,
    efficiency: float = config.PUMP_EFFICIENCY,
) -> float:
    """
    Compute excess parasitic hydraulic pumping power:
        P_pump [kW] = (V_dot [m³/s] * delta_P_excess [Pa]) / (eta_pump * 1000)
    """
    if efficiency <= 0:
        raise ValueError("Pump efficiency must be positive.")
    dP_pa = max(0.0, float(delta_P_excess_kPa)) * 1000.0
    p_watts = (float(volumetric_flow_m3_s) * dP_pa) / float(efficiency)
    return float(p_watts / 1000.0)


def compute_pumping_power(volumetric_flow_m3_s: float, delta_P_excess_kPa: float, efficiency: float = config.PUMP_EFFICIENCY) -> float:
    """Alias for compute_parasitic_pumping_power_kw."""
    return compute_parasitic_pumping_power_kw(volumetric_flow_m3_s, delta_P_excess_kPa, efficiency)


def compute_carbon_emissions_kg(
    excess_energy_kwh: float,
    carbon_intensity_kg_per_kwh: float = config.GRID_CARBON_INTENSITY_KG_CO2_PER_KWH,
) -> float:
    """
    Compute equivalent CO2 greenhouse gas emissions:
        CO2 [kg] = excess_energy [kWh] * grid_carbon_intensity [kg CO2/kWh]
    """
    return float(max(0.0, float(excess_energy_kwh)) * float(carbon_intensity_kg_per_kwh))


def compute_co2_penalty(excess_energy_kwh: float, carbon_intensity_kg_per_kwh: float = config.GRID_CARBON_INTENSITY_KG_CO2_PER_KWH) -> float:
    """Alias for compute_carbon_emissions_kg."""
    return compute_carbon_emissions_kg(excess_energy_kwh, carbon_intensity_kg_per_kwh)


def track_biocide_discharge(
    cleaning_events_count: int,
    biocide_kg_per_event: float = config.BIOCIDE_KG_PER_EVENT,
    annual_permit_limit_kg: float = config.ANNUAL_BIOCIDE_PERMIT_LIMIT_KG,
) -> Dict[str, Any]:
    """
    Track cumulative chemical biocide usage and verify environmental discharge compliance.
    """
    total_biocide = float(max(0, int(cleaning_events_count)) * float(biocide_kg_per_event))
    within_limit = bool(total_biocide <= float(annual_permit_limit_kg))
    util_pct = (total_biocide / float(annual_permit_limit_kg)) * 100.0 if annual_permit_limit_kg > 0 else 0.0

    return {
        "cleaning_events_count": int(cleaning_events_count),
        "biocide_kg_per_event": round(float(biocide_kg_per_event), 2),
        "total_biocide_kg": round(total_biocide, 2),
        "annual_permit_limit_kg": round(float(annual_permit_limit_kg), 2),
        "within_permit_limit": within_limit,
        "permit_utilization_pct": round(util_pct, 2),
    }


def compute_biocide_impact(cleaning_events_count: int, biocide_kg_per_event: float = config.BIOCIDE_KG_PER_EVENT, annual_permit_limit_kg: float = config.ANNUAL_BIOCIDE_PERMIT_LIMIT_KG):
    """Alias for track_biocide_discharge."""
    return track_biocide_discharge(cleaning_events_count, biocide_kg_per_event, annual_permit_limit_kg)


def compute_total_sustainability_impact(
    df: pd.DataFrame,
    clean_dP_kpa: float = config.CLEAN_PRESSURE_DROP_KPA,
    pump_eff: float = config.PUMP_EFFICIENCY,
    carbon_intensity: float = config.GRID_CARBON_INTENSITY_KG_CO2_PER_KWH,
) -> Dict[str, float]:
    """
    Aggregate annual sustainability impact metrics over time-series DataFrame.
    """
    # 1. Parasitic pumping power excess
    if "delta_P_meas_kPa" in df.columns:
        excess_dp = (df["delta_P_meas_kPa"] - clean_dP_kpa).clip(lower=0.0).values
    else:
        excess_dp = np.zeros(len(df))

    vol_flow = config.COOLING_WATER_MASS_FLOW_KG_S / config.WATER_DENSITY_KG_M3
    power_kw = (vol_flow * (excess_dp * 1000.0)) / (pump_eff * 1000.0)
    total_excess_kwh = float(np.sum(power_kw))  # 1 hour per row

    # 2. Equivalent CO2 emissions
    total_co2_kg = compute_carbon_emissions_kg(total_excess_kwh, carbon_intensity)

    # 3. Chemical biocide events
    chem_events = 0
    if "cleaning_event" in df.columns and "cleaning_type" in df.columns:
        chem_events = int(len(df[(df["cleaning_event"] == 1) & (df["cleaning_type"].str.contains("Biocide", case=False))]))

    biocide_info = track_biocide_discharge(chem_events)

    return {
        "excess_pumping_kwh": round(total_excess_kwh, 2),
        "equivalent_co2_kg": round(total_co2_kg, 2),
        "equivalent_co2_tonnes": round(total_co2_kg / 1000.0, 3),
        "biocide_events_count": chem_events,
        "total_biocide_kg": biocide_info["total_biocide_kg"],
        "within_biocide_permit": biocide_info["within_permit_limit"],
    }
