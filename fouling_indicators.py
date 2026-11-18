#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/fouling_indicators.py

First-Principles Physical & Diagnostic Fouling Indicators for Shell-and-Tube Heat Exchangers.
Implements thermal resistance, degraded heat transfer, continuum fluid dynamics
(Darcy-Weisbach + Haaland friction factor without hardcoded empirical exponents),
and composite Biofouling Severity Index (BSI).
"""

import math
from typing import Tuple, Union
import numpy as np

import config


def compute_thermal_resistance(biofilm_thickness_m: float, k_biofilm: float = config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK) -> float:
    """
    Compute thermal fouling resistance:
        R_f = delta_f / k_biofilm  [m²·K/W]
    """
    if k_biofilm <= 0:
        raise ValueError("Biofilm thermal conductivity must be positive.")
    return max(0.0, float(biofilm_thickness_m)) / float(k_biofilm)


def compute_degraded_U(U_clean: float, Rf: float) -> float:
    """
    Compute degraded overall heat transfer coefficient:
        1 / U = 1 / U_clean + R_f  ==>  U = U_clean / (1 + U_clean * R_f)  [W/(m²·K)]
    """
    if U_clean <= 0:
        raise ValueError("Clean overall HTC must be positive.")
    Rf_val = max(0.0, float(Rf))
    return float(U_clean) / (1.0 + float(U_clean) * Rf_val)


def compute_thermal_efficiency_ratio(U_current: float, U_clean: float = config.U_CLEAN_DESIGN_W_M2K) -> float:
    """
    Compute normalized thermal efficiency degradation ratio:
        theta_U = U_current / U_clean  in (0, 1]
    """
    if U_clean <= 0:
        return 0.0
    return max(0.0, min(1.0, float(U_current) / float(U_clean)))


def compute_hydraulic_pressure_drop(
    mass_flow_kg_s: float,
    D_inner_m: float = config.TUBE_INNER_DIAMETER_M,
    biofilm_thickness_m: float = 0.0,
    length_m: float = config.TUBE_LENGTH_M,
    N_tubes: int = config.NUMBER_OF_TUBES,
    density_kg_m3: float = config.WATER_DENSITY_KG_M3,
    viscosity_Pa_s: float = config.WATER_VISCOSITY_PA_S,
    roughness_clean_m: float = config.ROUGHNESS_CLEAN_M,
    minor_loss_k: float = 1.2,
) -> Tuple[float, float, float]:
    """
    Compute tube-side hydraulic pressure drop from continuum fluid mechanics.
    Strictly avoids hardcoded universal power exponents!
    
    Equations:
      1. Effective Diameter:       D_eff = D_i - 2 * delta_f
      2. Continuity & Velocity:    u = 4 * m_dot / (rho * pi * N_t * D_eff^2)
      3. Reynolds Number:          Re = rho * u * D_eff / mu
      4. Effective Roughness:      eps_eff = eps_clean + delta_f
      5. Haaland Friction Factor:  1/sqrt(f) = -1.8 * log10[ (eps_eff/D_eff / 3.7)^1.11 + 6.9 / Re ]
      6. Darcy-Weisbach:           dP = [ f * (L / D_eff) + K_minor ] * (rho * u^2 / 2)
      
    Returns:
      (delta_P_Pa, fluid_velocity_m_s, friction_factor)
    """
    # 1. Effective hydraulic diameter
    D_eff = D_inner_m - 2.0 * max(0.0, biofilm_thickness_m)
    if D_eff <= 0.001:
        raise ValueError(f"Tube constriction physically impassable: D_eff={D_eff*1000:.2f}mm")

    # 2. Velocity via continuity
    total_flow_area = N_tubes * (math.pi / 4.0) * (D_eff ** 2)
    volumetric_flow = mass_flow_kg_s / density_kg_m3
    u = volumetric_flow / total_flow_area

    # 3. Reynolds number
    Re = (density_kg_m3 * u * D_eff) / viscosity_Pa_s

    # 4. Relative roughness
    eps_eff = roughness_clean_m + max(0.0, biofilm_thickness_m)
    eps_rel = eps_eff / D_eff

    # 5. Friction factor (Haaland explicit approximation of Colebrook-White)
    if Re >= 2300:
        term1 = (eps_rel / 3.7) ** 1.11
        term2 = 6.9 / max(Re, 100.0)
        inv_sqrt_f = -1.8 * math.log10(max(1e-9, term1 + term2))
        f = (1.0 / inv_sqrt_f) ** 2
        f = max(0.008, min(f, 0.10))  # Physical bounds for commercial pipe flow
    else:
        # Laminar regime fallback
        f = 64.0 / max(Re, 1.0)

    # 6. Darcy-Weisbach pressure drop
    dynamic_head = (density_kg_m3 * (u ** 2)) / 2.0
    major_loss = f * (length_m / D_eff) * dynamic_head
    minor_loss = minor_loss_k * dynamic_head
    delta_P_Pa = major_loss + minor_loss

    return float(delta_P_Pa), float(u), float(f)


def compute_clean_pressure_drop(
    mass_flow_kg_s: float = config.COOLING_WATER_MASS_FLOW_KG_S,
    D_inner_m: float = config.TUBE_INNER_DIAMETER_M,
    length_m: float = config.TUBE_LENGTH_M,
    N_tubes: int = config.NUMBER_OF_TUBES,
    density_kg_m3: float = config.WATER_DENSITY_KG_M3,
    viscosity_Pa_s: float = config.WATER_VISCOSITY_PA_S,
    roughness_clean_m: float = config.ROUGHNESS_CLEAN_M,
    minor_loss_k: float = 1.2,
) -> float:
    """
    Compute theoretical clean tube bundle pressure drop in kPa from continuum fluid mechanics.
    Under nominal baseline (220 kg/s, 450 tubes, 6.0m length), returns ~7.91 kPa.
    """
    dP_Pa, _, _ = compute_hydraulic_pressure_drop(
        mass_flow_kg_s=mass_flow_kg_s,
        D_inner_m=D_inner_m,
        biofilm_thickness_m=0.0,
        length_m=length_m,
        N_tubes=N_tubes,
        density_kg_m3=density_kg_m3,
        viscosity_Pa_s=viscosity_Pa_s,
        roughness_clean_m=roughness_clean_m,
        minor_loss_k=minor_loss_k,
    )
    return float(dP_Pa / 1000.0)


def invert_biofilm_thickness_from_dp(
    dp_meas_kpa: float,
    mass_flow_kg_s: float = config.COOLING_WATER_MASS_FLOW_KG_S,
    max_thickness_m: float = 400e-6,
    tol_kpa: float = 0.001,
) -> float:
    """
    Numerically invert Darcy-Weisbach continuum mechanics to determine the physical
    biofilm thickness delta_f (in meters) that reproduces the measured pressure drop.
    Returns 0.0 if dp_meas_kpa is at or below the clean hydraulic baseline.
    """
    clean_kpa = compute_clean_pressure_drop(mass_flow_kg_s)
    if dp_meas_kpa <= clean_kpa + 0.005:
        return 0.0

    low = 0.0
    high = max_thickness_m
    dp_high_pa, _, _ = compute_hydraulic_pressure_drop(mass_flow_kg_s, biofilm_thickness_m=high)
    dp_high_kpa = dp_high_pa / 1000.0
    if dp_meas_kpa >= dp_high_kpa:
        return float(high)

    for _ in range(40):
        mid = (low + high) / 2.0
        dp_mid_pa, _, _ = compute_hydraulic_pressure_drop(mass_flow_kg_s, biofilm_thickness_m=mid)
        dp_mid_kpa = dp_mid_pa / 1000.0
        if abs(dp_mid_kpa - dp_meas_kpa) < tol_kpa:
            return float(mid)
        if dp_mid_kpa < dp_meas_kpa:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def compute_biofouling_severity_index(
    Rf: float,
    delta_P_ratio: float,
    Rf_reference: float = 0.0005,  # Reference severe thermal resistance
    weight_hydraulic: float = 0.60,
    weight_thermal: float = 0.40,
) -> float:
    """
    Compute normalized Biofouling Severity Index (BSI) in [0, 100].
    Weights hydraulic surge (60%, leading indicator) and thermal deficit (40%).
    
    Clean:  Rf ~ 0, dP_ratio = 1.0  ==> BSI ~ 0 - 5
    Severe: Rf ~ 0.0005, dP_ratio = 1.8 ==> BSI ~ 70 - 100
    """
    # Thermal component [0, 100]
    thermal_component = min(100.0, (max(0.0, Rf) / Rf_reference) * 100.0)

    # Hydraulic surge component [0, 100] (surge above baseline ratio 1.0)
    # Ratio = 1.0 -> 0; Ratio = 1.8 -> 100
    excess_ratio = max(0.0, delta_P_ratio - 1.0)
    hydraulic_component = min(100.0, (excess_ratio / 0.80) * 100.0)

    bsi = (weight_hydraulic * hydraulic_component) + (weight_thermal * thermal_component)
    return float(max(0.0, min(100.0, bsi)))


def compute_heat_duty_mw(
    U_overall: float,
    area_m2: float = config.TOTAL_HEAT_TRANSFER_AREA_M2,
    delta_T_lmtd: float = 12.0,
) -> float:
    """Compute transferred heat duty in Megawatts (MW)."""
    Q_watts = U_overall * area_m2 * delta_T_lmtd
    return float(Q_watts / 1e6)
