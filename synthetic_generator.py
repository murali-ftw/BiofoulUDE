#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/synthetic_generator.py

Coupled Biophysical and Thermal-Hydraulic Differential Simulator.
Generates physics-informed, reproducible synthetic time-series of heat-exchanger
biofouling in nuclear power plant secondary cooling systems.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import config
import fouling_indicators


def ratkowsky_growth_factor(T_celsius: float) -> float:
    """
    Normalized Ratkowsky mesophilic cardinal temperature biological growth multiplier:
        mu(T) in [0, 1]
    
    T < 6°C:   Metabolic dormancy (mu -> 0)
    T = 28°C:  Optimal growth peak (mu ~ 1.0)
    T > 45°C:  Thermal denaturation & cell death (mu -> 0)
    """
    T = float(T_celsius)
    T_min = config.T_BIO_MIN_C
    T_opt = config.T_BIO_OPT_C
    T_max = config.T_BIO_MAX_C

    if T <= T_min or T >= T_max:
        return 0.0

    # Cardinal temperature model with inflection
    num = ((T - T_max) * (T - T_min) ** 2)
    denom = (T_opt - T_min) * (
        (T_opt - T_min) * (T - T_opt) - (T_opt - T_max) * (T_opt + T_min - 2.0 * T)
    )
    if denom == 0:
        return 0.0
    mu = num / denom
    return float(max(0.0, min(1.0, mu)))


def apply_cleaning_reset(
    delta_f: float,
    cleaning_type: str = "Mechanical_Balls",
    efficiency: Optional[float] = None,
) -> float:
    """
    Apply a cleaning intervention reset:
        delta_f_new = (1 - eta) * delta_f_old
    """
    if efficiency is None:
        if "chem" in cleaning_type.lower() or "biocide" in cleaning_type.lower():
            eta = config.CHEMICAL_BIOCIDE_EFFICIENCY
        else:
            eta = config.MECHANICAL_CLEANING_EFFICIENCY
    else:
        eta = float(efficiency)

    eta = max(0.50, min(0.99, eta))
    residual = (1.0 - eta) * max(0.0, float(delta_f))
    return float(residual)


# Cleaning actions returned by a cleaning policy
MECHANICAL = "mechanical"
CHEMICAL = "chemical"

# Labels written to the dataset's cleaning_type column
CLEANING_TYPE_LABELS = {MECHANICAL: "Mechanical_Balls", CHEMICAL: "Biocide_Shock"}


@dataclass
class FoulingState:
    """Simulator state carried from one hour to the next."""
    delta_f_m: float = 0.0               # biofilm thickness (m)
    cum_clean_hours: float = 0.0         # conditioning-film age used by the induction lag (h)
    time_since_last_clean: float = 0.0   # hours since the last cleaning
    n_cleanings: int = 0                 # cleanings performed so far


# A cleaning policy is called once per hour, after that hour's growth, as
#     policy(hour, state) -> None | MECHANICAL | CHEMICAL
CleaningPolicy = Callable[[int, FoulingState], Optional[str]]


def legacy_cleaning_policy(hour: int, state: FoulingState) -> Optional[str]:
    """
    Original generator rule: clean every 720 h, or at >= 230 um once 240 h have passed.
    The type is chosen by hour parity, which always gives CHEMICAL on the 720 h schedule.
    """
    if state.time_since_last_clean >= 720.0 or (state.delta_f_m * 1e6 >= 230.0 and state.time_since_last_clean >= 240.0):
        return MECHANICAL if (hour % 2 == 0) else CHEMICAL
    return None


def default_cleaning_policy(hour: int, state: FoulingState) -> Optional[str]:
    """Default cleaning policy (currently the legacy rule)."""
    return legacy_cleaning_policy(hour, state)


def no_cleaning_policy(hour: int, state: FoulingState) -> Optional[str]:
    """Never clean."""
    return None


def step(
    state: FoulingState,
    hour: int,
    cleaning_action: Union[None, str, CleaningPolicy],
    rng: np.random.Generator,
    start_dt: datetime,
    dP_clean_kpa: float,
) -> Tuple[FoulingState, Dict[str, Any]]:
    """
    Advance the simulator by one hour.

    cleaning_action is None, MECHANICAL, CHEMICAL, or a policy callable. A callable is
    evaluated after this hour's growth, on the updated state, and returns one of the others.
    Returns the new state and the dataset record for this hour.
    """
    h = hour
    state = replace(state)
    delta_f_m = state.delta_f_m
    dt = start_dt + timedelta(hours=h)

    # 1. Season categorization
    day_of_year = dt.timetuple().tm_yday
    if day_of_year < 80 or day_of_year >= 355:
        season = "Winter"
    elif day_of_year < 172:
        season = "Spring"
    elif day_of_year < 264:
        season = "Summer"
    else:
        season = "Autumn"

    # 2. Ambient water temperature cycle
    # Seasonal swing: 9°C in winter to 27°C in summer, peak around day 200 (~hour 4800)
    phase = (2.0 * math.pi * (h - 2190.0)) / 8760.0
    diurnal_phase = (2.0 * math.pi * (h % 24)) / 24.0

    T_cool_true = (
        config.ANNUAL_MEAN_WATER_TEMP_C
        + config.SEASONAL_TEMP_AMPLITUDE_C * math.sin(phase)
        + config.DIURNAL_TEMP_AMPLITUDE_C * math.sin(diurnal_phase)
    )

    # 3. Mass flow rate with stochastic turbulence perturbations
    flow_noise = rng.normal(0.0, 0.012)
    mass_flow_true = config.COOLING_WATER_MASS_FLOW_KG_S * (1.0 + flow_noise)

    # 4. Biological growth kinetics
    mu_bio = ratkowsky_growth_factor(T_cool_true)

    # Induction phase multiplier (sigmoidal conditioning film development)
    # Slow initial growth during first 120-168 hours
    ind_progress = (state.cum_clean_hours - config.INDUCTION_TIME_HOURS) / 48.0
    phi_ind = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, ind_progress))))

    # Asymptotic carrying capacity factor
    carry_factor = max(0.0, 1.0 - (delta_f_m / config.BIOFILM_MAX_THICKNESS_M))

    # Biological deposition rate [m/h]
    # Nutrient availability index (higher in spring/summer runoff)
    c_nutrient = 1.0 + 0.3 * math.sin(phase)
    m_dot_dep = config.K_DEPOSITION * mu_bio * c_nutrient * phi_ind * carry_factor

    # Hydraulic wall shear stress calculation
    try:
        dP_true_pa, u_true, f_true = fouling_indicators.compute_hydraulic_pressure_drop(
            mass_flow_kg_s=mass_flow_true,
            biofilm_thickness_m=delta_f_m,
        )
    except Exception:
        # Boundary protection
        delta_f_m = min(delta_f_m, config.BIOFILM_MAX_THICKNESS_M)
        dP_true_pa, u_true, f_true = fouling_indicators.compute_hydraulic_pressure_drop(
            mass_flow_kg_s=mass_flow_true,
            biofilm_thickness_m=delta_f_m,
        )

    # Wall shear stress tau_w = (f / 8) * rho * u^2 [Pa]
    tau_w = (f_true / 8.0) * config.WATER_DENSITY_KG_M3 * (u_true ** 2)

    # Shear detachment rate [m/h]
    m_dot_rem = config.K_REMOVAL * tau_w * delta_f_m

    # Euler integration of biofilm thickness
    d_delta = (m_dot_dep - m_dot_rem) * 1.0  # 1 hour step
    state.delta_f_m = max(0.0, min(config.BIOFILM_MAX_THICKNESS_M, delta_f_m + d_delta))
    state.cum_clean_hours += 1.0
    state.time_since_last_clean += 1.0

    # 5. Cleaning event (decided by the caller or its policy)
    action = cleaning_action(h, state) if callable(cleaning_action) else cleaning_action
    cleaning_event = 0
    cleaning_type = "None"
    if action is not None:
        if action not in CLEANING_TYPE_LABELS:
            raise ValueError(f"Unknown cleaning action: {action!r}")
        cleaning_event = 1
        cleaning_type = CLEANING_TYPE_LABELS[action]
        state.delta_f_m = apply_cleaning_reset(state.delta_f_m, cleaning_type=cleaning_type)
        state.cum_clean_hours = max(24.0, state.cum_clean_hours * 0.20)  # Memory effect
        state.time_since_last_clean = 0.0
        state.n_cleanings += 1
    delta_f_m = state.delta_f_m

    # 6. Physical thermal & hydraulic KPIs
    Rf_true = fouling_indicators.compute_thermal_resistance(delta_f_m)
    U_true = fouling_indicators.compute_degraded_U(config.U_CLEAN_DESIGN_W_M2K, Rf_true)
    thermal_eff_ratio = fouling_indicators.compute_thermal_efficiency_ratio(U_true)
    heat_duty_mw = fouling_indicators.compute_heat_duty_mw(U_true, delta_T_lmtd=14.0)

    # Recalculate true pressure drop after potential cleaning
    dP_true_pa, u_true, f_true = fouling_indicators.compute_hydraulic_pressure_drop(
        mass_flow_kg_s=mass_flow_true,
        biofilm_thickness_m=delta_f_m,
    )
    dP_true_kpa = dP_true_pa / 1000.0

    # 7. Realistic sensor measurement noise (ASTM E230 & flow sensor noise)
    t_sensor_noise = float(rng.normal(0.0, 0.22))  # ~0.22°C thermocouple noise
    p_sensor_noise = float(rng.normal(0.0, 0.015)) # ~1.5% pressure sensor noise
    m_sensor_noise = float(rng.normal(0.0, 0.010)) # ~1.0% mass flow noise

    T_cool_in_meas = T_cool_true + t_sensor_noise
    # Secondary cooling discharge temp based on heat duty
    delta_T_cool = (heat_duty_mw * 1e6) / (mass_flow_true * config.WATER_SPECIFIC_HEAT_J_KG_K)
    T_cool_out_meas = T_cool_in_meas + delta_T_cool + float(rng.normal(0.0, 0.25))

    delta_P_meas_kpa = dP_true_kpa * (1.0 + p_sensor_noise)
    mass_flow_meas = mass_flow_true * (1.0 + m_sensor_noise)

    # Measured heat transfer coefficient with sensor noise propagation
    U_meas = max(800.0, U_true + float(rng.normal(0.0, 25.0)))

    # Diagnostic indices
    dP_ratio = delta_P_meas_kpa / max(1.0, dP_clean_kpa)
    bsi = fouling_indicators.compute_biofouling_severity_index(Rf_true, dP_ratio)

    record = {
        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "time_hours": int(h),
        "season": season,
        "T_cool_in_true": round(T_cool_true, 3),
        "T_cool_in_meas": round(T_cool_in_meas, 3),
        "T_cool_out_meas": round(T_cool_out_meas, 3),
        "T_process_in": 48.0,
        "T_process_out": round(48.0 - delta_T_cool * 0.85, 3),
        "mass_flow_cool_meas": round(mass_flow_meas, 3),
        "fluid_velocity_m_s": round(u_true, 4),
        "reynolds_number": round((config.WATER_DENSITY_KG_M3 * u_true * (config.TUBE_INNER_DIAMETER_M - 2*delta_f_m)) / config.WATER_VISCOSITY_PA_S, 1),
        "wall_shear_Pa": round(tau_w, 3),
        "biofilm_thickness_um": round(delta_f_m * 1e6, 3),
        "Rf_fouling_resistance": round(Rf_true, 7),
        "delta_P_true_kPa": round(dP_true_kpa, 3),
        "delta_P_meas_kPa": round(delta_P_meas_kpa, 3),
        "U_clean_W_m2K": round(config.U_CLEAN_DESIGN_W_M2K, 1),
        "U_overall_meas": round(U_meas, 2),
        "thermal_efficiency_ratio": round(thermal_eff_ratio, 4),
        "heat_duty_MW": round(heat_duty_mw, 3),
        "cleaning_event": int(cleaning_event),
        "cleaning_type": cleaning_type,
        "bsi_score": round(bsi, 2),
        "data_source": "synthetic_nuclear_biofouling",
    }
    return state, record


class BiofoulingSyntheticGenerator:
    """
    Object-oriented simulator orchestrating the coupled thermal-hydraulic
    and microbiological differential state updates.
    """

    def __init__(self, seed: int = config.DEFAULT_RANDOM_SEED):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def generate(
        self,
        hours: int = config.DATASET_DEFAULT_HOURS,
        inject_cleanings: bool = True,
        start_date: str = "2026-01-01 00:00:00",
        cleaning_policy: Optional[CleaningPolicy] = None,
    ) -> pd.DataFrame:
        """
        Generate an hourly dataset.
        cleaning_policy overrides the default policy; inject_cleanings=False means never clean.
        """
        # Reset RNG for strict reproducibility
        self.rng = np.random.default_rng(self.seed)
        start_dt = datetime.fromisoformat(start_date)

        if cleaning_policy is None:
            cleaning_policy = default_cleaning_policy if inject_cleanings else no_cleaning_policy

        # Baseline clean hydraulic pressure drop
        dP_clean_pa, u_clean, _ = fouling_indicators.compute_hydraulic_pressure_drop(
            mass_flow_kg_s=config.COOLING_WATER_MASS_FLOW_KG_S,
            biofilm_thickness_m=0.0
        )
        dP_clean_kpa = dP_clean_pa / 1000.0

        state = FoulingState()  # Initial clean state (0 meters)
        records = []
        for h in range(hours):
            state, record = step(state, h, cleaning_policy, self.rng, start_dt, dP_clean_kpa)
            records.append(record)

        df = pd.DataFrame(records)
        return df


def generate_biofouling_dataset(
    hours: int = config.DATASET_DEFAULT_HOURS,
    seed: int = config.DEFAULT_RANDOM_SEED,
    inject_cleanings: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Functional interface for dataset generation."""
    gen = BiofoulingSyntheticGenerator(seed=seed)
    return gen.generate(hours=hours, inject_cleanings=inject_cleanings, **kwargs)


def run_simulation(
    hours: int = config.DATASET_DEFAULT_HOURS,
    seed: int = config.DEFAULT_RANDOM_SEED,
    inject_cleanings: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Alternative alias for simulation execution."""
    return generate_biofouling_dataset(hours=hours, seed=seed, inject_cleanings=inject_cleanings, **kwargs)
