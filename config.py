#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/config.py

Engineering and Physical Constants for Nuclear Secondary Cooling Heat Exchangers.
Defines geometric, fluid-dynamic, biophysical, and economic parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any

# =============================================================================
# Geometric & Design Parameters (Auxiliary Nuclear Heat Exchanger)
# =============================================================================
TUBE_INNER_DIAMETER_M: float = 0.021        # 21 mm inside diameter (titanium/Cu-Ni)
DIAMETER_INNER_M: float = TUBE_INNER_DIAMETER_M
TUBE_LENGTH_M: float = 6.0                  # 6 meters tube length
NUMBER_OF_TUBES: int = 450                  # 450 parallel tube passes
N_TUBES: int = NUMBER_OF_TUBES
TOTAL_HEAT_TRANSFER_AREA_M2: float = 3.141592653589793 * TUBE_INNER_DIAMETER_M * TUBE_LENGTH_M * NUMBER_OF_TUBES  # ~178.1 m²

# =============================================================================
# Clean State Thermal & Hydraulic Design Ratings
# =============================================================================
U_CLEAN_DESIGN_W_M2K: float = 2200.0        # Design overall HTC in clean state (W/(m²·K))
U_CLEAN: float = U_CLEAN_DESIGN_W_M2K
ROUGHNESS_CLEAN_M: float = 1.5e-6           # Clean drawn tube roughness (1.5 micrometers)
CLEAN_PRESSURE_DROP_KPA: float = 7.91       # Nominal clean design tube-bundle pressure drop (from Darcy-Weisbach + Haaland at 220 kg/s)
CLEAN_PRESSURE_DROP_NOMINAL_KPA: float = 7.91

# =============================================================================
# Fluid Physical Properties (Cooling Water at ~20°C)
# =============================================================================
WATER_DENSITY_KG_M3: float = 998.0          # kg/m³
WATER_VISCOSITY_PA_S: float = 1.002e-3      # Pa·s (Dynamic viscosity)
WATER_SPECIFIC_HEAT_J_KG_K: float = 4184.0  # J/(kg·K)
COOLING_WATER_MASS_FLOW_KG_S: float = 220.0 # Nominal cooling water throughput (kg/s)
PUMP_EFFICIENCY: float = 0.75               # 75% centrifugal pump efficiency

# =============================================================================
# Biophysical Kinetics (Microbial Biofilm & Extracellular Polymeric Substances)
# =============================================================================
BIOFILM_THERMAL_CONDUCTIVITY_W_MK: float = 0.60  # EPS thermal conductivity (W/(m·K))
K_BIOFILM: float = BIOFILM_THERMAL_CONDUCTIVITY_W_MK
BIOFILM_MAX_THICKNESS_UM: float = 350.0          # Asymptotic maximum biofilm thickness (micrometers)
BIOFILM_MAX_THICKNESS_M: float = BIOFILM_MAX_THICKNESS_UM * 1e-6

# Ratkowsky Cardinal Temperature Constants
T_BIO_MIN_C: float = 6.0    # Lower limit for biological metabolic growth (°C)
T_BIO_OPT_C: float = 28.0   # Optimal mesophilic bacterial growth temperature (°C)
T_BIO_MAX_C: float = 45.0   # Thermal inhibition & denaturation threshold (°C)

# Deposition and Shear Detachment Constants (Bott-Pinheiro Model)
K_DEPOSITION: float = 1.6e-7    # m/h deposition rate constant
K_REMOVAL: float = 4.2e-6       # 1/(Pa·h) shear removal coefficient (removal rate = K_REMOVAL * tau_w * delta)
INDUCTION_TIME_HOURS: float = 168.0  # 7-day initial conditioning lag phase

# Ambient Cooling Water Temperature Profile (Temperate Zone River/Lake Intake)
ANNUAL_MEAN_WATER_TEMP_C: float = 18.0
SEASONAL_TEMP_AMPLITUDE_C: float = 9.0  # Yields 9°C in winter to 27°C in summer
DIURNAL_TEMP_AMPLITUDE_C: float = 1.2   # Day/night swing

# =============================================================================
# Cleaning Intervention Dynamics
# =============================================================================
MECHANICAL_CLEANING_EFFICIENCY: float = 0.90   # 90% biofilm removal (Taprogge sponge balls)
CHEMICAL_BIOCIDE_EFFICIENCY: float = 0.95      # 95% biofilm removal (hypochlorite shock)
ANNUAL_BIOCIDE_PERMIT_LIMIT_KG: float = 250.0  # Annual environmental regulatory discharge limit
BIOCIDE_KG_PER_EVENT: float = 15.0             # Chemical dose per shock event

# =============================================================================
# Economic & Carbon Sustainability Constants
# =============================================================================
ELECTRICITY_COST_PER_KWH: float = 0.12         # $0.12 / kWh
THERMAL_PENALTY_COST_PER_MWH_LOSS: float = 35.0 # Cost per MWh thermal deficit
MECHANICAL_CLEANING_COST: float = 3500.0       # Cost per mechanical intervention
CHEMICAL_CLEANING_COST: float = 6500.0         # Cost per chemical biocide shock
DOWNTIME_HOURS_PER_CLEANING: float = 4.0       # Scheduled outage duration (hours)
DOWNTIME_COST_PER_HOUR: float = 250.0          # Opportunity and technician cost per downtime hour ($/h)
GRID_CARBON_INTENSITY_KG_CO2_PER_KWH: float = 0.45  # 0.45 kg CO2 / kWh

# =============================================================================
# Critical Operational Safety & Operating Thresholds
# =============================================================================
CRITICAL_RF_M2K_W: float = 0.00030             # Maximum allowable thermal fouling resistance (m²·K/W)
CRITICAL_DELTA_P_KPA: float = 16.0             # Hydraulic surge alarm threshold (clean: 7.91 kPa, ~102% surge)
CRITICAL_BSI_SCORE: float = 75.0               # Biofouling Severity Index operational limit [0-100]

# =============================================================================
# Web Dashboard & Service Configuration
# =============================================================================
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8050
DATASET_DEFAULT_HOURS: int = 8760
DEFAULT_RANDOM_SEED: int = 42
