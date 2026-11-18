"""
Torch port of the fouling physics in synthetic_generator.py and fouling_indicators.py.

Same equations and constants as the numpy versions (parity tested to < 1e-5 relative), but
differentiable and vectorised over any tensor shape (typically [batch] per time step, or
[batch, time]). No in-place operations.
"""

import math

import ml  # noqa: F401
import torch

import config

K_BIOFILM = config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK
U_CLEAN = config.U_CLEAN_DESIGN_W_M2K
DELTA_MAX = config.BIOFILM_MAX_THICKNESS_M
MINOR_LOSS_K = 1.2           # fouling_indicators.compute_hydraulic_pressure_drop default
_MIN_D_EFF = 0.0011          # numpy raises below 1 mm; delta <= DELTA_MAX keeps D_eff ~ 20.3 mm


def ratkowsky_growth_factor(T: torch.Tensor) -> torch.Tensor:
    """Cardinal-temperature growth factor in [0, 1]; mirrors synthetic_generator.ratkowsky_growth_factor."""
    t_min, t_opt, t_max = config.T_BIO_MIN_C, config.T_BIO_OPT_C, config.T_BIO_MAX_C
    num = (T - t_max) * (T - t_min) ** 2
    denom = (t_opt - t_min) * ((t_opt - t_min) * (T - t_opt) - (t_opt - t_max) * (t_opt + t_min - 2.0 * T))
    safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    mu = torch.clamp(num / safe_denom, 0.0, 1.0)
    inside = (T > t_min) & (T < t_max) & (denom != 0)
    return torch.where(inside, mu, torch.zeros_like(mu))


def hydraulics(mass_flow_kg_s: torch.Tensor, delta_m: torch.Tensor):
    """
    Darcy-Weisbach + Haaland with the film narrowing and roughening the tube.
    Returns (delta_P [Pa], velocity [m/s], friction factor, wall shear tau_w [Pa]).
    """
    delta = torch.clamp(delta_m, min=0.0)
    d_eff = torch.clamp(config.TUBE_INNER_DIAMETER_M - 2.0 * delta, min=_MIN_D_EFF)
    area = config.NUMBER_OF_TUBES * (math.pi / 4.0) * d_eff ** 2
    u = (mass_flow_kg_s / config.WATER_DENSITY_KG_M3) / area
    re = config.WATER_DENSITY_KG_M3 * u * d_eff / config.WATER_VISCOSITY_PA_S
    eps_rel = (config.ROUGHNESS_CLEAN_M + delta) / d_eff

    term1 = (eps_rel / 3.7) ** 1.11
    term2 = 6.9 / torch.clamp(re, min=100.0)
    inv_sqrt_f = -1.8 * torch.log10(torch.clamp(term1 + term2, min=1e-9))
    f_turb = torch.clamp((1.0 / inv_sqrt_f) ** 2, 0.008, 0.10)
    f_lam = 64.0 / torch.clamp(re, min=1.0)
    f = torch.where(re >= 2300.0, f_turb, f_lam)

    dyn = config.WATER_DENSITY_KG_M3 * u ** 2 / 2.0
    dp = f * (config.TUBE_LENGTH_M / d_eff) * dyn + MINOR_LOSS_K * dyn
    tau_w = (f / 8.0) * config.WATER_DENSITY_KG_M3 * u ** 2
    return dp, u, f, tau_w


def rf_from_delta(delta_m: torch.Tensor) -> torch.Tensor:
    return torch.clamp(delta_m, min=0.0) / K_BIOFILM


def u_from_rf(rf: torch.Tensor) -> torch.Tensor:
    """1/U = 1/U_clean + Rf."""
    return U_CLEAN / (1.0 + U_CLEAN * torch.clamp(rf, min=0.0))


def observe(delta_m: torch.Tensor, mass_flow_kg_s: torch.Tensor):
    """Observation model: delta -> (Rf [m²K/W], U [W/m²K], dP [kPa])."""
    rf = rf_from_delta(delta_m)
    dp_pa, _, _, _ = hydraulics(mass_flow_kg_s, delta_m)
    return rf, u_from_rf(rf), dp_pa / 1000.0


# -----------------------------------------------------------------------------
# Inversions (no gradient needed): used for state initialisation from observations
# -----------------------------------------------------------------------------

def delta_from_u(u_meas: torch.Tensor) -> torch.Tensor:
    """delta = k_biofilm * (1/U - 1/U_clean), clipped to [0, DELTA_MAX]."""
    rf = 1.0 / u_meas - 1.0 / U_CLEAN
    return torch.clamp(rf * K_BIOFILM, 0.0, DELTA_MAX)


@torch.no_grad()
def delta_from_dp(dp_kpa: torch.Tensor, mass_flow_kg_s: torch.Tensor, iterations: int = 50) -> torch.Tensor:
    """Vectorised bisection of dP(delta, flow) = dP_meas on [0, DELTA_MAX] (dP rises with delta)."""
    lo = torch.zeros_like(dp_kpa)
    hi = torch.full_like(dp_kpa, DELTA_MAX)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        dp_mid = hydraulics(mass_flow_kg_s, mid)[0] / 1000.0
        below = dp_mid < dp_kpa
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)
