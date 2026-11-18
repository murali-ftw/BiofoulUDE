"""
FoulingUDE: biofilm-thickness ODE with a learned multiplicative correction on deposition.

Hourly explicit Euler, mirroring synthetic_generator.step():

    dep   = K_dep * r(T) * g_theta(x) * phi_lag(cum_clean_hours) * (1 - delta/delta_max)
    rem   = K_rem * tau_w(v) * delta
    delta <- clamp(delta + dep - rem, 0, delta_max)
    cum_clean_hours += 1; hours_since_cleaning += 1
    on a cleaning: delta <- (1 - eta[type]) * delta, cum <- max(24, 0.2 * cum), hours_since <- 0

The lag phase and carrying capacity are the generator's own terms ("known physics"). What the
generator has and this physics lacks is its seasonal nutrient factor (1 + 0.3 sin(phase)); on
nominal data that is what g_theta has to learn. Drift, scale deposits and blooms (mismatch set)
are absent too.

g_theta = exp(net(x)) with the last layer zero-initialised, so g = 1 and UDE == ODE at init.
Inputs (5 variables, 6 numbers): T, v, season as sin/cos, delta/delta_max, hours_since_cleaning,
normalised with fixed constants from config.
"""

import math
from typing import Dict, Optional

import ml  # noqa: F401
import torch
from torch import nn

import config
from ml import physics_torch as pt

NONE, MECHANICAL, CHEMICAL = 0, 1, 2
CLEANING_CODES = {"mechanical": MECHANICAL, "Mechanical_Balls": MECHANICAL, "chemical": CHEMICAL, "Biocide_Shock": CHEMICAL}
EFFICIENCY = {MECHANICAL: config.MECHANICAL_CLEANING_EFFICIENCY, CHEMICAL: config.CHEMICAL_BIOCIDE_EFFICIENCY}

# Fixed input normalisation
T_LO, T_SPAN = config.T_BIO_MIN_C, config.T_BIO_MAX_C - config.T_BIO_MIN_C
V_NOM = float(pt.hydraulics(torch.tensor(config.COOLING_WATER_MASS_FLOW_KG_S, dtype=torch.float64),
                            torch.tensor(0.0, dtype=torch.float64))[1])   # clean tube, nominal flow (~1.414 m/s)
HSC_SCALE = 720.0
FEATURE_NAMES = ["T_norm", "v_norm", "season_sin", "season_cos", "delta_rel", "hours_since_cleaning_norm"]


def lag_factor(cum_clean_hours: torch.Tensor) -> torch.Tensor:
    """Generator's induction (lag-phase) factor."""
    prog = torch.clamp((cum_clean_hours - config.INDUCTION_TIME_HOURS) / 48.0, -10.0, 10.0)
    return 1.0 / (1.0 + torch.exp(-prog))


def correction_features(T, v, hour, delta, hours_since_cleaning) -> torch.Tensor:
    w = 2.0 * math.pi * hour / 8760.0
    return torch.stack([
        (T - T_LO) / T_SPAN,
        v / V_NOM,
        torch.sin(w),
        torch.cos(w),
        delta / pt.DELTA_MAX,
        torch.clamp(hours_since_cleaning / HSC_SCALE, 0.0, 1.0),
    ], dim=-1)


class FoulingUDE(nn.Module):

    def __init__(self, k_dep: float = config.K_DEPOSITION, k_rem: float = config.K_REMOVAL,
                 hidden: int = 16, use_correction: bool = True):
        super().__init__()
        self.log_k_dep = nn.Parameter(torch.tensor(math.log(k_dep), dtype=torch.float64))
        self.log_k_rem = nn.Parameter(torch.tensor(math.log(k_rem), dtype=torch.float64))
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.use_correction = use_correction

    @property
    def k_dep(self) -> torch.Tensor:
        return torch.exp(self.log_k_dep)

    @property
    def k_rem(self) -> torch.Tensor:
        return torch.exp(self.log_k_rem)

    def log_correction(self, T, v, hour, delta, hours_since_cleaning) -> torch.Tensor:
        """log g; zero when the correction is frozen (plain ODE)."""
        if not self.use_correction:
            return torch.zeros_like(T)
        x = correction_features(T, v, hour, delta, hours_since_cleaning).to(self.net[0].weight.dtype)
        return self.net(x).squeeze(-1).to(T.dtype)

    def step(self, delta, cum, hsc, T, flow, hour, cleaning, g_override=None):
        """One hour. All arguments are [B] tensors; cleaning holds NONE/MECHANICAL/CHEMICAL codes."""
        _, v, _, tau_w = pt.hydraulics(flow, delta)
        if g_override is not None:
            g, log_g = g_override, torch.log(g_override)
        else:
            log_g = self.log_correction(T, v, hour, delta, hsc)
            g = torch.exp(log_g)
        carry = torch.clamp(1.0 - delta / pt.DELTA_MAX, min=0.0)
        dep = self.k_dep.to(delta.dtype) * pt.ratkowsky_growth_factor(T) * g * lag_factor(cum) * carry
        rem = self.k_rem.to(delta.dtype) * tau_w * delta
        delta = torch.clamp(delta + dep - rem, 0.0, pt.DELTA_MAX)
        cum = cum + 1.0
        hsc = hsc + 1.0

        eta = torch.where(cleaning == MECHANICAL, torch.full_like(delta, EFFICIENCY[MECHANICAL]),
                          torch.where(cleaning == CHEMICAL, torch.full_like(delta, EFFICIENCY[CHEMICAL]), torch.zeros_like(delta)))
        cleaned = cleaning != NONE
        delta = (1.0 - eta) * delta
        cum = torch.where(cleaned, torch.clamp(cum * 0.20, min=24.0), cum)
        hsc = torch.where(cleaned, torch.zeros_like(hsc), hsc)
        return delta, cum, hsc, log_g

    def rollout(self, delta0, exog: Dict[str, torch.Tensor], cleaning: torch.Tensor,
                cum0, hours_since_clean0, g_override: Optional[torch.Tensor] = None, return_log_g: bool = False):
        """
        delta0, cum0, hours_since_clean0: [B]. exog: "T", "flow", "hour" as [B, L]. cleaning: [B, L]
        codes (known events, applied at the end of that hour, as in the generator).
        Returns delta after each hour, [B, L] (and log g per hour if requested).
        """
        delta, cum, hsc = delta0, cum0, hours_since_clean0
        out, log_gs = [], []
        for t in range(exog["T"].shape[1]):
            delta, cum, hsc, log_g = self.step(
                delta, cum, hsc, exog["T"][:, t], exog["flow"][:, t], exog["hour"][:, t], cleaning[:, t],
                None if g_override is None else g_override[:, t],
            )
            out.append(delta)
            log_gs.append(log_g)
        deltas = torch.stack(out, dim=1)
        return (deltas, torch.stack(log_gs, dim=1)) if return_log_g else deltas

    @staticmethod
    def observe(delta, flow):
        """delta -> (Rf, U, dP kPa)."""
        return pt.observe(delta, flow)


def cum_clean_hours_from_log(cleaning_hours, upto_hour: int) -> float:
    """
    Replay the generator's lag-phase bookkeeping from commissioning (hour 0, clean) using the
    maintenance log, and return cum_clean_hours at the start of hour `upto_hour`.
    """
    cum = 0.0
    events = sorted(h for h in cleaning_hours if h < upto_hour)
    prev = 0
    for h in events:
        cum += h + 1 - prev
        cum = max(24.0, cum * 0.20)
        prev = h + 1
    return cum + (upto_hour - prev)


def hours_since_cleaning_from_log(cleaning_hours, upto_hour: int) -> float:
    events = [h for h in cleaning_hours if h < upto_hour]
    return float(upto_hour - (max(events) + 1)) if events else float(upto_hour)
