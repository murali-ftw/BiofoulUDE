#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cleaning_optimizer.py

Cleaning-schedule evaluation and interval optimisation by forward re-simulation.

Every strategy is evaluated the same way:
  1. simulate_policy() runs the synthetic generator's step function forward from a given state,
     so each candidate schedule's own cleanings change the fouling that follows;
  2. cost_breakdown() prices the resulting trajectory using config.py constants only.
No strategy reads fouling from a trajectory generated under a different cleaning schedule.

All results are synthetic and for academic demonstration only.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
import fouling_indicators
import sustainability_metrics
import synthetic_generator
from synthetic_generator import (
    CHEMICAL,
    MECHANICAL,
    NOMINAL_SCENARIO,
    CleaningPolicy,
    FoulingState,
    GeneratorScenario,
)

DEFAULT_SWEEP_DAYS: Tuple[int, ...] = tuple(range(1, 91))
DEFAULT_CONDITION_FRACTIONS: Tuple[float, ...] = tuple(round(0.40 + 0.05 * i, 2) for i in range(12))  # 0.40..0.95
_START_DATE = "2026-01-01 00:00:00"


# =============================================================================
# Cost model (config.py constants only)
# =============================================================================

@dataclass(frozen=True)
class CostParameters:
    mechanical_cleaning_cost: float = config.MECHANICAL_CLEANING_COST
    chemical_cleaning_cost: float = config.CHEMICAL_CLEANING_COST
    downtime_hours_per_cleaning: float = config.DOWNTIME_HOURS_PER_CLEANING
    downtime_cost_per_hour: float = config.DOWNTIME_COST_PER_HOUR
    electricity_cost_per_kwh: float = config.ELECTRICITY_COST_PER_KWH
    thermal_penalty_cost_per_mwh: float = config.THERMAL_PENALTY_COST_PER_MWH_LOSS
    biocide_kg_per_event: float = config.BIOCIDE_KG_PER_EVENT


DEFAULT_COSTS = CostParameters()


def fouling_penalty_rates(delta_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-hour penalties caused by a biofilm of thickness delta_m (m), at nominal flow:
      extra pump power  [kW] = (dP(delta) - dP(0)) * V_dot / PUMP_EFFICIENCY
      heat-duty shortfall [MW] = Q(U_clean) - Q(U(delta)),  Q = U * A * DESIGN_LMTD_K
    """
    m_dot = config.COOLING_WATER_MASS_FLOW_KG_S
    vol_flow = m_dot / config.WATER_DENSITY_KG_M3
    dp_clean_kpa = fouling_indicators.compute_hydraulic_pressure_drop(m_dot, biofilm_thickness_m=0.0)[0] / 1000.0
    q_clean_mw = fouling_indicators.compute_heat_duty_mw(config.U_CLEAN_DESIGN_W_M2K)

    pump_kw = np.empty(len(delta_m))
    shortfall_mw = np.empty(len(delta_m))
    for i, d in enumerate(delta_m):
        dp_kpa = fouling_indicators.compute_hydraulic_pressure_drop(m_dot, biofilm_thickness_m=float(d))[0] / 1000.0
        pump_kw[i] = sustainability_metrics.compute_parasitic_pumping_power_kw(vol_flow, dp_kpa - dp_clean_kpa)
        u = fouling_indicators.compute_degraded_U(
            config.U_CLEAN_DESIGN_W_M2K, fouling_indicators.compute_thermal_resistance(float(d))
        )
        shortfall_mw[i] = q_clean_mw - fouling_indicators.compute_heat_duty_mw(u)
    return pump_kw, shortfall_mw


def cost_breakdown(
    delta_m: np.ndarray,
    cleaning_types: Sequence[str],
    costs: CostParameters = DEFAULT_COSTS,
) -> Dict[str, float]:
    """
    Shared cost function for every strategy. delta_m is the hourly biofilm thickness (m);
    cleaning_types lists each cleaning as MECHANICAL or CHEMICAL.

    total_cost = cleaning_cost + downtime_cost + pumping_energy_cost + thermal_penalty_cost
    """
    n_mech = sum(1 for t in cleaning_types if t == MECHANICAL)
    n_chem = sum(1 for t in cleaning_types if t == CHEMICAL)
    n_events = n_mech + n_chem

    pump_kw, shortfall_mw = fouling_penalty_rates(np.asarray(delta_m, dtype=float))
    pump_kwh = float(pump_kw.sum())         # 1 h per sample
    shortfall_mwh = float(shortfall_mw.sum())

    cleaning_cost = n_mech * costs.mechanical_cleaning_cost + n_chem * costs.chemical_cleaning_cost
    downtime_hours = n_events * costs.downtime_hours_per_cleaning
    downtime_cost = downtime_hours * costs.downtime_cost_per_hour
    pumping_energy_cost = pump_kwh * costs.electricity_cost_per_kwh
    thermal_penalty_cost = shortfall_mwh * costs.thermal_penalty_cost_per_mwh

    return {
        "cleaning_cost": cleaning_cost,
        "downtime_cost": downtime_cost,
        "pumping_energy_cost": pumping_energy_cost,
        "thermal_penalty_cost": thermal_penalty_cost,
        "total_cost": cleaning_cost + downtime_cost + pumping_energy_cost + thermal_penalty_cost,
        "cleanings_mechanical": n_mech,
        "cleanings_chemical": n_chem,
        "cleanings_total": n_events,
        "downtime_hours": downtime_hours,
        "biocide_kg": n_chem * costs.biocide_kg_per_event,
        "pump_energy_kwh": pump_kwh,
        "thermal_shortfall_mwh": shortfall_mwh,
        "co2_kg": sustainability_metrics.compute_carbon_emissions_kg(pump_kwh),
    }


# =============================================================================
# Policies
# =============================================================================

def fixed_interval_policy(interval_days: float) -> CleaningPolicy:
    """Clean every interval_days, alternating mechanical (1st, 3rd, ...) and chemical."""
    return synthetic_generator.fixed_interval_policy(interval_days * 24.0)


def never_clean_policy() -> CleaningPolicy:
    return synthetic_generator.no_cleaning_policy


def immediate_once_policy() -> CleaningPolicy:
    """Clean once in the first simulated hour, then never again."""
    def policy(hour: int, state: FoulingState) -> Optional[str]:
        return MECHANICAL if state.n_cleanings == 0 else None
    return policy


def condition_based_policy(
    rf_fraction: float = 0.8,
    dp_fraction: float = 0.8,
    min_hours_between: float = 24.0,
) -> CleaningPolicy:
    """
    Clean when Rf >= rf_fraction * CRITICAL_RF_M2K_W or dP >= dp_fraction * CRITICAL_DELTA_P_KPA,
    alternating mechanical and chemical. Rf and dP are computed from the true simulated state at
    nominal flow, so this is an idealised (perfectly observed) condition-based policy.
    """
    rf_trigger = rf_fraction * config.CRITICAL_RF_M2K_W
    dp_trigger_kpa = dp_fraction * config.CRITICAL_DELTA_P_KPA

    def policy(hour: int, state: FoulingState) -> Optional[str]:
        if state.time_since_last_clean < min_hours_between:
            return None
        rf = fouling_indicators.compute_thermal_resistance(state.delta_f_m)
        dp_kpa = fouling_indicators.compute_hydraulic_pressure_drop(
            config.COOLING_WATER_MASS_FLOW_KG_S, biofilm_thickness_m=state.delta_f_m
        )[0] / 1000.0
        if rf >= rf_trigger or dp_kpa >= dp_trigger_kpa:
            return MECHANICAL if (state.n_cleanings % 2 == 0) else CHEMICAL
        return None

    return policy


# =============================================================================
# Forward simulation
# =============================================================================

@dataclass
class PolicyRun:
    """Trajectory of one policy: dataset-format records plus the exact state and cleaning log."""
    df: pd.DataFrame
    delta_m: np.ndarray
    cleaning_types: List[str] = field(default_factory=list)


def simulate_policy(
    policy: CleaningPolicy,
    hours: int = config.DATASET_DEFAULT_HOURS,
    seed: int = config.DEFAULT_RANDOM_SEED,
    scenario: GeneratorScenario = NOMINAL_SCENARIO,
    initial_state: Optional[FoulingState] = None,
    start_hour: int = 0,
) -> PolicyRun:
    """
    Simulate `policy` forward for `hours` hours from `initial_state` (clean by default) at
    `start_hour`, with the generator's step function. With the defaults, simulating the
    generator's default policy reproduces db/biofouling_synthetic_dataset.csv exactly.
    The noise sequence depends only on the seed, so policies are compared on identical noise.
    """
    rng = np.random.default_rng(int(seed))
    start_dt = datetime.fromisoformat(_START_DATE)
    dp_clean_kpa = fouling_indicators.compute_hydraulic_pressure_drop(
        config.COOLING_WATER_MASS_FLOW_KG_S, biofilm_thickness_m=0.0
    )[0] / 1000.0

    state = replace(initial_state) if initial_state is not None else FoulingState()
    records, delta_m, cleaning_types = [], np.empty(hours), []
    for i, h in enumerate(range(start_hour, start_hour + hours)):
        n_before = state.n_cleanings
        state, record = synthetic_generator.step(state, h, policy, rng, start_dt, dp_clean_kpa, scenario)
        records.append(record)
        delta_m[i] = state.delta_f_m
        if state.n_cleanings > n_before:
            cleaning_types.append(MECHANICAL if record["cleaning_type"] == "Mechanical_Balls" else CHEMICAL)
    return PolicyRun(pd.DataFrame(records), delta_m, cleaning_types)


def evaluate_run(run: PolicyRun, costs: CostParameters = DEFAULT_COSTS) -> Dict[str, Any]:
    """
    Cost breakdown, cleanings, peaks, hours above any limit, and feasibility for one run.
    Feasible = Rf, true dP and BSI never above their limits, and biocide within the permit.
    """
    result = cost_breakdown(run.delta_m, run.cleaning_types, costs)
    crossings = fouling_indicators.first_limit_crossings(run.df)
    # Annual biocide permit, pro-rated for runs shorter than a year
    biocide_limit_kg = config.ANNUAL_BIOCIDE_PERMIT_LIMIT_KG * len(run.df) / 8760.0
    within_biocide_permit = result["biocide_kg"] <= biocide_limit_kg
    above_any = (
        (run.df["Rf_fouling_resistance"] > config.CRITICAL_RF_M2K_W)
        | (run.df["delta_P_true_kPa"] > config.CRITICAL_DELTA_P_KPA)
        | (run.df["bsi_score"] > config.CRITICAL_BSI_SCORE)
    )
    first_breach = None
    for name, info in crossings.items():
        if info["first_hour"] is not None and (first_breach is None or info["first_hour"] < first_breach["hour"]):
            first_breach = {"metric": name, "hour": info["first_hour"], "limit": info["limit"]}
    result.update({
        "hours": len(run.df),
        "peak_rf": crossings["Rf"]["peak"],
        "peak_dp_kpa": crossings["dP"]["peak"],
        "peak_bsi": crossings["BSI"]["peak"],
        "min_thermal_efficiency": float(run.df["thermal_efficiency_ratio"].min()),
        "hours_above_limit": int(above_any.sum()),
        "first_breach": first_breach,
        "limit_crossings": crossings,
        "biocide_limit_kg": biocide_limit_kg,
        "within_biocide_permit": bool(within_biocide_permit),
        "is_feasible": bool(not above_any.any() and within_biocide_permit),
    })
    return result


def evaluate_policy(
    policy: CleaningPolicy,
    hours: int = config.DATASET_DEFAULT_HOURS,
    seed: int = config.DEFAULT_RANDOM_SEED,
    scenario: GeneratorScenario = NOMINAL_SCENARIO,
    initial_state: Optional[FoulingState] = None,
    start_hour: int = 0,
    costs: CostParameters = DEFAULT_COSTS,
) -> Dict[str, Any]:
    run = simulate_policy(policy, hours, seed, scenario, initial_state, start_hour)
    return evaluate_run(run, costs)


def sweep_fixed_intervals(
    days: Iterable[int] = DEFAULT_SWEEP_DAYS,
    hours: int = config.DATASET_DEFAULT_HOURS,
    seed: int = config.DEFAULT_RANDOM_SEED,
    scenario: GeneratorScenario = NOMINAL_SCENARIO,
    costs: CostParameters = DEFAULT_COSTS,
) -> Dict[str, Any]:
    """Exact sweep of fixed intervals over one simulated period; best = cheapest feasible."""
    results = []
    for d in days:
        r = evaluate_policy(fixed_interval_policy(d), hours, seed, scenario, costs=costs)
        r["interval_days"] = int(d)
        results.append(r)
    feasible = [r for r in results if r["is_feasible"]]
    best = min(feasible, key=lambda r: r["total_cost"]) if feasible else None
    return {"results": results, "best": best, "any_feasible": best is not None}


# =============================================================================
# Annual strategy comparison
# =============================================================================

class CleaningScheduleOptimizer:
    """Annual evaluation of cleaning strategies, all by re-simulation and the shared cost function."""

    def __init__(self, costs: CostParameters = DEFAULT_COSTS):
        self.costs = costs

    def evaluate_fixed_calendar_schedule(
        self,
        interval_days: int = 30,
        hours: int = config.DATASET_DEFAULT_HOURS,
        seed: int = config.DEFAULT_RANDOM_SEED,
        scenario: GeneratorScenario = NOMINAL_SCENARIO,
    ) -> Dict[str, Any]:
        r = evaluate_policy(fixed_interval_policy(interval_days), hours, seed, scenario, costs=self.costs)
        r.update({"strategy": f"Fixed_{interval_days}d", "interval_days": interval_days})
        return r

    def evaluate_condition_based_schedule(
        self,
        rf_fraction: float = 0.8,
        dp_fraction: float = 0.8,
        hours: int = config.DATASET_DEFAULT_HOURS,
        seed: int = config.DEFAULT_RANDOM_SEED,
        scenario: GeneratorScenario = NOMINAL_SCENARIO,
    ) -> Dict[str, Any]:
        r = evaluate_policy(condition_based_policy(rf_fraction, dp_fraction), hours, seed, scenario, costs=self.costs)
        r.update({"strategy": f"Condition_{rf_fraction:g}xRf_{dp_fraction:g}xdP"})
        return r

    def evaluate_never_clean(
        self,
        hours: int = config.DATASET_DEFAULT_HOURS,
        seed: int = config.DEFAULT_RANDOM_SEED,
        scenario: GeneratorScenario = NOMINAL_SCENARIO,
    ) -> Dict[str, Any]:
        r = evaluate_policy(never_clean_policy(), hours, seed, scenario, costs=self.costs)
        r.update({"strategy": "Never_clean"})
        return r

    def best_fixed_interval(
        self,
        days: Iterable[int] = DEFAULT_SWEEP_DAYS,
        hours: int = config.DATASET_DEFAULT_HOURS,
        seed: int = config.DEFAULT_RANDOM_SEED,
        scenario: GeneratorScenario = NOMINAL_SCENARIO,
    ) -> Dict[str, Any]:
        sweep = sweep_fixed_intervals(days, hours, seed, scenario, self.costs)
        if sweep["best"] is not None:
            sweep["best"]["strategy"] = f"Best_fixed_{sweep['best']['interval_days']}d"
        return sweep

    def best_condition_based(
        self,
        fractions: Iterable[float] = DEFAULT_CONDITION_FRACTIONS,
        hours: int = config.DATASET_DEFAULT_HOURS,
        seed: int = config.DEFAULT_RANDOM_SEED,
        scenario: GeneratorScenario = NOMINAL_SCENARIO,
    ) -> Optional[Dict[str, Any]]:
        """Cheapest feasible condition-based policy over trigger fractions (same fraction for Rf and dP)."""
        results = [self.evaluate_condition_based_schedule(f, f, hours, seed, scenario) for f in fractions]
        feasible = [r for r in results if r["is_feasible"]]
        if not feasible:
            return None
        best = min(feasible, key=lambda r: r["total_cost"])
        best["strategy"] = "Best_" + best["strategy"]
        return best

    def compare_strategies(
        self,
        hours: int = config.DATASET_DEFAULT_HOURS,
        seed: int = config.DEFAULT_RANDOM_SEED,
        scenario: GeneratorScenario = NOMINAL_SCENARIO,
    ) -> List[Dict[str, Any]]:
        """
        Fixed 30 d; best fixed interval (1..90 d sweep); condition-based at 0.8 x limits;
        best condition-based (fraction sweep); never clean.
        """
        rows = [self.evaluate_fixed_calendar_schedule(30, hours, seed, scenario)]
        sweep = self.best_fixed_interval(hours=hours, seed=seed, scenario=scenario)
        rows.append(sweep["best"] if sweep["best"] is not None else {"strategy": "Best_fixed (none feasible)"})
        rows.append(self.evaluate_condition_based_schedule(hours=hours, seed=seed, scenario=scenario))
        best_cb = self.best_condition_based(hours=hours, seed=seed, scenario=scenario)
        rows.append(best_cb if best_cb is not None else {"strategy": "Best_condition (none feasible)"})
        rows.append(self.evaluate_never_clean(hours, seed, scenario))
        return rows


def format_comparison_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        f"{'strategy':32s} {'total $':>10s} {'clean $':>8s} {'down $':>7s} {'pump $':>7s} {'therm $':>8s} "
        f"{'#M':>3s} {'#C':>3s} {'bioc kg':>7s} {'peak Rf':>9s} {'pk dP':>6s} {'pk BSI':>6s} {'h>lim':>5s} feasible"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        if "total_cost" not in r:
            lines.append(f"{r['strategy']:32s} (no feasible candidate)")
            continue
        lines.append(
            f"{r['strategy']:32s} {r['total_cost']:10,.0f} {r['cleaning_cost']:8,.0f} {r['downtime_cost']:7,.0f} "
            f"{r['pumping_energy_cost']:7,.0f} {r['thermal_penalty_cost']:8,.0f} {r['cleanings_mechanical']:3d} "
            f"{r['cleanings_chemical']:3d} {r['biocide_kg']:7.0f} {r['peak_rf']:9.2e} {r['peak_dp_kpa']:6.2f} "
            f"{r['peak_bsi']:6.1f} {r['hours_above_limit']:5d} "
            f"{'yes' if r['is_feasible'] else ('NO (biocide)' if r['hours_above_limit'] == 0 else 'NO')}"
        )
    return "\n".join(lines)


# =============================================================================
# Interval optimiser from the current state
# =============================================================================

class CleaningIntervalOptimizer:
    """
    Recommends the next fixed cleaning interval from the current fouling state by an exact sweep
    over candidate intervals, each simulated forward with simulate_policy and priced with
    cost_breakdown. Candidates are ranked by cost per day over max(planning horizon, interval),
    so every candidate's evaluation window contains its own first cleaning.

    TODO(UDE): replace the generator physics in the forward simulation with the UDE forecast
    ensemble and add the chance constraint P(Rf reaches CRITICAL_RF_M2K_W before T) <= epsilon
    (see docs/model.md, Section 6). Today this optimiser uses no ML forecast.
    """

    def __init__(
        self,
        costs: CostParameters = DEFAULT_COSTS,
        seed: int = config.DEFAULT_RANDOM_SEED,
        scenario: GeneratorScenario = NOMINAL_SCENARIO,
        cleaning_cost: Optional[float] = None,
    ):
        if cleaning_cost is not None:
            # Compatibility: a single cleaning cost applies to both cleaning types.
            costs = replace(costs, mechanical_cleaning_cost=float(cleaning_cost), chemical_cleaning_cost=float(cleaning_cost))
        self.costs = costs
        self.seed = int(seed)
        self.scenario = scenario
        self.rf_crit = config.CRITICAL_RF_M2K_W
        self.dp_crit = config.CRITICAL_DELTA_P_KPA
        self.bsi_crit = config.CRITICAL_BSI_SCORE

    @staticmethod
    def _initial_state(current_biofilm_um, current_rf=None, current_dp_kpa=None) -> FoulingState:
        if current_biofilm_um is not None and current_biofilm_um > 0.0:
            delta_m = current_biofilm_um * 1e-6
        elif current_rf is not None and current_rf > 0.0:
            delta_m = current_rf * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK
        elif current_dp_kpa is not None and current_dp_kpa > 0.0:
            delta_m = fouling_indicators.invert_biofilm_thickness_from_dp(current_dp_kpa)
        else:
            delta_m = 0.0
        # Assumption: an already fouled tube has a mature conditioning film (no induction lag).
        cum_clean_hours = 168.0 if delta_m > 30e-6 else 24.0
        return FoulingState(delta_f_m=delta_m, cum_clean_hours=cum_clean_hours)

    @staticmethod
    def _check_unsupported(mass_flow_kg_s, operating_temp_c):
        if mass_flow_kg_s is not None or operating_temp_c is not None:
            raise NotImplementedError(
                "mass_flow_kg_s / operating_temp_c overrides are not supported by the re-simulation "
                "optimiser; it uses the generator's nominal flow and seasonal temperature."
            )

    def find_uncleaned_threshold_crossing(
        self,
        current_biofilm_um: float,
        current_hour: int,
        max_lookahead_hours: int = 2160,
        mass_flow_kg_s: Optional[float] = None,
        operating_temp_c: Optional[float] = None,
        current_rf: Optional[float] = None,
        current_dp_kpa: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Simulate without cleaning and report the first limit crossed, if any."""
        self._check_unsupported(mass_flow_kg_s, operating_temp_c)
        state = self._initial_state(current_biofilm_um, current_rf, current_dp_kpa)
        run = simulate_policy(never_clean_policy(), max_lookahead_hours, self.seed, self.scenario, state, current_hour)
        breach = evaluate_run(run, self.costs)["first_breach"]
        if breach is None:
            return {"breached": False, "breach_hour_relative": None, "breach_days_relative": None,
                    "limiting_parameter": None, "breach_value": None, "threshold_limit": None, "unit": None}
        names = {"Rf": ("Rf_fouling_resistance", "Rf_fouling_resistance", "m²·K/W"),
                 "dP": ("delta_P_true_kPa", "delta_P_true_kPa", "kPa"),
                 "BSI": ("bsi_score", "bsi_score", "index [0-100]")}
        param, col, unit = names[breach["metric"]]
        rel = breach["hour"] - current_hour + 1
        return {
            "breached": True,
            "breach_hour_relative": rel,
            "breach_hour_absolute": breach["hour"],
            "breach_days_relative": round(rel / 24.0, 1),
            "limiting_parameter": param,
            "breach_value": float(run.df.loc[run.df["time_hours"] == breach["hour"], col].iloc[0]),
            "threshold_limit": breach["limit"],
            "unit": unit,
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
        Simulate one candidate interval from the current state over max(horizon, interval) days.
        Candidate 0 means one immediate cleaning and none afterwards within the horizon.
        """
        self._check_unsupported(mass_flow_kg_s, operating_temp_c)
        state = self._initial_state(current_biofilm_um, current_rf, current_dp_kpa)
        days = int(candidate_interval_days)
        window_days = max(int(planning_horizon_days), days)
        policy = immediate_once_policy() if days == 0 else fixed_interval_policy(days)
        r = evaluate_policy(policy, window_days * 24, self.seed, self.scenario, state, current_hour, self.costs)
        r.update({
            "strategy": "Immediate (0d)" if days == 0 else f"{days}-day Cleaning",
            "candidate_interval_days": days,
            "candidate_interval_hours": days * 24,
            "evaluation_window_days": window_days,
            "cost_per_day": r["total_cost"] / window_days,
            # Compatibility keys
            "intervention_count": r["cleanings_total"],
            "pump_energy_cost": r["pumping_energy_cost"],
            "thermal_deficit_mwh": r["thermal_shortfall_mwh"],
            "total_projected_cost": r["total_cost"],
            "max_rf": r["peak_rf"],
            "max_delta_P_kPa": r["peak_dp_kpa"],
            "breach_info": r["first_breach"],
        })
        return r

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
        """Exact sweep (default T = 1..90 days); pick the feasible candidate with the lowest cost per day."""
        if candidates is None:
            candidates = list(DEFAULT_SWEEP_DAYS)
        crossing = self.find_uncleaned_threshold_crossing(
            current_biofilm_um, current_hour, mass_flow_kg_s=mass_flow_kg_s, operating_temp_c=operating_temp_c,
            current_rf=current_rf, current_dp_kpa=current_dp_kpa,
        )
        scenarios = [
            self.evaluate_scenario(current_biofilm_um, current_hour, c, planning_horizon_days,
                                   current_rf=current_rf, current_dp_kpa=current_dp_kpa)
            for c in candidates
        ]
        feasible = [s for s in scenarios if s["is_feasible"]]
        fixed_30 = next((s for s in scenarios if s["candidate_interval_days"] == 30), None)

        if feasible:
            best = min(feasible, key=lambda s: s["cost_per_day"])
            rec_days = best["candidate_interval_days"]
            if rec_days == 0:
                window = "Immediate (0 - 2 days)"
            else:
                window = f"Day {max(1, rec_days - 2)} to Day {rec_days + 3}"
            reason = (
                f"{rec_days}-day interval has the lowest cost per day (${best['cost_per_day']:,.2f}/day over "
                f"{best['evaluation_window_days']} days) among {len(feasible)} feasible of {len(scenarios)} candidates, "
                f"by forward re-simulation from the current state."
            )
            status, is_feasible = "OPTIMAL_FEASIBLE", True
        else:
            best = None
            rec_days = 0
            window = "Immediate (0 - 24 hours)"
            limit_txt = (
                f"the {crossing['limiting_parameter']} limit ({crossing['threshold_limit']}) in "
                f"{crossing['breach_days_relative']} days" if crossing["breached"] else "a limit"
            )
            reason = f"No candidate interval is feasible: without cleaning the exchanger crosses {limit_txt}. Immediate cleaning is indicated."
            status, is_feasible = "CRITICAL_INFEASIBLE_EMERGENCY", False

        return {
            "status": status,
            "is_feasible": is_feasible,
            "recommended_interval_days": rec_days,
            "recommended_interval_hours": rec_days * 24,
            "recommended_window_days": window,
            "planning_horizon_days": planning_horizon_days,
            "primary_reason": reason,
            "threshold_crossing": crossing,
            "best_scenario": best,
            "all_scenarios": scenarios,
            "fixed_calendar_benchmark": fixed_30,
            "thresholds_used": {"rf_crit": self.rf_crit, "dp_crit": self.dp_crit, "bsi_crit": self.bsi_crit},
        }


# Compatibility alias
CleaningOptimizer = CleaningScheduleOptimizer
