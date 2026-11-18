"""
Write canned API responses to web/mock/*.json so the UI can run without trained models.

    python -m web.make_mock

Series come from the db/ CSVs (every 6th hour); forecast / decision / metrics / correction are
hand-built with plausible magnitudes and are labelled MOCK. Nothing here uses ml/ artifacts.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config
from web.server import DATASET_INFO, DISCLAIMER, MOCK_DIR

FILES = {"nominal": "biofouling_synthetic_dataset.csv", "stress_no_clean": "stress_no_clean.csv",
         "stress_long_interval": "stress_long_interval.csv", "mismatch_stress": "mismatch_stress.csv"}
STRIDE = 6


def events(df):
    ev = df[df["cleaning_event"] == 1]
    return [{"hour": int(h), "type": "mechanical" if t == "Mechanical_Balls" else "chemical"}
            for h, t in zip(ev["time_hours"], ev["cleaning_type"])]


def write(name, obj):
    (MOCK_DIR / name).write_text(json.dumps(obj, separators=(",", ":")))


def band(mean, rel):
    mean = np.asarray(mean)
    return {"mean": mean.tolist(), "lo": (mean * (1 - rel)).tolist(), "hi": (mean * (1 + rel)).tolist()}


def main():
    MOCK_DIR.mkdir(exist_ok=True)
    meta_ds = []
    for ds_id, fname in FILES.items():
        df = pd.read_csv(ROOT / "db" / fname, keep_default_na=False)
        label, desc = DATASET_INFO[ds_id]
        meta_ds.append({"id": ds_id, "label": label, "hours": len(df), "description": desc, "cleaning_events": events(df)})
        keep = df[(df["time_hours"] % STRIDE == 0) | (df["cleaning_event"] == 1)]
        write(f"series_{ds_id}.json", {
            "t": keep["time_hours"].astype(int).tolist(), "T_in": keep["T_cool_in_meas"].tolist(),
            "flow": keep["mass_flow_cool_meas"].tolist(), "U": keep["U_overall_meas"].tolist(),
            "dP_kpa": keep["delta_P_meas_kPa"].tolist(), "Rf_true": keep["Rf_fouling_resistance"].tolist(),
            "bsi": keep["bsi_score"].tolist(), "cleaning_events": events(df)})

    write("meta.json", {
        "datasets": meta_ds,
        "limits": {"rf_crit": config.CRITICAL_RF_M2K_W, "dp_crit_kpa": config.CRITICAL_DELTA_P_KPA, "bsi_crit": config.CRITICAL_BSI_SCORE},
        "costs": {"mechanical_cleaning_cost": config.MECHANICAL_CLEANING_COST, "chemical_cleaning_cost": config.CHEMICAL_CLEANING_COST,
                  "downtime_hours_per_cleaning": config.DOWNTIME_HOURS_PER_CLEANING, "downtime_cost_per_hour": config.DOWNTIME_COST_PER_HOUR,
                  "electricity_cost_per_kwh": config.ELECTRICITY_COST_PER_KWH,
                  "thermal_penalty_cost_per_mwh": config.THERMAL_PENALTY_COST_PER_MWH_LOSS,
                  "biocide_kg_per_event": config.BIOCIDE_KG_PER_EVENT, "biocide_permit_kg_per_year": config.ANNUAL_BIOCIDE_PERMIT_LIMIT_KG},
        "model": {"ensemble_size": 5, "k_dep": 1.8e-7, "k_rem": 3.0e-6, "device": "MOCK", "trained_on": "MOCK (no model loaded)"},
        "disclaimer": "MOCK MODE - " + DISCLAIMER,
    })

    origin, H = 3000, 168
    hours = np.arange(origin + 1, origin + 1 + H)
    rf = 1.2e-4 + 2.5e-7 * np.arange(H)
    u = config.U_CLEAN_DESIGN_W_M2K / (1 + config.U_CLEAN_DESIGN_W_M2K * rf)
    dp = 10.0 + 0.004 * np.arange(H)
    write("forecast.json", {
        "hours": hours.tolist(), "Rf": band(rf, 0.08), "U": band(u, 0.01), "dP_kpa": band(dp, 0.02),
        "time_to_limit": {"rf": {"median": float(origin + 700), "lo": float(origin + 560), "hi": float(origin + 900), "fraction_crossing": 1.0},
                          "dp": None},
        "truth": {"Rf": (rf * 1.03).tolist()}, "model": "ude", "notes": ["MOCK forecast with plausible shapes; not a model output"]})

    cands = []
    for d in range(1, 91):
        clean = (config.MECHANICAL_CLEANING_COST + config.CHEMICAL_CLEANING_COST) / 2 / d
        down = config.DOWNTIME_HOURS_PER_CLEANING * config.DOWNTIME_COST_PER_HOUR / d
        therm = 18.0 * d
        pump = 0.05 * d
        risk = min(1.0, max(0.0, (d - 35) / 20.0))
        cands.append({"interval_days": d, "window_days": max(60, d), "total_cost": (clean + down + therm + pump) * max(60, d),
                      "cost_per_day": clean + down + therm + pump,
                      "cost_breakdown": {"cleaning_cost": clean * max(60, d), "downtime_cost": down * max(60, d),
                                         "pumping_energy_cost": pump * max(60, d), "thermal_penalty_cost": therm * max(60, d),
                                         "cleanings_mechanical": 1, "cleanings_chemical": 1, "biocide_kg": 15.0},
                      "feasible": d <= 45, "hours_above_limit": 0 if d <= 45 else 24 * (d - 45), "peak_rf": 1e-4 + 4e-6 * d,
                      "peak_dp_kpa": 9 + 0.1 * d, "risk_limit_cross": risk, "risk_ok": risk <= 0.05})
    ok = [c for c in cands if c["feasible"] and c["risk_ok"]]
    comp = lambda cpd, feas: {"window_days": 90, "total_cost": cpd * 90, "cost_per_day": cpd, "cost_breakdown": {}, "feasible": feas,
                              "hours_above_limit": 0 if feas else 500, "peak_rf": 2e-4, "peak_dp_kpa": 12.0}
    write("decision.json", {"candidates": cands, "recommended_interval_days": min(ok, key=lambda c: c["cost_per_day"])["interval_days"],
                            "compared": {"fixed_30": comp(700, True), "condition_based": comp(900, True), "never_clean": comp(1700, False)},
                            "epsilon": 0.05, "origin_hour": origin, "state_estimate_um": 60.0,
                            "notes": ["MOCK decision; advisory only; operator review and plant procedures remain mandatory"]})

    hz = [1, 24, 72, 168]
    m = lambda base: {str(h): {q: {"mae": base * (1 + h / 50) * s, "rmse": base * 1.3 * (1 + h / 50) * s, "r2": 0.9 - h / 400, "n": 50}
                               for q, s in (("rf", 1e-5), ("u", 30.0), ("dp", 0.2))} for h in hz}
    ds = {name: {"origins": 50, "metrics": {k: m(b) for k, b in (("persistence", 1.0), ("ridge", 1.1), ("gbm", 1.05), ("ode", 0.9), ("ude", 0.7))},
                 "coverage_90": {"ude": {str(h): 0.85 for h in hz}, "ode": {str(h): 0.6 for h in hz}}}
          for name in ("nominal_test", "stress_no_clean", "stress_long_interval", "mismatch_stress")}
    write("metrics.json", {"synthetic": True, "mock": True, "horizons_h": hz, "datasets": ds, "disclaimer": "MOCK metrics"})

    x = np.linspace(0, 365, 61)
    write("correction.json", {"inputs": ["T", "v", "season_sin", "season_cos", "delta/delta_max", "hours_since_cleaning/720"],
                              "slices": [{"name": "day of year", "x": x.tolist(), "g": (1 + 0.25 * np.sin(2 * math.pi * (x - 91) / 365)).tolist()},
                                         {"name": "water temperature (°C)", "x": np.linspace(6, 32, 61).tolist(), "g": [1.0] * 61}],
                              "mock": True})
    print(f"wrote mock files to {MOCK_DIR}")


if __name__ == "__main__":
    main()
