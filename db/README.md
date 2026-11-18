# db/

All files are **synthetic** output of `synthetic_generator.py`: 8,760 hourly rows (one year) each, same columns. None of it is plant data.

| File | Seed | Cleaning | Scenario | Limits crossed (first hour) |
|---|---|---|---|---|
| `biofouling_synthetic_dataset.csv` | 42 | default: every 720 h, alternating mechanical/chemical (6 + 6, 90 kg biocide of the 250 kg permit) | nominal | none |
| `stress_no_clean.csv` | 43 | none | nominal | BSI 3137, Rf 3305, dP 4399 |
| `stress_long_interval.csv` | 44 | every 90 days, alternating | nominal | BSI 3533, Rf 3722 (dP peaks at 15.3 kPa, below 16) |
| `mismatch_stress.csv` | 45 | every 60 days, alternating | `k_dep_drift=0.5`, `scale_rate_m_per_h=5e-9`, bloom `(4300 h, 504 h, x3.0)` | BSI 3871, Rf 3964, dP 4985 |

`mismatch_stress.csv` is the held-out test set for the future UDE, because it contains mechanisms the nominal ODE lacks. Limits are those in `config.py` (Rf 3.0e-4 m²K/W and true dP 16 kPa; BSI 75, computed from measured dP).

Columns, briefly: time (`timestamp`, `time_hours`, `season`); measured signals (`T_cool_in_meas`, `T_cool_out_meas`, `mass_flow_cool_meas`, `delta_P_meas_kPa`, `U_overall_meas`); true simulator state, for evaluation only (`biofilm_thickness_um`, `Rf_fouling_resistance`, `delta_P_true_kPa`, `T_cool_in_true`, `fluid_velocity_m_s`, `wall_shear_Pa`); derived values (`heat_duty_MW`, `thermal_efficiency_ratio`, `bsi_score`); and the cleaning log (`cleaning_event`, `cleaning_type`).

Regenerate everything from the repository root with `python db/generate.py`. `--seed` sets the base seed (file seed = base + 0/1/2/3), and `--out-dir` writes elsewhere. The script runs offline, and the same seed gives byte-identical files.
