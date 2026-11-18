# db/

`biofouling_synthetic_dataset.csv` is **synthetic**: 8,760 hourly rows (one year) from `synthetic_generator.py`, seed 42, with cleanings injected. It is not plant data.

Columns, briefly: time (`timestamp`, `time_hours`, `season`); measured signals (`T_cool_in_meas`, `T_cool_out_meas`, `mass_flow_cool_meas`, `delta_P_meas_kPa`, `U_overall_meas`); true simulator state, for evaluation only (`biofilm_thickness_um`, `Rf_fouling_resistance`, `delta_P_true_kPa`, `T_cool_in_true`, `fluid_velocity_m_s`, `wall_shear_Pa`); derived values (`heat_duty_MW`, `thermal_efficiency_ratio`, `bsi_score`); and the cleaning log (`cleaning_event`, `cleaning_type`).

Regenerate it from the repository root with `python db/generate.py`. Options: `--seed`, `--hours`, `--out`. The script runs offline, and the same seed gives a byte-identical file.
