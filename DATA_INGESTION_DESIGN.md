# Biofouling AI — External Data Ingestion Architecture & Design Specification

**Application Target:** `workspace/biofouling_ai_repaired/`  
**Phase:** Benchmark 04 — Phase 2B  
**Document Version:** 1.0  
**Classification:** Academic Demonstration Prototype — Nuclear Secondary Cooling Auxiliary Systems  

---

## 1. Executive Overview

The Biofouling AI platform has been extended to support real-world industrial usability through **External Operating Data Ingestion & Condition-Based Cleaning Optimization**. Previously restricted to fixed synthetic annual simulation, the platform now provides a flexible three-mode data pipeline capable of ingesting live single-snapshot operator measurements and historical plant telemetry CSVs while maintaining strict physical validity, transparent data provenance, and robust fallback mechanisms when ML lag features cannot be constructed.

```
+----------------------------------------------------------------------------------------------------+
|                                      DATA INGESTION PIPELINE                                       |
+------------------------------------+----------------------------------+----------------------------+
|        MODE A: SIMULATION          |      MODE B: MANUAL SNAPSHOT     |   MODE C: HISTORICAL CSV   |
|   • Seed-locked 8,760h Dataset     |   • Single Operating State       |   • Multi-day Telemetry    |
|   • Benchmark Physical Baseline    |   • Tout > Tin Range Checks      |   • Delimiter Auto-Sniff   |
|   • Synthetic Noise & Kinetics     |   • Instantaneous Indicators     |   • Quality Scoring (0-100)|
|   • Complete 24h ML Lag Windows    |   • Biophysical ODE Projection   |   • Discontinuity Detection|
+------------------------------------+----------------------------------+----------------------------+
                                                  │
                                                  ▼
                         +--------------------------------------------------+
                         |          DATA PROVENANCE & VALIDATION            |
                         |   • Range Validation (Physical Bounds)           |
                         |   • Thermodynamic Gradient (Tout > Tin)          |
                         |   • Explicit Provenance Tagging                  |
                         +--------------------------------------------------+
                                                  │
                                                  ▼
                         +--------------------------------------------------+
                         |           DUAL FORECASTING ENGINE                |
                         |   • Primary: Biophysical ODE Kinetic State       |
                         |   • Auxiliary: Hybrid ML Ensemble (if N >= 25)   |
                         |     (Graceful fallback notice if N < 25 rows)    |
                         +--------------------------------------------------+
                                                  │
                                                  ▼
                         +--------------------------------------------------+
                         |      PHYSICS-BASED CLEANING OPTIMIZER            |
                         |   • Initialized at Active Operating Condition    |
                         |   • Evaluates Candidate Intervals (0d to 60d)    |
                         |   • Identifies Cost-Optimal Feasible Schedule    |
                         +--------------------------------------------------+
```

---

## 2. Multi-Mode Operational Architecture

The application operates in three distinct, mutually isolated modes switchable via REST API or interactive UI controls:

### Mode A: Simulation Mode (Default)
* **Description:** Continuous synthetic simulation spanning a full 8,760-hour (365-day) annual operating cycle.
* **Characteristics:** Seed-locked (`seed=42`) biophysical simulation coupling Ratkowsky growth kinetics, ambient seasonal temperature sinusoidal variation (12.0°C to 28.0°C), continuum Darcy-Weisbach hydraulics, and mechanical cleaning interventions.
* **Provenance Tag:** `synthetic_simulation`
* **Forecasting:** Both physics ODE projections and trained hybrid ensemble forecasts are fully available with complete 24-hour historical rolling and lag features.

### Mode B: Manual Operating Input Mode
* **Description:** Instantaneous operating snapshot entered directly by a plant operator or research engineer.
* **Required Inputs:** Intake temperature ($T_{in}$), discharge temperature ($T_{out}$), cooling water mass flow rate ($\dot{m}_{cw}$), and tube-bundle differential pressure drop ($\Delta P$).
* **Optional Inputs:** Cumulative operating hours ($t$), timestamp, measured heat duty ($Q$), and overall heat transfer coefficient ($U$).
* **Instantaneous Indicator Derivation:**
  $$\dot{Q} = \dot{m}_{cw} \cdot C_p \cdot (T_{out} - T_{in})$$
  $$U = \frac{\dot{Q}}{A \cdot \Delta T_{LMTD}}$$
  $$R_f = \max\left(0, \frac{1}{U} - \frac{1}{U_{clean}}\right)$$
  $$\Delta P_{ratio} = \frac{\Delta P}{\Delta P_{clean}}$$
  $$\text{BSI} = \min\left(100, \left(0.60 \cdot \frac{\Delta P_{ratio} - 1.0}{3.0} + 0.40 \cdot \frac{R_f}{R_{f,crit}}\right) \cdot 100\right)$$
  $$P_{pump,excess} = \frac{\dot{V} \cdot \Delta P_{excess}}{\eta_{pump}}$$
* **Transient Growth Rate Caveat:** A single point in time cannot mathematically establish the time derivative $dR_f/dt$. The engine issues an explicit engineering caveat and initializes the calibrated biophysical ODE at the validated instantaneous operating condition to project future progression.
* **ML Ensemble Status:** Because the hybrid ML ensemble relies on 24-hour rolling statistics and lagged features, it is automatically bypassed with an informative notice: *"Single manual snapshot provides 0 historical lag steps. ML feature extraction requires >= 25 chronological hours. Forecasting defaults strictly to calibrated physics ODE projection."*

### Mode C: Historical CSV Upload Mode
* **Description:** Multi-day or multi-month historical time-series datasets uploaded via drag-and-drop or file selector.
* **Capabilities:** Delimiter sniffing (comma, semicolon, tab), schema normalization, column alias auto-mapping, multi-factor data quality scoring, physical range filtering, and historical cleaning event detection.
* **Forecasting:** If dataset length $N \ge 25$ rows, the ML ensemble reconstructs rolling lag features; if $N < 25$, it reports an informative warning and defers to the biophysical ODE.

---

## 3. Supported CSV Schema & Column Mapping

The CSV ingestion parser supports canonical naming and standard industrial alias mappings:

| Canonical Field | Type | Physical Units | Nominal Value | Accepted Column Aliases |
| :--- | :---: | :---: | :---: | :--- |
| `timestamp` | String / ISO | DateTime | Current UTC | `time`, `date`, `datetime`, `date_time`, `record_time`, `utc_time` |
| `time_hours` | Integer / Float | hours | 0 to 200,000 | `hours`, `hour`, `operating_hours`, `op_hours`, `operating_time_h`, `sim_h` |
| `T_cool_in_meas` | Float | °C | 18.5 | `t_in`, `tin`, `t_cool_in`, `inlet_temp`, `inlet_temperature`, `water_inlet_c`, `temp_in` |
| `T_cool_out_meas`| Float | °C | 26.2 | `t_out`, `tout`, `t_cool_out`, `outlet_temp`, `outlet_temperature`, `water_outlet_c`, `temp_out` |
| `mass_flow_cool_meas`| Float | kg/s | 220.0 | `flow`, `mass_flow`, `flow_rate`, `cooling_flow`, `m_dot`, `mass_flow_kg_s`, `flow_kg_s` |
| `delta_P_meas_kPa`| Float | kPa | 7.91 | `delta_p`, `dp`, `dp_kpa`, `pressure_drop`, `diff_pressure_kpa`, `delta_p_kpa`, `dp_meas` |
| `heat_duty_MW` | Float (Optional) | MW | 4.70 | `heat_duty`, `duty_mw`, `q_mw`, `thermal_duty_mw`, `heat_load_mw` |
| `U_overall_meas` | Float (Optional) | W/(m²·K) | 2200.0 | `u_overall`, `u_meas`, `overall_htc`, `u_w_m2k`, `htc_meas` |
| `cleaning_event` | Integer (Optional)| 0 or 1 | 0 | `cleaning`, `cleaned`, `clean_flag`, `is_cleaned` |

---

## 4. Physical Validation & Constraint Enforcement

Industrial telemetry undergoes rigorous boundary verification:

1. **Intake Temperature Boundary ($T_{in}$):**  
   Valid domain: $[2.0^\circ\text{C}, 45.0^\circ\text{C}]$. Values below 2°C risk freeze-up; values above 45°C exceed secondary cooling design specifications.
2. **Discharge Temperature Boundary ($T_{out}$):**  
   Valid domain: $[5.0^\circ\text{C}, 65.0^\circ\text{C}]$.
3. **Thermodynamic Consistency ($\Delta T = T_{out} - T_{in} > 0$):**  
   Strict physical enforcement. In heat exchanger cooling service, the cooling water must absorb heat. If $T_{out} \le T_{in}$, the reading violates the Second Law of Thermodynamics and is rejected. Warnings are generated for $\Delta T < 1.0^\circ\text{C}$ (minimal duty / sensor resolution limit) and $\Delta T > 25.0^\circ\text{C}$ (excessive thermal gradient).
4. **Mass Flow Rate Boundary ($\dot{m}_{cw}$):**  
   Valid domain: $[20.0\text{ kg/s}, 500.0\text{ kg/s}]$. Negative or zero flow is physically impossible in an operating loop.
5. **Differential Pressure Boundary ($\Delta P$):**  
   Valid domain: $[1.0\text{ kPa}, 100.0\text{ kPa}]$. Values below clean baseline ($7.91\text{ kPa}$) indicate sensor calibration drift or reduced flow; values above $100\text{ kPa}$ indicate catastrophic tube blockage.

---

## 5. Multi-Factor Data Quality Scoring Engine

The Data Quality Score ($S_Q \in [0, 100\%]$) is evaluated across three weighted dimensions:

$$S_Q = 40.0 \cdot R_{completeness} + 45.0 \cdot R_{validity} + 15.0 \cdot R_{temporal}$$

* **Completeness ($R_{completeness}$, 40% weight):**
  $$R_{completeness} = \max\left(0, 1 - \frac{N_{missing\_cells}}{N_{total\_rows} \times N_{required\_columns}}\right)$$
  Missing values are tallied, reported, and **never silently filled or fabricated**.
* **Physical Validity ($R_{validity}$, 45% weight):**
  $$R_{validity} = \frac{N_{valid\_rows}}{N_{total\_rows}}$$
  Rows violating physical ranges or thermodynamic gradients ($T_{out} \le T_{in}$) are flagged. If all rows are invalid ($N_{valid\_rows} = 0$), $S_Q$ is forced to 0.0%.
* **Temporal Regularity ($R_{temporal}$, 15% weight):**
  $$R_{temporal} = \max\left(0, 1 - \frac{N_{duplicates} + N_{time\_inversions}}{N_{total\_rows}}\right)$$
  Monitors timestamp monotonicity and duplicate records.

### Quality Rating Tiers
* **EXCELLENT ($S_Q \ge 90.0\%$):** High-integrity telemetry; fully accepted for diagnostics, ML inference, and optimization.
* **GOOD ($75.0\% \le S_Q < 90.0\%$):** Minor non-critical anomalies; accepted with engineering notices.
* **DEGRADED ($50.0\% \le S_Q < 75.0\%$):** Significant missing observations or noise; accepted for basic indicator derivation, but ML confidence bounds are widened.
* **REJECTED ($S_Q < 50.0\%$ or $N_{valid\_rows} < 2$):** Critical corruption or physically invalid readings. Ingestion is halted, no files are imported, and clear diagnostics are returned to the user.

---

## 6. Historical Telemetry Diagnostics & Cleaning Detection

The `historical_analyzer` module scans imported time series to detect:
1. **Fouling Progression Rates:**
   Linear regression over time gives the empirical fouling growth rate $\frac{dR_f}{dt}$ (in $\text{m}^2\text{K/W per day}$) and hydraulic degradation $\frac{d\Delta P}{dt}$ (in $\text{kPa per day}$).
2. **Historical Cleaning Events:**
   Identifies sharp downward discontinuities where:
   $$\Delta R_f < -2.0 \times 10^{-5}\text{ m}^2\text{K/W} \quad \text{or} \quad \Delta(\Delta P) < -1.2\text{ kPa}$$
   Calculates cleaning recovery efficiency:
   $$\eta_{recovery} = \frac{R_{f,before} - R_{f,after}}{R_{f,before}} \times 100\%$$
   Categorizes intervention into mechanical cleaning ($\eta < 92\%$) vs chemical biocide shock treatment ($\eta \ge 92\%$).
3. **Operational Threshold Breaches:**
   Extrapolates linear trajectory to project days remaining until critical limits ($R_{f,crit} = 0.00030\text{ m}^2\text{K/W}$ or $\Delta P_{crit} = 16.0\text{ kPa}$).

---

## 7. Data Provenance & Lineage Tracking

Every KPI, telemetry value, and optimizer output is explicitly tagged with its operational provenance:

| Provenance Tag | Definition | Examples |
| :--- | :--- | :--- |
| `direct_user_measurement` | Raw sensor value supplied directly by the user | $T_{cool,in}, T_{cool,out}, \dot{m}_{cw}, \Delta P$ |
| `derived_calculation` | Instantaneous deterministic physical calculation | $R_f, U, \eta_{thermal}, \text{BSI}, P_{pump,excess}$ |
| `physics_based_prediction`| Forward projection from biophysical kinetic ODE | Projected $R_f(t)$, recommended interval, cost curves |
| `ml_prediction` | Data-driven residual ensemble inference | ML forecast trajectory, split-conformal intervals |
| `synthetic_simulation` | Generated by seed-locked annual benchmark simulation | Default 8,760h baseline telemetry |

---

## 8. Security, Data Privacy & Safety Notice

1. **Zero External Transmission:** All user data, manual snapshots, and CSV datasets are processed strictly in-memory or persisted locally in `workspace/biofouling_ai_repaired/data/user_uploads/`. No external network requests or third-party cloud APIs are contacted.
2. **Academic Demonstration Prototype Disclaimer:**  
   > *This application is an engineering demonstration prototype developed for academic benchmarking of nuclear auxiliary heat exchanger biofouling control. It is not licensed for direct connection to nuclear reactor safety protection systems, nor does it possess plant control authority. All operational cleaning decisions must be reviewed by qualified station system engineers.*
