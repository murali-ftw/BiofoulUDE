# Benchmark 04: Industrial Black UI & Predictive Cleaning Interval Optimizer Upgrade Report
## AI-Assisted Optimisation of Heat-Exchanger Cleaning for Biofouling Control in Nuclear Cooling Systems

**Date:** November 14, 2026  
**System:** MetaGPT Multi-Agent Engineering Framework (Benchmark 04 Phase 4 Upgrade)  
**Target Workspace:** `workspace/biofouling_ai_repaired/`  
**Test Suite Status:**  
* **Independent Acceptance Evaluator (Frozen 25-Point Suite):** **25 / 25 PASSED (100.00% PASS RATE)**  
* **Dedicated Application Unit Tests:** **20 / 20 PASSED (100.00% PASS RATE)** (11 base + 9 optimizer tests)  
* **Evaluator Bitwise Hash Integrity:** **100% UNCHANGED** (Bitwise identical to `EVALUATOR_INITIAL_HASHES.txt`)  
* **Zero External Dependencies:** Strict standard library + pre-installed scientific core (`numpy`, `pandas`, `scipy`, `scikit-learn`)  

---

## 1. Executive Summary & Upgrade Objectives

This upgrade transitions the Biofouling AI application from an initial demonstrator into a **professional industrial engineering monitoring dashboard with an autonomous, scenario-based predictive cleaning interval optimizer**.

### Core Engineering Question Addressed:
> **"Based on the current simulated operating condition and predicted fouling progression, when should the heat exchanger next be cleaned, and why is that interval economically and operationally preferable?"**

The system answers this quantitatively by:
1. Evaluating candidate maintenance intervals ($0\text{d}$ [Immediate], $7\text{d}$, $14\text{d}$, $21\text{d}$, $30\text{d}$, $45\text{d}$, $60\text{d}$) over a configurable planning horizon.
2. Modeling forward physical fouling progression, degraded thermal conductance, dynamic hydraulic friction factor via Darcy-Weisbach with Haaland approximation, and cleaning reset recoveries ($90\text{--}95\%$).
3. Calculating discrete cleaning and downtime costs against cumulative operational penalties (excess pumping electricity + thermal replacement deficit).
4. Enforcing strict thermal and hydraulic feasibility constraints ($R_f \le 3.0 \times 10^{-4}\text{ m}^2\cdot\text{K/W}$, $\Delta P \le 75.0\text{ kPa}$, $\text{BSI} \le 75.0$).
5. Selecting the lowest-cost feasible candidate and providing transparent engineering rationale, breach warnings, and statistical uncertainty bands.
6. Overhauling the user interface into a restrained **Industrial Black UI** matching modern nuclear and process plant control consoles.

---

## 2. Files Inspected & Files Changed

### Files Inspected:
* `workspace/biofouling_ai_repaired/config.py`
* `workspace/biofouling_ai_repaired/cleaning_optimizer.py`
* `workspace/biofouling_ai_repaired/improved_model.py`
* `workspace/biofouling_ai_repaired/baseline_model.py`
* `workspace/biofouling_ai_repaired/fouling_indicators.py`
* `workspace/biofouling_ai_repaired/sustainability_metrics.py`
* `workspace/biofouling_ai_repaired/synthetic_generator.py`
* `workspace/biofouling_ai_repaired/dashboard_server.py`
* `workspace/biofouling_ai_repaired/dashboard/index.html`
* `workspace/biofouling_ai_repaired/dashboard/style.css`
* `workspace/biofouling_ai_repaired/dashboard/app.js`
* `workspace/biofouling_ai_repaired/tests/test_application.py`
* `workspace/benchmark_evaluators/biofouling_ai/test_acceptance_suite.py`

### Files Modified & Added:
1. `workspace/biofouling_ai_repaired/config.py`: Added operational thresholds (`CRITICAL_RF_M2K_W`, `CRITICAL_DELTA_P_KPA`, `CRITICAL_BSI_SCORE`) and downtime constants (`DOWNTIME_HOURS_PER_CLEANING`, `DOWNTIME_COST_PER_HOUR`).
2. `workspace/biofouling_ai_repaired/improved_model.py`: Added empirical residual uncertainty standard deviation (`residual_std_`) and statistical prediction intervals via `predict_with_uncertainty(confidence=0.95)`.
3. `workspace/biofouling_ai_repaired/cleaning_optimizer.py`: Implemented `CleaningIntervalOptimizer`, featuring `find_uncleaned_threshold_crossing`, `evaluate_scenario`, and `optimize_interval`.
4. `workspace/biofouling_ai_repaired/dashboard_server.py`: Enhanced GET `/api/recommendation`, `/api/forecast`, `/api/kpis` and POST `/api/optimize` endpoints.
5. `workspace/biofouling_ai_repaired/dashboard/index.html`: Completely redesigned into a 5-section industrial layout with semantic KPI readouts, hero recommendation box, strategy table, and parameter controls.
6. `workspace/biofouling_ai_repaired/dashboard/style.css`: Replaced blue-gray UI with the industrial black design system (`#080A0D` background, `#11151B` panels, `#252B34` borders, `#00C8D7` accent).
7. `workspace/biofouling_ai_repaired/dashboard/app.js`: Upgraded to render multi-series native SVG charts with uncertainty envelopes, current simulation markers, cleaning indicators, and live scenario matrix binding.
8. `workspace/biofouling_ai_repaired/main.py`: Integrated Section 4.B into default CLI execution.
9. `workspace/biofouling_ai_repaired/tests/test_cleaning_interval_optimizer.py`: Added 9 dedicated unit tests.

### Preserved Without Modification:
* All original Benchmark 04 execution data in `workspace/biofouling_ai_benchmark/`.
* The frozen independent acceptance evaluator suite in `workspace/benchmark_evaluators/biofouling_ai/`.
* All existing physical equations and calculations (Darcy-Weisbach, Haaland friction, Ratkowsky growth, Kern-Seaton removal).
* 100% backward compatibility for existing 11 application tests and 25 evaluator tests.

---

## 3. Industrial Black UI Specification

The visual interface has been completely transformed into a professional industrial monitoring system:

| UI Token | Hex Code | Purpose |
| :--- | :---: | :--- |
| **Primary Background** | `#080A0D` | Deep black background, minimizing operator eye fatigue |
| **Panel Surface** | `#11151B` | Secondary dark surface for KPI and chart cards |
| **Panel Header / Sub** | `#161B22` / `#0D1117` | Structural grouping and table headers |
| **Border Normal** | `#252B34` | Crisp, low-contrast boundary lines |
| **Border Active / Light**| `#323A46` | Active controls and hover states |
| **Primary Text** | `#F5F5F5` | High-contrast legible body text and KPI values |
| **Secondary Text** | `#929AA5` | Engineering units, field labels, axis annotations |
| **Muted Text** | `#5B6574` | Footers, disabled parameters |
| **Primary Accent** | `#00C8D7` | Cyan accent for active telemetry and key metrics |
| **Status Positive** | `#22C55E` | Muted green indicator for nominal compliance |
| **Status Warning** | `#F59E0B` | Amber warning for threshold proximity |
| **Status Critical** | `#EF4444` | Red alert for safety threshold violations |

Typography utilizes clean sans-serif typography for structural labeling and monospace (`JetBrains Mono`, `Consolas`) for numerical values, units, and timestamps.

---

## 4. Information Architecture

The dashboard is structured into five distinct operational sections:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HEADER: Nuclear Auxiliary Heat-Exchanger Biofouling Monitor (AI CBO Engine) │
│ STATUS: SYSTEM ONLINE (Zero-Install Stdlib Engine)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ DISCLAIMER: SYNTHETIC SIMULATION DATA — ACADEMIC DEMONSTRATION ONLY         │
├─────────────────────────────────────────────────────────────────────────────┤
│ SECTION A: Operating Condition (8 Real-Time KPI Cards)                      │
│ [ Rf ] [ U/U₀ ] [ ΔP ] [ Flow Rate ] [ Temps ] [ BSI ] [ Pump kW ] [ Heat ] │
├─────────────────────────────────────────────────────────────────────────────┤
│ SECTION C: Predictive Cleaning Recommendation (Hero Decision Box)           │
│ • Recommended Next Interval: 14 Days (336 Operating Hours)                  │
│ • Recommended Window: Day 12 to Day 17                                      │
│ • Status-Quo Uncleaned Breach: Day 32.8 (Hour 3,786) — Rf Limit             │
│ • Primary Engineering Rationale & Trade-off Justification                   │
│ • Expected Operating Condition if Maintenance is Delayed                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ SECTION B: Historical & Predicted Fouling Trajectory (Vector SVG Chart)     │
│ • Historical Rf & ΔP (Solid lines)                                          │
│ • Forecast Rf & ΔP (Dashed lines with 95% Confidence Band Polygon)          │
│ • Critical Safety Thresholds (Red dashed horizontal lines)                  │
│ • Current Hour (t₀) & Recommended Cleaning Event Markers                    │
│ • Configurable Horizon Switcher (7d, 14d, 30d, 60d)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ SECTION D: Strategy Comparison (Scenario Trade-Off Matrix)                  │
│ [ Immediate ] [ 7-Day ] [ 14-Day ] [ 21-Day ] [ 30-Day ] [ 45-Day ] [ 60-Day│
│ [ Fixed 30-Day Calendar ] [ AI Condition-Based Strategy ]                   │
│ Columns: Feasibility, Cleanings, Downtime, Pumping, Thermal, Total, Delta  │
├─────────────────────────────────────────────────────────────────────────────┤
│ SECTION E: Sustainability & Parameter Tuning Form                           │
│ • Excess Energy (kWh), Grid CO₂, Biocide Permit Tracking                    │
│ • Dynamic Tuning: Cleaning Cost ($), Power ($/kWh), Thermal Loss ($/MWh)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Mathematical Formulation of the Interval Optimizer

### 5.1 Objective Function
For a candidate interval $D$ (in days, $T_{interval} = 24 \cdot D$ hours) over planning horizon $H$ (typically 60 days / 1,440 hours), the optimizer minimizes the total projected cost:
$$\min_{D \in \mathcal{F}} C_{total}(D) = C_{cleaning}(D) + C_{downtime}(D) + C_{pumping}(D) + C_{thermal}(D)$$
Subject to feasibility set $\mathcal{F}$:
$$\mathcal{F} = \left\{ D \in \mathcal{D} \;\middle|\; \max_{t \in [0, H]} R_f(t; D) \le R_{f,crit} \;\land\; \max_{t \in [0, H]} \Delta P(t; D) \le \Delta P_{crit} \;\land\; \max_{t \in [0, H]} \text{BSI}(t; D) \le \text{BSI}_{crit} \right\}$$

Where:
1. **Cleaning Cost:**
   $$C_{cleaning}(D) = N_{clean}(D) \cdot C_{event}$$
   ($C_{event} = \$3,500$ nominal).
2. **Downtime Opportunity Cost:**
   $$C_{downtime}(D) = N_{clean}(D) \cdot t_{downtime} \cdot c_{downtime\_rate}$$
   ($t_{downtime} = 4.0\text{ h}$, $c_{downtime\_rate} = \$250.00/\text{h}$).
3. **Parasitic Pumping Energy Cost:**
   $$C_{pumping}(D) = \left( \sum_{h=1}^H P_{excess}(h; D) \cdot 1\text{ h} \right) \cdot c_{elec}$$
   Where $P_{excess}(h) = \frac{(\dot{m} / \rho) \cdot \max(0, \Delta P(h) - \Delta P_{clean})}{\eta_{pump}}$ ($c_{elec} = \$0.12/\text{kWh}$, $\eta_{pump} = 0.75$).
4. **Thermal Deficit Cost:**
   $$C_{thermal}(D) = \left( \sum_{h=1}^H \max\left(0, 1 - \frac{U(h; D)}{U_{clean}}\right) \cdot Q_{nominal} \cdot 1\text{ h} \right) \cdot c_{thermal}$$
   ($c_{thermal} = \$35.00/\text{MWh}$, $Q_{nominal} = 3.5\text{ MW}$).

### 5.2 Dynamic Forward Projection
At each simulation step $h$:
* **Intake Temperature:** $T(h) = T_{mean} + T_{amp} \sin(\omega h - \phi) + T_{diurnal} \sin(\omega_d h)$.
* **Biological Growth:** $\mu_{bio}(h) = \text{Ratkowsky}(T(h))$.
* **Haaland Turbulent Friction:** $\frac{1}{\sqrt{f}} = -1.8 \log_{10}\left[ \left(\frac{(\epsilon_{clean} + \delta_f)/D_{eff}}{3.7}\right)^{1.11} + \frac{6.9}{Re} \right]$.
* **Wall Shear Detachment:** $\tau_w = \frac{1}{8} f \rho u^2$.
* **Differential Biofilm Evolution:** $\delta_f(h+1) = \delta_f(h) + (\dot{m}_{dep} - \dot{m}_{rem}) \cdot \Delta t$.
* **Cleaning Event:** If scheduled at hour $h$, $\delta_f \leftarrow (1 - \eta_{clean}) \delta_f$, resetting $90\text{--}95\%$ of deposit thickness.

### 5.3 Infeasible State Handling
If the current operating state or near-term trajectory breaches constraints prior to the earliest maintenance window (e.g., initial $R_f \ge 0.00030$), the optimizer reports:
* `status: "CRITICAL_INFEASIBLE_EMERGENCY"`
* `is_feasible: False`
* `recommended_interval_days: 0`
* `recommended_window_days: "Immediate (0 - 24 hours)"`
* `primary_reason`: Explicit warning detailing the breached parameter, current value, limit, and days until unmitigated derate.

---

## 6. Verification & Test Results

### 6.1 Application Test Results (20 / 20 Passed)
```powershell
.venv\Scripts\python.exe -m unittest discover -s workspace/biofouling_ai_repaired/tests -p "test_*.py"
```
```
.......... [test_application.py: 11 tests passed]
.........  [test_cleaning_interval_optimizer.py: 9 tests passed]
----------------------------------------------------------------------
Ran 20 tests in 2.853s

OK
```

#### Detailed Breakdown of New Optimizer Unit Tests:
1. `test_01_candidate_scenario_evaluation`: Validates evaluation of all 7 candidate intervals and exact summation of cleaning, downtime, pumping, and thermal penalty costs.
2. `test_02_threshold_crossing_detection`: Validates detection of status-quo breach hour, relative days, and limiting parameter.
3. `test_03_lowest_cost_feasible_selection`: Confirms optimizer chooses lowest-cost feasible candidate over simply minimizing cleaning cost.
4. `test_04_infeasible_condition_handling`: Verifies emergency reporting when starting fouling condition exceeds critical limits.
5. `test_05_cost_sensitivity`: Proves that increasing cleaning cost shifts recommendation to longer intervals (e.g., $14\text{d} \to 21\text{d}$), while increasing power cost shifts recommendation earlier.
6. `test_06_cleaning_recovery_reset`: Proves mechanical cleaning resets $90\%$ and biocide shock resets $95\%$ of biofilm thickness.
7. `test_07_deterministic_output`: Confirms identical inputs yield bitwise identical intervals and cost calculations.
8. `test_08_uncertainty_bounds`: Proves that empirical residual standard deviation produces non-negative upper and lower confidence intervals.
9. `test_09_dashboard_api_endpoints`: Verifies HTTP 200 responses on `/api/recommendation`, `/api/forecast`, and POST `/api/optimize`.

### 6.2 Frozen Independent 25-Point Acceptance Evaluator Results (25 / 25 Passed)
```powershell
.venv\Scripts\python.exe workspace/benchmark_evaluators/biofouling_ai/run_evaluation.py --target workspace/biofouling_ai_repaired
```
```
================================================================================
BENCHMARK 04: INDEPENDENT 25-POINT ACCEPTANCE EVALUATOR
Target Directory: workspace/biofouling_ai_repaired
Total Tests:      25
================================================================================
[✓ PASS ] test_01_module_imports                        [Software Acceptance]
[✓ PASS ] test_02_configuration_constants               [Software Acceptance]
[✓ PASS ] test_03_zero_external_installation_compliance [Software Acceptance]
[✓ PASS ] test_04_file_structure_integrity              [Software Acceptance]
[✓ PASS ] test_05_synthetic_data_dimensions_and_schema  [Software Acceptance]
[✓ PASS ] test_06_seasonal_thermal_cycle_realism        [Model Quality]
[✓ PASS ] test_07_ratkowsky_growth_temperature_response [Model Quality]
[✓ PASS ] test_08_biofilm_accumulation_and_lag_phase    [Model Quality]
[✓ PASS ] test_09_hydraulic_pressure_scaling_consistency [Model Quality]
[✓ PASS ] test_10_thermal_resistance_and_efficiency_calc [Model Quality]
[✓ PASS ] test_11_biofouling_severity_index_bounds      [Model Quality]
[✓ PASS ] test_12_sensor_noise_distribution_realism     [Model Quality]
[✓ PASS ] test_13_chronological_split_integrity         [Model Quality]
[✓ PASS ] test_14_data_leakage_scaler_isolation         [Model Quality]
[✓ PASS ] test_15_baseline_model_accuracy               [Model Quality]
[✓ PASS ] test_16_improved_ml_model_superiority_and_horiz [Model Quality]
[✓ PASS ] test_17_cleaning_event_recovery_behavior      [Model Quality]
[✓ PASS ] test_18_condition_based_vs_preventive_comparis [Model Quality]
[✓ PASS ] test_19_cleaning_cost_tradeoff_sensitivity    [Model Quality]
[✓ PASS ] test_20_parasitic_pumping_power_calculation   [Model Quality]
[✓ PASS ] test_21_carbon_emission_accounting_accuracy   [Model Quality]
[✓ PASS ] test_22_biocide_discharge_compliance_tracking [Model Quality]
[✓ PASS ] test_23_dashboard_http_server_endpoints       [Software Acceptance]
[✓ PASS ] test_24_dashboard_svg_chart_rendering         [Software Acceptance]
[✓ PASS ] test_25_reproducibility_seed_locking          [Software Acceptance]
================================================================================
OVERALL EVALUATION SUMMARY:
  Total Tests:             25
  Passed:                  25 / 25
  Failed:                  0
  Errors:                  0
  Overall Pass Rate:       100.00%
  Software Acceptance:     8/8 (100.0%)
  Model Quality:           17/17 (100.0%)
  Elapsed Time:            4.09 seconds
================================================================================
```

### 6.3 Frozen Evaluator Hash Verification
All files in `workspace/benchmark_evaluators/biofouling_ai/` remain bitwise identical:
* `evaluator_adapter.py`: `A7320C8889E263156E98F1D5400B0F1BAAEB5CE3E0831CE9D5A08C168F9E6BDD`
* `EVALUATOR_DESIGN_REPORT.md`: `629E99CAFFDBB77E82374CC75848F15652877AE7CB0AD1423ED09C756BC0F93E`
* `run_evaluation.py`: `E9D32AA815DB32C7BFDAE7313D0315CB2559437CFB101ABB1ECF91DF52B6942C`
* `test_acceptance_suite.py`: `EE1DB3568642F4F55421D44C0BEB3FBEDCBE14911F284997C1029042B21659A3`

---

## 7. Launch Instructions

### 7.1 Start Industrial Black UI Dashboard
From PowerShell inside repository root:
```powershell
.venv\Scripts\python.exe workspace/biofouling_ai_repaired/main.py --start-dashboard --port 8050
```
Open a browser and navigate to:
```
http://localhost:8050/
```

### 7.2 Run Unified CLI Simulation & Optimizer Pipeline
```powershell
.venv\Scripts\python.exe workspace/biofouling_ai_repaired/main.py
```

### 7.3 Run All 20 Application Unit Tests
```powershell
.venv\Scripts\python.exe -m unittest discover -s workspace/biofouling_ai_repaired/tests -p "test_*.py"
```

### 7.4 Execute Frozen Independent Acceptance Evaluator
```powershell
.venv\Scripts\python.exe workspace/benchmark_evaluators/biofouling_ai/run_evaluation.py --target workspace/biofouling_ai_repaired
```

---

## 8. Scientific Limitations & Safety Disclaimers

> [!CAUTION]
> **CRITICAL NUCLEAR SAFETY DISCLAIMER:**  
> This application is an **academic engineering demonstration and benchmark evaluation prototype**. It is **NOT** validated against operational nuclear power station telemetry and **MUST NEVER** be connected to or utilized for real-time safety, protection, or operational control systems of any nuclear reactor or safety-related cooling circuit.

1. **Synthetic Data Origin:** All time-series profiles are physically modeled synthetic datasets incorporating empirical kinetic relationships from literature (Kern & Seaton, Ratkowsky, Haaland, Darcy-Weisbach).
2. **Simplified Microbial Assumptions:** The model treats the biofilm as a homogeneous biological layer with constant thermal conductivity ($k_f = 0.60\text{ W/m}\cdot\text{K}$). Real-world biofouling features multi-species bacterial consortia, inorganic particulate entrapment, extracellular polymeric substance (EPS) viscoelastic deformation, and spatial non-uniformities along tube length.
3. **Formal Nuclear Regulatory Qualification:** Any operational deployment within nuclear plant cooling loops requires formal software verification and validation (V&V) under ASME NQA-1, IEEE 7-4.3.2, and 10 CFR 50 Appendix B standards.

---

## 9. Phase 1 Engineering Upgrades & ML Dashboard Integration (2026-11-14)

### 9.1 Calibrated Darcy-Weisbach Baseline Pressure Drop
- **Initial State:** Hardcoded 45.0 kPa baseline suppressed all parasitic pumping energy calculations to 0.0 kWh because fouled pressure drop ranged from 9.1 to 12.3 kPa.
- **Calibrated State:** Continuum Darcy-Weisbach + Haaland calculation at nominal flow (220 kg/s) across 450 tubes ($D_i = 21\text{ mm}$, $L = 6.0\text{ m}$) yields **7.91 kPa**.
- **Impact:** Excess pumping energy increased to **3,100.74 kWh/year**, unlocking **1,395.33 kg CO₂** in authentic auxiliary carbon emissions and $372.09 in realistic pumping power penalties.

### 9.2 Disambiguation of Maintenance vs Thermal Deficit Economics
- **Mechanical Cleaning Intervention Cost:** **$3,500.00 USD/event** (contractor mobilization, elastomeric sponge balls, strainer inspection).
- **Chemical Biocide Shock Treatment:** **$6,500.00 USD/event** (sodium hypochlorite dosing, neutralization, NPDES sampling).
- **Thermal Deficit Replacement Penalty:** **$35.00 USD/MWh** (cost per megawatt-hour of cooling deficit requiring auxiliary chiller operation).
- **Outage Downtime Cost:** **$250.00 USD/hour** ($1,000.00 per 4-hour intervention).
- **Electricity Cost:** **$0.12 USD/kWh**.

### 9.3 End-to-End ML Model Serving & Visualization
- **Serialized Model:** Physics-informed hybrid ensemble persisted to `models/improved_ensemble.pkl` (235 KB).
- **Dynamic Loading:** `dashboard_server.py` loads the pickled model at startup via `get_or_load_model()`.
- **API Endpoints:**
  - `POST /api/predict`: Real-time inference returning point prediction, 95% confidence intervals, and BSI score.
  - `GET /api/forecast`: Dual-curve forecast returning both deterministic ODE physics projection and ML ensemble inference.
- **Vector SVG UI:** Dual-line rendering in browser with distinct purple dashed series (`#A855F7`) for ML predictions alongside cyan physics curves.

### 9.4 Feature Ablation Findings
- **Target Leakage ($U_{meas}$):** Because $R_f \approx 1/U_{meas} - 1/U_0$, including $U_{meas}$ with thermocouple noise caused tree models to overfit to sensor noise rather than biological kinetics.
- **Ablated Performance:** Removing $U_{meas}$ improved model $R^2$ to **0.9472** and reduced RMSE to **$5.86 \times 10^{-6}\text{ m}^2\cdot\text{K/W}$** (**22.02% improvement** over baseline) by forcing the model to rely on genuine hydraulic constriction dynamics ($\Delta P$, $\Delta P_{roll24}$, $\Delta P_{diff24}$, $\dot{m}$).

### 9.5 Current Test & Verification Status
- **Application Unit Tests:** 31/31 passing (100.0%) across `test_application.py`, `test_cleaning_interval_optimizer.py`, and `test_phase1_engineering.py`.
- **Independent Acceptance Evaluator:** 25/25 passing (100.0%).
- **Evaluator Checksum Integrity:** All evaluator files strictly bitwise verified against `EVALUATOR_INITIAL_HASHES.txt`.
