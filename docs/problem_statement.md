# Problem Statement

**Project:** AI-Assisted Optimisation of Heat-Exchanger Cleaning for Biofouling Control in Nuclear Cooling Systems
**Type:** University EVS case study, academic demonstration
**Data:** synthetic only (see [Scope and limitations](#7-objectives-scope-assumptions-and-limitations))

---

## 1. Background

A nuclear power plant has to reject heat all the time. That includes the condenser while it is
generating power, and auxiliary systems such as component cooling water and essential service water,
which remove heat from pumps, bearings, and safety-related equipment, including decay heat after
shutdown. Most of this heat ends up in a natural water body (river, lake, estuary or sea) or a
cooling tower.

The device that moves the heat is a **heat exchanger**. It transfers heat from a hot fluid to a
cold fluid through a solid wall, without the two fluids mixing. In the shell-and-tube design
modelled here, raw cooling water flows **inside the tubes** and the process fluid flows around them
in the shell.

The rate of heat transfer is

```
Q = U * A * dT_lm
```

where `U` is the overall heat-transfer coefficient (W/m²K), `A` is the tube surface area (m²) and
`dT_lm` is the log-mean temperature difference (K). Anything that lowers `U` lowers the heat the
exchanger can remove at the same temperatures.

## 2. The problem: biofouling

Cooling water carries microorganisms, nutrients and silt. Bacteria attach to the tube wall and build
a **biofilm**: cells held together by extracellular polymeric substances (EPS). In practice the
deposit is often mixed with corrosion products and mineral scale. This is **biofouling**
(microfouling; macrofouling by mussels or barnacles is out of scope here).

Biofouling has four effects that matter for operation:

1. **Thermal resistance.** Biofilm conducts heat poorly (about 0.6 W/mK, close to water), so even
   a thin layer adds a resistance `Rf` in series with the clean wall and lowers `U`.
2. **Flow-area loss.** The film narrows the effective tube diameter and roughens the wall.
3. **Higher pressure drop and pumping energy.** A narrower, rougher tube raises the friction
   pressure drop, so the pumps use more electricity to move the same flow.
4. **Reduced heat-duty margin.** With a lower `U`, the exchanger may no longer meet the required
   heat duty at the design temperatures, especially in summer when the water is warm and growth is fastest.

How much this matters for safety is **plant-specific**. It depends on the system, its margins, its
redundancy, and the plant's technical specifications. This project does not assess any plant's
safety case and makes no claims about one.

## 3. Evidence from a nuclear facility

Rao et al. (2009) investigated a thermo-fluid heat exchanger at a nuclear test reactor at
Kalpakkam, India:

- over **90% of the inspected tube bundle was clogged**;
- deposits contained **iron corrosion products, calcium and magnesium carbonates, iron sulphide and
  magnetite**, so the fouling was biological, corrosion and scale combined;
- **iron-oxidising bacteria** and **sulphate-reducing bacteria** were present, at about
  **1e5 to 1e6 CFU/g** of deposit;
- the operating temperature was **45 to 50 °C**.

This is **one facility** with its own water chemistry, materials and operating history. It shows
that severe fouling does occur in nuclear cooling service. It must **not** be generalised to
other plants, and none of its numbers are used as parameters in this project.

## 4. The key relation

Every part of this project uses the same two equations, which link the biofilm to what the
instruments measure:

```
1/U = 1/U_clean + Rf          (thermal resistances in series)
Rf  = delta / k_biofilm       (conduction through a film of thickness delta)
```

- `U_clean`: overall coefficient of the clean exchanger (2200 W/m²K in `core/config.py`)
- `Rf`: fouling resistance (m²K/W)
- `delta`: biofilm thickness (m)
- `k_biofilm`: biofilm thermal conductivity (0.60 W/mK in `core/config.py`)

**Consequence:** if `U` is measured, `Rf` follows from it algebraically. This matters for how
models are evaluated (Section 6).

## 5. Existing control methods and scheduling strategies

### Control methods

| Method | How it works | Main constraints |
|---|---|---|
| Mechanical cleaning (brushes, pigs, hydro-lancing) | Physically removes deposits, usually offline | Needs the exchanger isolated (downtime, redundancy needed); labour; can damage tubes |
| Chemical cleaning (acid, alkaline, chelating agents) | Dissolves scale and corrosion products | Downtime; waste handling; compatibility with tube materials |
| Biocide / water treatment (e.g. chlorination) | Kills or suppresses organisms in the intake water | Environmental discharge permits limit the dose; organisms may become tolerant; does not remove existing scale |
| Online cleaning (e.g. sponge-ball systems) | Circulates cleaning balls through tubes during operation | Capital cost; ball loss and recovery; less effective on hard or mixed deposits |

### Scheduling strategies

| Strategy | Rule | Limitation |
|---|---|---|
| Fixed interval | Clean every N days | Cleans too early in cold months and too late in warm months |
| Condition-based | Clean when a measured indicator (dP, U) crosses a threshold | Reacts to the current state only; thresholds are usually set by experience, not by cost |
| Prediction-assisted | Forecast the indicator and clean before it crosses the limit | Only as good as the forecast; often not linked to cost |
| Optimised | Choose timing (and method) to minimise total cost subject to limits | Needs a trustworthy model of future fouling *and* a cost model |

## 6. The research gap

A more accurate predictor is not enough on its own. What is missing is a **validated cleaning
decision**: when to clean, chosen against real costs and operating limits, with an honest
statement of uncertainty.

Common data-driven approaches have specific weaknesses for this problem:

1. **Not enough labelled cycles.** A year of hourly data contains only about 12 fouling and
   cleaning cycles (exactly 12 in this project's synthetic dataset). A purely data-driven model
   (LSTM, large tree ensembles) has very few independent examples of how fouling builds up from a
   clean tube.
2. **Physics can be violated.** A black-box model can predict a negative thickness, fouling that
   shrinks without cleaning or shear, or behaviour that changes if units are rescaled.
3. **Target leakage through related inputs.** Because `U` determines `Rf` (Section 4), a model
   given same-time `U` as an input and asked to "predict" `Rf` is mostly doing algebra. It can
   score a high R² without being able to forecast anything. The existing Ridge and Ridge+GBM models
   in this repository have exactly this problem (see [model.md](model.md), Section 9).
4. **Predictor and decision are disconnected.** A forecast that the optimiser never uses does not
   improve the decision. In this repository's current optimiser, the ML forecast is not used.

## 7. Objectives, scope, assumptions and limitations

### Objectives

1. Build a **hybrid physics + ML model** (a Universal Differential Equation) of biofilm growth that
   needs few cycles to train and cannot violate basic physics.
2. Train it on **measured quantities only** (U, dP), so that `Rf` is never used as a shortcut.
3. Produce **forecasts with uncertainty** of fouling state and time-to-limit.
4. Turn the forecast into a **cost-optimal cleaning interval** subject to thermal and hydraulic limits.
5. Compare against simple baselines honestly, including on data the model was not designed for.

### Scope

- One shell-and-tube exchanger, tube-side microfouling only.
- Hourly time step, one-year horizon.
- Decision variable: the cleaning interval. Cleaning method and biocide dose are fixed for now.

### Assumptions

- Lumped, 1-D model: one representative tube with uniform film thickness along its length.
- Film thermal conductivity is constant.
- Growth depends on water temperature through a cardinal-temperature (Ratkowsky-type) function.
- Removal is proportional to wall shear stress and thickness.
- Cleaning removes a fixed fraction of the film (90% mechanical, 95% chemical in `core/config.py`).

### Limitations

- **Synthetic data only.** No plant telemetry is used, and no plant has been modelled.
- Microfouling only; no macrofouling, scale kinetics or corrosion kinetics.
- Cost parameters in `core/config.py` are illustrative, not plant data.
- **Results are not transferable to a real plant** without calibration against that plant's data
  and review under its procedures.

## 8. Success criteria and evaluation indicators

The proposed system is compared against a **fixed 30-day cleaning interval** and a **persistence
forecast** (tomorrow equals today). It is evaluated on:

| Indicator | Unit | Source |
|---|---|---|
| Cleaning events per year | count | schedule simulation |
| Thermal performance: mean and minimum U, heat duty Q | W/m²K, MW | observation model |
| Excess pumping energy | kWh/yr | `optimizer/sustainability_metrics.py` |
| Downtime | h/yr | events x `DOWNTIME_HOURS_PER_CLEANING` (4 h) |
| Biocide use vs permit limit | kg/yr vs 250 kg/yr | `optimizer/sustainability_metrics.py` |
| CO2e from excess pumping | kg/yr | 0.45 kg CO2/kWh (`core/config.py`) |
| Constraint violations | count, hours | Rf, dP and BSI limits in `core/config.py` |
| Forecast error at 1, 24, 72 and 168 h | MAE, RMSE, R² | leak-free evaluation (model.md, Section 8) |

The project counts as successful if the proposed schedule has **lower total cost than fixed 30-day
with no more constraint violations**, on both the matched and the mismatched synthetic test sets,
and if forecast skill beats persistence at the 24 h horizon and longer.

## 9. References

- Rao et al. (2009). Biofouling 25(7), 581–591. DOI: 10.1080/08927010903016543.
  Failure analysis of a thermo-fluid heat exchanger at a nuclear test reactor, Kalpakkam.
- Chen et al. (2024). Energy 294, 130814. PSO-LSTM approach applied to condenser fouling. Cited as
  precedent for ML-assisted cleaning decisions only; the performance gains reported there are the
  authors' results and are not claimed for this project.
