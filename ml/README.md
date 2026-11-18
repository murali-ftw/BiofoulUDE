# ml/: Universal Differential Equation (UDE) fouling model

> **SYNTHETIC DATA, academic demonstration only. Not for plant use.** Every number below comes from the project's
> own generator (`core/synthetic_generator.py`). Nothing has been validated against plant telemetry.

## What the model is

The state is biofilm thickness δ (m). The model takes hourly explicit Euler steps that mirror `synthetic_generator.step()`:

```
dep = K_dep · r(T) · g_θ(x) · φ_lag(cum_clean_hours) · (1 − δ/δ_max)
rem = K_rem · τ_w(v) · δ
δ  ← clamp(δ + dep − rem, 0, δ_max);  on a cleaning: δ ← (1 − η[type]) · δ
```

- **Known physics:**
  - Ratkowsky factor `r(T)`.
  - The generator's lag phase, driven by `cum_clean_hours`, which is replayed from the maintenance log with the
    generator's `max(24, 0.2·cum)` memory rule.
  - Carrying capacity.
  - Shear removal with Darcy–Weisbach/Haaland wall shear.
  - Cleaning efficiencies of 0.90 (mechanical) and 0.95 (chemical).
- **What the physics lacks:** on nominal data, the generator's seasonal **nutrient factor** `1 + 0.3·sin(phase)`.
  `g_θ` has to learn it. The mismatch set adds drift, a scale deposit and a bloom on top.
- **The correction `g_θ = exp(net(x))`:** an MLP with 2 × 16 tanh layers. Its last layer starts at zero, so the UDE
  equals the ODE at initialisation.
  - **Inputs:** 5 variables given as **6 numbers**, because season is encoded as sin and cos:
    - T, as (T − 6)/39;
    - v, as v/1.414 m/s;
    - season, as sin and cos of 2πh/8760;
    - δ/δ_max;
    - hours since cleaning, as /720 clipped to [0, 1].
  - **δ/δ_max is the latent state.** In training and forecasting the model rolls it forward itself; it is never
    read from the data.
- **`K_dep`, `K_rem`:** learnable and log-parameterised, initialised from `core/config.py`.
- **Observation model:** δ → Rf = δ/k → U (1/U = 1/U_clean + Rf), and δ → D_eff, roughness → dP. The torch port in
  `physics_torch.py` matches numpy to within 1e-5 relative error.
- **Check:** with `g` set to the generator's nutrient factor, true inputs and true δ0, the module reproduces the
  generator's δ to `rtol 1e-9` over two cleaning cycles (`tests/test_ml_ude.py`).

## No leakage

- **Training inputs:** only measured T, flow, U and dP, plus the maintenance log.
- **Loss:** Huber on U and dP residuals, standardised by sensor noise (σ_U = 25 W/m²K, σ_dP = 0.20 kPa).
  **Rf is never an input or part of the loss.**
- **Initial state:** from the median of the U-based and dP-based inversions over 12 burn-in hours, never from truth.
- **Forecasts:** use only rows at or before the origin. Future U and dP are never read. Future T and flow are read
  only in the "oracle" ablation.
- **Tested:** corrupting future observations leaves the forecast unchanged; the baseline features ignore rows after t0.

## How to run (from the repository root)

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, scikit-learn, torch
python -m ml.bench_device           # time CPU vs MPS, write ml/artifacts/device.json
python -m ml.train --float64        # ODE + 5-seed UDE ensemble on nominal train (~4 min on CPU)
python -m ml.evaluate               # all models x datasets x horizons -> ml/artifacts/metrics.json (~5 min)
python -m pytest tests -q
```

`matplotlib` is optional: if it's installed, `ml.evaluate` also writes the PNG plots. Use the forecast API from Python:

```python
from ml import data, forecast
hist = data.load("nominal").iloc[:3001]                       # history up to the origin (hour 3000)
f = forecast.forecast(hist, 168, planned_cleanings=[{"hour": 3100, "type": "mechanical"}], model="ude")
r = forecast.risk_by_interval(data.load("nominal"), 3000)    # P(limit crossed before T), T = 1..90 d
g = forecast.correction_slices()
```

## Device

`python -m ml.bench_device` times one real training step: 128 windows × 336 h, rolled out in batch and run with
backward.

| Device | Median step time |
|---|---|
| CPU float32 | 105 ms |
| CPU float64 | 101 ms |
| MPS float32 | 314 ms |

**CPU wins** (Apple M4 Pro, torch 2.15). The model is a single state with about 400 network parameters, so MPS
kernel-launch overhead dominates. Training therefore ran on CPU in float64.

- **MPS still works:** `ML_DEVICE=mps python -m ml.train` uses float32, because MPS has no float64. Results are not
  bit-reproducible on MPS.
- **CPU checks:** parity, gradcheck and determinism are all checked on CPU in float64.
- **Checkpoints:** stored as CPU tensors; inference runs on CPU.

## Data and training

- **Split:** the nominal year is split chronologically by whole cleaning cycles.
  - **Train:** cycles 0–7, h 0–5758.
  - **Validation:** cycles 8–9, h 5759–7198.
  - **Test:** cycles 10–12, h 7199–8759.
- **Windows:** 14 days long, starting after each cleaning and every 24 h within cycles, with no cleaning inside the
  12 h burn-in. That gives 226 training and 46 validation windows. Each ensemble seed shifts the start offsets.
- **Stage 1 (ODE):** `g` frozen at 1; `K_dep` and `K_rem` fitted with Adam, LR 0.03, 400 iterations.
- **Stage 2 (UDE):** 5 seeds, each starting from the ODE fit.
  - Learning rates: K's 3e-3, net 3e-3.
  - Regularisation: weight decay 1e-4 and 0.01·mean(log g)².
  - Gradient clipping at 1.0, and early stopping on nominal validation.
- **Hyperparameters:** chosen on nominal validation only. `lr_net` values of 1e-3, 3e-3 and 1e-2 gave equal
  validation loss, so I kept the fastest.

| | K_dep | K_rem | Validation loss |
|---|---|---|---|
| `core/config.py` (generator) | 1.60e-7 | 4.2e-6 | 1.033 (true physics; noise floor) |
| ODE | 1.92e-7 | 3.55e-7 | 1.138 |
| UDE, 5 seeds | 1.76–1.83e-7 | 3.47–3.70e-7 | 1.018–1.019 |

**What the fitted parameters show:**
- **`K_dep` (ODE):** above the config value, because it absorbs the missing nutrient factor, which is largest when
  growth is fastest. The UDE moves it back toward config.
- **`K_rem`:** fitted about 12× too low by both models. On nominal data it is **practically unidentifiable**: with
  cleaning every 30 days, removal is only about 2% of deposition, a time constant of about 3.5 years.
- **Validation loss:** the UDE reaches the noise floor.

## Results

Evaluation setup:
- **Origins:** every 24 h. Nominal test has 58 origins; each stress and mismatch file has 357, covering the whole year.
- **Future inputs:** climatology, i.e. seasonal T and nominal flow.
- **Cleanings:** planned cleanings in the horizon are known to every model.
- **Final evaluation:** the stress and mismatch files were evaluated once, as a final evaluation.
- **Baselines:** Ridge and GBM are trained only on nominal-train origins, with true Rf at t0+h as the target. They do
  see Rf labels; the UDE never does.

### Rf RMSE (1e-5 m²K/W) by horizon

| Dataset | Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|---|
| nominal test | persistence | **0.13** | 0.61 | 1.09 | 1.64 |
| | Ridge | **0.13** | 0.47 | 1.13 | 1.65 |
| | GBM | 0.17 | 0.27 | 0.57 | 1.05 |
| | ODE | 0.20 | **0.19** | 0.20 | 0.34 |
| | UDE | 0.20 | 0.20 | **0.18** | **0.16** |
| stress_no_clean | persistence | **0.39** | **0.46** | 0.75 | 1.45 |
| | Ridge | 2.77 | 9.18 | 18.45 | 38.81 |
| | GBM | 20.59 | 26.71 | 27.11 | 28.67 |
| | ODE | 0.52 | 0.50 | 0.47 | 0.51 |
| | UDE | 0.52 | 0.50 | **0.47** | **0.47** |
| stress_long_interval | persistence | **0.31** | 2.70 | 4.68 | 7.17 |
| | Ridge | 0.50 | 2.29 | 4.57 | 8.45 |
| | GBM | 6.57 | 10.81 | 10.94 | 10.94 |
| | ODE | 0.41 | **0.39** | 0.37 | 0.42 |
| | UDE | 0.41 | **0.39** | **0.36** | **0.31** |
| mismatch_stress (held-out) | persistence | **0.38** | 3.57 | 6.06 | 8.99 |
| | Ridge | 0.44 | 2.80 | 6.22 | 11.09 |
| | GBM | 9.31 | 12.64 | 12.79 | 12.98 |
| | ODE | 0.43 | 0.64 | 1.19 | 2.28 |
| | UDE | 0.43 | **0.63** | **1.17** | **2.21** |

`metrics.json` also has MAE, R², and the U and dP errors. With oracle future T and flow, the ODE and UDE errors
change by at most 0.01. The reason is that the generator's true temperature is exactly seasonal plus a ±1.2 °C diurnal
term, so climatology is already nearly the oracle.

### 90% band coverage for Rf (target 0.90)

| Dataset | Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|---|
| nominal test | UDE | 0.97 | 0.97 | 0.97 | 0.98 |
| | ODE | 0.97 | 0.93 | 0.88 | 0.55 |
| stress_no_clean | UDE | 0.97 | 0.98 | 0.98 | 0.97 |
| | ODE | 0.95 | 0.96 | 0.95 | 0.87 |
| stress_long_interval | UDE | 0.97 | 0.98 | 0.99 | 0.98 |
| | ODE | 0.96 | 0.96 | 0.92 | 0.78 |
| mismatch_stress | UDE | 0.97 | 0.92 | 0.85 | **0.64** |
| | ODE | 0.96 | 0.91 | 0.82 | 0.69 |

How the band is built: each of the 5 members runs from 3 initial states, δ̂0 ± 1.645·se. On matched data the UDE is
slightly over-covered; on the mismatch set it is under-covered at long horizons.

### Time to limit

Origins are every 24 h, from up to 60 days before the true first crossing. The table gives the mean absolute error of
the predicted crossing hour (median of the ensemble).

| Dataset, limit (actual h) | Method | lead 1–7 d | lead 7–30 d | lead 30–60 d | 90% interval covers actual |
|---|---|---|---|---|---|
| stress_no_clean, Rf (3305) | UDE | 12 h | 32 h | 70 h | 0.48 |
| | ODE | 13 h | 31 h | 106 h | 0.38 |
| | 7-day linear trend | 11 h | 39 h | 198 h | – |
| stress_long_interval, Rf (3722) | UDE | 26 h | 28 h | 67 h | 0.42 |
| | ODE | 26 h | **15 h** | **48 h** | 0.60 |
| | 7-day linear trend | 38 h | 56 h | 83 h | – |
| mismatch_stress, Rf (3964) | UDE | 42 h | 83 h | 161 h | 0.18 |
| | ODE | 57 h | 131 h | 234 h | 0.12 |
| | 7-day linear trend | **13 h** | 72 h | 483 h | – |
| stress_no_clean, dP (4399) | UDE | 916 h | 885 h | 779 h | 0.00 |
| | ODE | 1028 h | 1037 h | 979 h | 0.00 |
| | 7-day linear trend | 808 h | 1027 h | 592 h | – |
| mismatch_stress, dP (4985) | UDE / ODE | never predicted at any of the 60 origins | | | 0.00 |
| | 7-day linear trend | 301 h | 172 h | 253 h | – |

`stress_long_interval` never crosses the dP limit.

## Verdict

**Where the UDE is best:**
- **Beyond 24 h it has the lowest Rf error on every dataset.** On nominal test at 168 h its RMSE is 1.6e-6 against
  3.4e-6 for the ODE (about 2× better) and 1.6e-5 for persistence (about 10× better). On the held-out mismatch set its
  lead over the ODE is small (2.21 against 2.28) but present.
- **Its bands are well calibrated on matched data**, where the ODE's are badly under-covered at 168 h (0.55).
- **For time to the Rf limit it beats the 7-day trend at 30–60 days' lead on all three crossing datasets**, and beats
  the ODE there on two of the three.

**Where the UDE does not win:**
- **At 1 h, persistence (and nominal Ridge) is better everywhere**, because the 12 h state estimate is noisier than a
  24 h U-median.
- **On `stress_no_clean`, persistence also wins at 24 h.**
- **On `stress_long_interval`, the ODE beats the UDE for time to the Rf limit at 7–60 days' lead.**
- **On mismatch at short lead (1–7 d), the linear trend beats it for time to the Rf limit.**
- **Its mismatch bands under-cover** (0.64 at 168 h): the network cannot know about the bloom or the scale deposit.
- **Time to the dP limit is poor for every method.** The dP limit is crossed near the carrying capacity, where growth
  is slow and the crossing time depends on the poorly identified `K_rem`. On the mismatch set the UDE and ODE never
  predict the crossing at all.

**Ridge and GBM:** competitive only on nominal data. On the stress sets Rf goes beyond anything in their training
range, and they fail badly.

**What `g` learned:** a seasonal multiplier from about 0.72 (winter) to 1.16 (midsummer). It closely follows the
generator's nutrient factor rescaled by config K_dep/fitted K_dep (0.62–1.16), departing only in deep winter, where
growth is near zero. It depends only weakly on T, δ and hours since cleaning. See `artifacts/correction_slices.png`.

## Limitations

- **Synthetic data only.** The UDE shares most of its structure with the generator, so the nominal test is easy. The
  mismatch set is the honest test, and there the UDE's advantage over the ODE is small.
- **`K_rem` is unidentifiable from nominal data.** This mainly hurts forecasts near the carrying capacity (the dP limit).
- **The uncertainty is crude.** Five members times three initial states give 15 trajectories. That is not enough for
  small exceedance probabilities, and it does not include sensor noise or missing mechanisms.
- **`risk_by_interval`** gives the first-cycle chance constraint only. It is not wired into `optimizer/cleaning_optimizer.py`.
- **Planned-cleaning features in the baselines.** Ridge and GBM were given the planned-cleaning timing as features,
  which is known in advance. Persistence ignores planned cleanings by definition.
