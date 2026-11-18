# Results 1: UDE ensemble vs plain ODE

> **Synthetic data only; academic demonstration, not for plant use.** All numbers come from the project's own
> generator (`core/synthetic_generator.py`). Nothing here has been validated against plant telemetry.

Generated from `ml/artifacts/metrics.json` and `ml/artifacts/training_log.json` (produced by `python -m ml.train --float64`
and `python -m ml.evaluate`). Regenerate those files and this report changes accordingly.

## 1. The two models

| | Plain ODE | UDE ensemble |
|---|---|---|
| Structure | Generator physics with g = 1 | Same physics × learned correction g = exp(MLP), 2×16 tanh |
| Learned | K_dep, K_rem | K_dep, K_rem and the network weights |
| Members | 1 | 5 seeds (11, 12, 13, 14, 15) |
| Forecast trajectories | 1 model × 3 initial states | 5 models × 3 initial states |
| Training data | Measured U and dP only (never Rf) | Measured U and dP only (never Rf) |

Training used the nominal year, split by whole cleaning cycles: train h 0-5758, validation h 5759-7198, test h 7199-8759. That gives 226 training and 46 validation windows of 14 days. It ran on cpu (float64) in 225 s. In the benchmark the CPU took 101 ms per step against 313 ms on MPS.

## 2. Fitted parameters and validation loss

The validation loss is a Huber loss on noise-standardised U and dP residuals over the nominal validation windows; lower is
better. True physics with the same estimated initial states scores 1.033, which is the noise floor.

| Model | K_dep | K_rem | Validation loss |
|---|---|---|---|
| `core/config.py` (generator) | 1.600e-07 | 4.200e-06 | 1.033 (true physics) |
| Plain ODE | 1.918e-07 | 3.555e-07 | 1.1380 |
| UDE seed 11 | 1.761e-07 | 3.704e-07 | 1.0175 |
| UDE seed 12 | 1.796e-07 | 3.617e-07 | 1.0179 |
| UDE seed 13 | 1.828e-07 | 3.466e-07 | 1.0193 |
| UDE seed 14 | 1.815e-07 | 3.485e-07 | 1.0179 |
| UDE seed 15 | 1.760e-07 | 3.500e-07 | 1.0178 |

Both models fit `K_rem` about 12× below the generator value. On nominal data removal is only about 2% of deposition, so
`K_rem` is practically unidentifiable. The ODE's higher `K_dep` absorbs the generator's seasonal nutrient factor, which
its physics lacks; the UDE moves `K_dep` back toward the config value because `g` takes over the seasonal part.

## 3. Forecast scores

- **Origins:** every 24 h.
- **Future inputs:** climatology (seasonal T, nominal flow).
- **Cleanings:** the planned cleanings inside the horizon are known.
- **Point forecast:** the mean over each model's trajectories.
- **Ground truth:** Rf errors are against the true Rf; U and dP errors are against the measured (noisy) values.

Units: Rf errors in 1e-5 m²K/W, U in W/m²K, dP in kPa. The **better** model per horizon is in bold. This is decided
on unrounded values, so two entries can look identical after rounding and still have one in bold; the difference is
then in later digits and is negligible.

### Nominal test (cycles 10-12) (58 origins, h 7223-8591)

| Quantity | Metric | Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|---|---|
| Rf | RMSE | ODE | **0.20** | **0.19** | 0.20 | 0.34 |
| Rf | RMSE | UDE | 0.20 | 0.20 | **0.18** | **0.16** |
| Rf | MAE | ODE | **0.17** | **0.16** | 0.16 | 0.28 |
| Rf | MAE | UDE | 0.17 | 0.16 | **0.15** | **0.12** |
| Rf | R² | ODE | **0.973** | **0.976** | 0.974 | 0.934 |
| Rf | R² | UDE | 0.973 | 0.974 | **0.979** | **0.986** |
| U | RMSE | ODE | **26.9** | 25.1 | 27.7 | 30.8 |
| U | RMSE | UDE | 26.9 | **24.9** | **26.2** | **25.0** |
| U | MAE | ODE | **22.6** | 21.1 | 22.2 | 25.3 |
| U | MAE | UDE | 22.6 | **21.0** | **21.6** | **21.0** |
| U | R² | ODE | **0.770** | 0.802 | 0.785 | 0.764 |
| U | R² | UDE | 0.769 | **0.805** | **0.808** | **0.845** |
| dP | RMSE | ODE | **0.239** | 0.270 | 0.274 | 0.286 |
| dP | RMSE | UDE | 0.239 | **0.270** | **0.270** | **0.270** |
| dP | MAE | ODE | **0.194** | 0.221 | 0.222 | 0.234 |
| dP | MAE | UDE | 0.194 | **0.221** | **0.219** | **0.219** |
| dP | R² | ODE | **0.460** | 0.453 | 0.451 | 0.433 |
| dP | R² | UDE | 0.460 | **0.455** | **0.466** | **0.495** |

Coverage of the 90% band for Rf (target 0.90):

| Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| ODE | 0.97 | 0.93 | 0.88 | 0.55 |
| UDE | 0.97 | 0.97 | 0.97 | 0.98 |

Ablation with oracle future inputs (measured future T and flow), Rf RMSE (1e-5 m²K/W):

| Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| ODE, oracle inputs | 0.20 | 0.19 | 0.20 | 0.35 |
| UDE, oracle inputs | 0.20 | 0.20 | 0.18 | 0.16 |

### Stress: never cleaned (357 origins, h 24-8568)

| Quantity | Metric | Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|---|---|
| Rf | RMSE | ODE | **0.52** | **0.50** | 0.47 | 0.51 |
| Rf | RMSE | UDE | 0.52 | 0.50 | **0.47** | **0.47** |
| Rf | MAE | ODE | **0.40** | **0.38** | 0.36 | 0.41 |
| Rf | MAE | UDE | 0.40 | 0.38 | **0.36** | **0.35** |
| Rf | R² | ODE | **0.999** | **0.999** | 0.999 | 0.999 |
| Rf | R² | UDE | 0.999 | 0.999 | **0.999** | **0.999** |
| U | RMSE | ODE | **25.1** | **26.1** | **25.4** | 26.1 |
| U | RMSE | UDE | 25.1 | 26.2 | 25.5 | **25.4** |
| U | MAE | ODE | **20.0** | **21.0** | **20.4** | 20.8 |
| U | MAE | UDE | 20.0 | 21.1 | 20.4 | **20.1** |
| U | R² | ODE | **0.996** | **0.996** | **0.996** | 0.996 |
| U | R² | UDE | 0.996 | 0.996 | 0.996 | **0.996** |
| dP | RMSE | ODE | **0.376** | **0.426** | **0.415** | 0.430 |
| dP | RMSE | UDE | 0.376 | 0.426 | 0.415 | **0.429** |
| dP | MAE | ODE | 0.289 | **0.326** | **0.314** | 0.332 |
| dP | MAE | UDE | **0.289** | 0.326 | 0.315 | **0.330** |
| dP | R² | ODE | **0.984** | **0.980** | **0.981** | 0.979 |
| dP | R² | UDE | 0.984 | 0.980 | 0.981 | **0.979** |

Coverage of the 90% band for Rf (target 0.90):

| Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| ODE | 0.95 | 0.96 | 0.95 | 0.87 |
| UDE | 0.97 | 0.98 | 0.98 | 0.97 |

Ablation with oracle future inputs (measured future T and flow), Rf RMSE (1e-5 m²K/W):

| Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| ODE, oracle inputs | 0.52 | 0.50 | 0.47 | 0.51 |
| UDE, oracle inputs | 0.52 | 0.50 | 0.47 | 0.46 |

### Stress: 90-day cleaning (357 origins, h 24-8568)

| Quantity | Metric | Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|---|---|
| Rf | RMSE | ODE | **0.41** | **0.39** | 0.37 | 0.42 |
| Rf | RMSE | UDE | 0.41 | 0.39 | **0.36** | **0.31** |
| Rf | MAE | ODE | **0.31** | **0.29** | 0.28 | 0.35 |
| Rf | MAE | UDE | 0.31 | 0.30 | **0.27** | **0.24** |
| Rf | R² | ODE | **0.999** | **0.999** | 0.999 | 0.999 |
| Rf | R² | UDE | 0.999 | 0.999 | **0.999** | **0.999** |
| U | RMSE | ODE | **27.2** | **26.3** | **26.1** | 26.8 |
| U | RMSE | UDE | 27.2 | 26.5 | 26.2 | **25.6** |
| U | MAE | ODE | **22.2** | **21.2** | 20.7 | 21.1 |
| U | MAE | UDE | 22.2 | 21.2 | **20.6** | **20.1** |
| U | R² | ODE | **0.992** | **0.992** | **0.992** | 0.992 |
| U | R² | UDE | 0.992 | 0.992 | 0.992 | **0.993** |
| dP | RMSE | ODE | 0.302 | **0.323** | 0.317 | 0.319 |
| dP | RMSE | UDE | **0.302** | 0.323 | **0.316** | **0.314** |
| dP | MAE | ODE | 0.236 | **0.255** | 0.248 | 0.252 |
| dP | MAE | UDE | **0.236** | 0.255 | **0.248** | **0.248** |
| dP | R² | ODE | 0.976 | **0.972** | 0.973 | 0.972 |
| dP | R² | UDE | **0.976** | 0.972 | **0.973** | **0.973** |

Coverage of the 90% band for Rf (target 0.90):

| Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| ODE | 0.96 | 0.96 | 0.92 | 0.78 |
| UDE | 0.97 | 0.98 | 0.99 | 0.98 |

Ablation with oracle future inputs (measured future T and flow), Rf RMSE (1e-5 m²K/W):

| Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| ODE, oracle inputs | 0.41 | 0.39 | 0.37 | 0.42 |
| UDE, oracle inputs | 0.41 | 0.39 | 0.36 | 0.31 |

### Mismatch stress (held out) (357 origins, h 24-8568)

| Quantity | Metric | Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|---|---|
| Rf | RMSE | ODE | **0.43** | 0.64 | 1.19 | 2.28 |
| Rf | RMSE | UDE | 0.43 | **0.63** | **1.17** | **2.21** |
| Rf | MAE | ODE | **0.33** | **0.41** | **0.59** | **0.98** |
| Rf | MAE | UDE | 0.33 | 0.41 | 0.61 | 1.02 |
| Rf | R² | ODE | **0.999** | 0.998 | 0.992 | 0.972 |
| Rf | R² | UDE | 0.999 | **0.998** | **0.993** | **0.973** |
| U | RMSE | ODE | **27.0** | **29.8** | 39.5 | 60.3 |
| U | RMSE | UDE | 27.0 | 29.8 | **39.1** | **58.9** |
| U | MAE | ODE | **21.5** | **22.8** | **26.7** | **34.4** |
| U | MAE | UDE | 21.5 | 22.9 | 26.8 | 34.7 |
| U | R² | ODE | **0.993** | **0.992** | 0.986 | 0.966 |
| U | R² | UDE | 0.993 | 0.992 | **0.986** | **0.968** |
| dP | RMSE | ODE | **0.322** | 0.322 | 0.357 | 0.480 |
| dP | RMSE | UDE | 0.322 | **0.322** | **0.355** | **0.471** |
| dP | MAE | ODE | **0.253** | **0.254** | **0.274** | **0.323** |
| dP | MAE | UDE | 0.253 | 0.254 | 0.275 | 0.324 |
| dP | R² | ODE | **0.980** | 0.980 | 0.976 | 0.956 |
| dP | R² | UDE | 0.980 | **0.980** | **0.976** | **0.957** |

Coverage of the 90% band for Rf (target 0.90):

| Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| ODE | 0.96 | 0.91 | 0.82 | 0.69 |
| UDE | 0.97 | 0.92 | 0.85 | 0.64 |

Ablation with oracle future inputs (measured future T and flow), Rf RMSE (1e-5 m²K/W):

| Model | 1 h | 24 h | 72 h | 168 h |
|---|---|---|---|---|
| ODE, oracle inputs | 0.43 | 0.64 | 1.20 | 2.29 |
| UDE, oracle inputs | 0.43 | 0.63 | 1.17 | 2.22 |

## 4. Time to limit

The origins are every 24 h, from up to 60 days before the true first crossing. Each model's predicted crossing hour is the
median over its trajectories. The table gives the mean absolute error in hours, binned by lead time. The 7-day linear
trend is a simple reference, not one of the two models.

| Dataset | Limit (actual hour) | Method | Lead 1-7 d | Lead 7-30 d | Lead 30-60 d | All | 90% interval covers actual | Never predicted |
|---|---|---|---|---|---|---|---|---|
| Stress: never cleaned | rf (3305) | ODE | 13 h | 31 h | 106 h | 66 h | 0.38 | 0 / 60 |
| Stress: never cleaned | rf (3305) | UDE | 12 h | 32 h | 70 h | 49 h | 0.48 | 0 / 60 |
| Stress: never cleaned | rf (3305) | 7-day trend (reference) | 11 h | 39 h | 198 h | 115 h | – | 0 / 60 |
| Stress: never cleaned | dp (4399) | ODE | 1028 h | 1037 h | 979 h | 1017 h | 0.00 | 16 / 60 |
| Stress: never cleaned | dp (4399) | UDE | 916 h | 885 h | 779 h | 838 h | 0.00 | 2 / 60 |
| Stress: never cleaned | dp (4399) | 7-day trend (reference) | 808 h | 1027 h | 592 h | 784 h | – | 0 / 60 |
| Stress: 90-day cleaning | rf (3722) | ODE | 26 h | 15 h | 48 h | 33 h | 0.60 | 0 / 60 |
| Stress: 90-day cleaning | rf (3722) | UDE | 26 h | 28 h | 67 h | 47 h | 0.42 | 0 / 60 |
| Stress: 90-day cleaning | rf (3722) | 7-day trend (reference) | 38 h | 56 h | 83 h | 67 h | – | 0 / 60 |
| Stress: 90-day cleaning | dp: never crossed | – | – | – | – | – | – | – |
| Mismatch stress (held out) | rf (3964) | ODE | 57 h | 131 h | 234 h | 174 h | 0.12 | 0 / 60 |
| Mismatch stress (held out) | rf (3964) | UDE | 42 h | 83 h | 161 h | 117 h | 0.18 | 0 / 60 |
| Mismatch stress (held out) | rf (3964) | 7-day trend (reference) | 13 h | 72 h | 483 h | 267 h | – | 1 / 60 |
| Mismatch stress (held out) | dp (4985) | ODE | – | – | – | – | 0.00 | 60 / 60 |
| Mismatch stress (held out) | dp (4985) | UDE | – | – | – | – | 0.00 | 60 / 60 |
| Mismatch stress (held out) | dp (4985) | 7-day trend (reference) | 301 h | 172 h | 253 h | 228 h | – | 0 / 60 |

## 5. Head to head

| Dataset | Horizons where the UDE has lower Rf RMSE | ODE lower | Tie | Rf RMSE at 168 h, UDE vs ODE (1e-5) | Difference | 90% coverage at 168 h, UDE vs ODE |
|---|---|---|---|---|---|---|
| Nominal test (cycles 10-12) | 2 of 4 | 2 | 0 | 0.16 vs 0.34 | UDE 54% lower | 0.98 vs 0.55 |
| Stress: never cleaned | 2 of 4 | 2 | 0 | 0.47 vs 0.51 | UDE 8% lower | 0.97 vs 0.87 |
| Stress: 90-day cleaning | 2 of 4 | 2 | 0 | 0.31 vs 0.42 | UDE 25% lower | 0.98 vs 0.78 |
| Mismatch stress (held out) | 3 of 4 | 1 | 0 | 2.21 vs 2.28 | UDE 3% lower | 0.64 vs 0.69 |

## 6. Scores as percentages

Rf is a continuous quantity, so there is no classification accuracy. Two standard regression scores can be stated as
percentages:

- **R² × 100:** share of the variation in true Rf that the forecast explains (100% = perfect).
- **Relative accuracy = 100 × (1 − MAE / mean true Rf):** for example, 94% means the forecast is off by about 6% of the
  average Rf.

| Dataset | Model | 1 h: R² / rel. acc. | 24 h: R² / rel. acc. | 72 h: R² / rel. acc. | 168 h: R² / rel. acc. |
|---|---|---|---|---|---|
| Nominal test (cycles 10-12) | ODE | 97.3% / 92.4% | 97.6% / 93.0% | 97.4% / 92.5% | 93.4% / 86.7% |
| Nominal test (cycles 10-12) | UDE | 97.3% / 92.4% | 97.4% / 92.7% | 97.9% / 93.2% | 98.6% / 94.1% |
| Stress: never cleaned | ODE | 99.9% / 98.7% | 99.9% / 98.8% | 99.9% / 98.9% | 99.9% / 98.8% |
| Stress: never cleaned | UDE | 99.9% / 98.7% | 99.9% / 98.8% | 99.9% / 98.9% | 99.9% / 98.9% |
| Stress: 90-day cleaning | ODE | 99.9% / 98.0% | 99.9% / 98.1% | 99.9% / 98.2% | 99.9% / 97.8% |
| Stress: 90-day cleaning | UDE | 99.9% / 98.0% | 99.9% / 98.1% | 99.9% / 98.3% | 99.9% / 98.5% |
| Mismatch stress (held out) | ODE | 99.9% / 98.0% | 99.8% / 97.6% | 99.2% / 96.4% | 97.2% / 94.1% |
| Mismatch stress (held out) | UDE | 99.9% / 98.0% | 99.8% / 97.5% | 99.3% / 96.3% | 97.3% / 93.9% |

Headline: on nominal test at 168 h the UDE scores **R² = 98.6%** with **relative accuracy = 94.1%** (average error about
6%), against 93.4% and 86.7% for the ODE.

Caution when quoting these numbers:
- **The stress sets inflate the scores.** Rf varies over a wide range there, which pushes R² toward 100% for any
  reasonable forecast; persistence alone scores above 99% R² on the never-cleaned set (see `ml/README.md`).
- **The mismatch set is the fairest test.** There the two models are essentially equal.

## 7. Reading the numbers

**Where the UDE is better:**
- **Long horizons.** At 72 h and 168 h its Rf error is lower than the ODE's on every dataset. The biggest gain is on nominal
  test (about 2× at 168 h); on the held-out mismatch set the gain is only about 3%.
- **Bands.** Its bands stay calibrated at long horizons on matched data, where the ODE's become far too narrow
  (0.55 coverage at 168 h on nominal test).
- **Learned correction.** `g` recovered the generator's seasonal nutrient factor: 0.72 in winter to 1.16 in midsummer
  (see `ml/artifacts/correction_slices.png`).

**Where it is not:**
- **Short horizons.** At 1-24 h the two models are essentially tied (Rf RMSE within 0.01e-5 m²K/W). Both lose to plain persistence
  at 1 h (see `ml/README.md`), because the 12 h state estimate is noisier than a 24 h median of U.
- **Mismatch bands.** On the mismatch set both under-cover at 168 h; the bloom and scale deposit are not in either model.
- **Time to the Rf limit.** The UDE is better on the never-cleaned and mismatch sets, but the ODE is better on the 90-day
  set at 7-60 days' lead.
- **Time to the dP limit.** Poor for both models, and on the mismatch set neither ever predicts the crossing. The dP limit is
  reached near the carrying capacity, where growth is slow and the crossing time depends on the poorly identified `K_rem`.

**Limitations:**
- The UDE shares most of its structure with the generator, so the nominal and stress scores are optimistic. The mismatch set
  is the fairest test.
- 15 trajectories cannot resolve small exceedance probabilities.
- All results are synthetic.
