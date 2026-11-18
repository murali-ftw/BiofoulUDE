# AI-Assisted Optimisation of Heat-Exchanger Cleaning for Biofouling Control in Nuclear Cooling Systems

> **Academic demonstration only.** All data in this repository is **synthetic**, produced by a
> numerical simulator. Nothing here is validated against plant telemetry, and it must not be used
> for operating, maintaining or licensing a real nuclear plant.

University EVS case study. Biofilm growing inside the tubes of a cooling-water heat exchanger
adds thermal resistance (`1/U = 1/U_clean + Rf`) and raises the pressure drop and pumping energy.
This project builds one pipeline that turns a model of that fouling into a cleaning decision:

```
synthetic data -> Universal Differential Equation (physics ODE + small neural-net correction)
               -> cost-optimal cleaning interval
```

The UDE and the new interval search are **not implemented yet**. The repository currently contains
the data generator, the fouling physics, two comparison forecasters (Ridge, Ridge+GBM) and the
existing scenario-based cleaning optimiser. See [docs/](docs/) for the problem statement, the
proposed model and an implementation-status table.

## Repository layout

```
config.py                  physical constants, thresholds, cost parameters
synthetic_generator.py     8,760 h synthetic fouling dataset (Ratkowsky growth + deposition/removal ODE)
fouling_indicators.py      Rf -> U, Darcy-Weisbach/Haaland pressure drop, severity index
preprocessor.py            chronological split and feature pipeline
baseline_model.py          Ridge comparison model
improved_model.py          Ridge + gradient-boosting comparison model
cleaning_optimizer.py      current scenario-based cleaning-interval optimiser
sustainability_metrics.py  pumping energy, CO2e and biocide accounting
main.py                    runs the full pipeline end to end
tests/                     unit tests
docs/                      problem statement and model description
```

## Setup (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Generate data and run

```bash
python main.py                   # simulate (seed 42, 8,760 h), train baselines, optimise, report
```

## Tests

```bash
python -m pytest tests -q
```

## License

MIT, see [LICENSE](LICENSE).
