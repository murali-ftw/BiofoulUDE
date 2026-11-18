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

The UDE is in [ml/](ml/README.md) (trained on measured U and dP only, with results), the
re-simulation cost evaluation is in `cleaning_optimizer.py`, and [web/](web/README.md) is a local
dashboard. See [docs/](docs/) for the problem statement, the model and its status.

## Repository layout

```
config.py                  physical constants, thresholds, cost parameters
synthetic_generator.py     8,760 h synthetic fouling dataset (Ratkowsky growth + deposition/removal ODE)
fouling_indicators.py      Rf -> U, Darcy-Weisbach/Haaland pressure drop, severity index
preprocessor.py            chronological split and feature pipeline
baseline_model.py          Ridge comparison model
improved_model.py          Ridge + gradient-boosting comparison model
cleaning_optimizer.py      shared cost function, policy re-simulation, interval sweep
ml/                        UDE model, training, baselines, evaluation, forecast API
web/                       local dashboard (python -m web.server; --mock without a model)
sustainability_metrics.py  pumping energy, CO2e and biocide accounting
main.py                    runs the full pipeline end to end
tests/                     unit tests
db/                        nominal, stress and mismatch datasets, and generate.py
docs/                      problem statement and model description
```

## Setup (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Generate data, run, test

```bash
python db/generate.py            # regenerate all datasets in db/
python -m ml.train --float64     # train ODE + UDE ensemble (~4 min, CPU)
python -m ml.evaluate            # metrics -> ml/artifacts/metrics.json
python -m web.server             # dashboard at http://127.0.0.1:8050/
python -m pytest tests -q        # unit tests
```

## License

MIT, see [LICENSE](LICENSE).
