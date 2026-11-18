# web/: interactive dashboard

> **SYNTHETIC DATA, academic demonstration only. Not for plant use.** Every chart carries a
> `SYNTHETIC` badge, plus `ML_FORECAST` or `PHYSICS` where model output or physics re-simulation is shown.

A local, single-user dashboard for the fouling UDE. It uses the Python standard library HTTP server, vanilla JS, and vendored
Chart.js 4.4.1 + chartjs-plugin-zoom 2.0.1 + hammer.js 2.0.8 in `static/vendor/`, so there is no CDN and no build step.

## Run

From the repository root:

```bash
python -m web.make_mock        # (re)write canned responses in web/mock/ (needs only db/)
python -m web.server --mock    # MOCK mode: canned shapes, no model needed
python -m web.server           # REAL mode: db/ datasets, ml/ forecasts, cleaning_optimizer costs
# then open http://127.0.0.1:8050/   (--port to change; always binds to 127.0.0.1)
```

Deep links: `http://127.0.0.1:8050/?dataset=stress_no_clean&origin=3200#forecast` (tabs: `#monitor`, `#forecast`,
`#decision`, `#models`, `#correction`, `#about`).

Verified: every tab was rendered in headless Chrome against the real server (and Forecast/Decision against the mock
server) and the screenshots were inspected; the API is covered by `tests/test_web_contract.py`. Not verified: touch
gestures (pinch/pan), screen-reader output, and browsers other than Chrome.

Real mode needs the trained artifacts in `ml/artifacts/` (`python -m ml.train`, then `python -m ml.evaluate` for the Models tab).
Inference runs on CPU. The first Decision request for an origin re-simulates 90 intervals plus the UDE risk and takes a few seconds;
results are cached per (dataset, origin, model).

## API (contract)

| Endpoint | Purpose |
|---|---|
| `GET /api/meta` | datasets (with cleaning log), limits, costs, model summary, disclaimer |
| `GET /api/series?dataset&start&end&stride` | measured T, flow, U, dP; true Rf (evaluation only); BSI; cleaning events |
| `POST /api/forecast` | fan forecast (mean, 5-95%) of Rf, U, dP; time to limit; truth overlay |
| `POST /api/decision` | cost per day for T = t_min..t_max d (re-simulation), UDE first-cycle risk, recommendation, comparisons |
| `GET /api/metrics` | `ml/artifacts/metrics.json` |
| `GET /api/correction` | partial-dependence slices of the learned g |

Errors return `{"error": "..."}` with a 4xx/5xx status. Inputs are validated (types, ranges, at most 50 planned cleanings,
64 KB bodies), and static files are served only from `web/static/` (no path traversal). Contract tests: `tests/test_web_contract.py`
(the real-mode tests skip with a message if the artifacts are missing).

## Tabs

1. **Monitor**: dataset selector; Rf (true, labelled evaluation-only), U, dP, inlet T and flow over the year. Cleanings are marked
   `M` / `C`, limits are dashed lines, and the origin is a yellow line. Wheel zooms, drag pans, the time axis is shared across
   charts, and *Reset zoom* undoes it.
2. **Forecast**: origin (slider snaps to whole hours), horizon 24-720 h, model (UDE / ODE / persistence), future inputs
   (climatology / oracle), temperature offset ±5 °C, and planned cleanings (add/remove, or copy the dataset's schedule). It shows
   fan charts with the last 7 days of history, the truth overlay and limit lines, plus a readout of the predicted time to limit
   (median, 5-95%) against the actual crossing.
3. **Decision**: risk tolerance ε and interval range. Stacked cost per day (cleaning, downtime, pumping, thermal) against T, the UDE
   risk curve on the right axis, hatched infeasible (`///`) and above-ε (`\\\`) ranges, the recommended interval, and fixed-30 /
   condition-based / never-clean reference lines with a table. Labelled advisory: operator review and plant procedures remain
   mandatory.
4. **Models**: verdict text from `metrics.json`; error-vs-horizon charts per dataset (nominal test, two stress sets, mismatch);
   metric tables (best per horizon underlined, not colour-coded); 90% band coverage; time-to-limit errors against a 7-day
   linear trend.
5. **Correction**: partial-dependence plots of g (ensemble mean and range) with g = 1 and, for season, the generator's own
   nutrient factor for comparison, plus a note when g stays near 1.
6. **About**: problem, model, limitations, references.

Accessibility: tabs are an ARIA tablist (arrow keys, Home/End); every control has a label; the Okabe-Ito palette is used,
and series also differ by dash pattern, marker shape and legend text; shading uses hatch patterns; toasts are announced
via `aria-live`.
