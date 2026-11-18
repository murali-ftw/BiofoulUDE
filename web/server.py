"""
Local API + static server for the fouling dashboard (stdlib only, plus torch through ml/).

    python -m web.server            # real mode: db/ datasets, ml/ forecasts, cleaning_optimizer costs
    python -m web.server --mock     # canned responses from web/mock/*.json (same shapes)

Binds to 127.0.0.1 only. All data is SYNTHETIC; academic demonstration, not for plant use.
"""

import argparse
import json
import math
import mimetypes
import sys
import threading
import traceback
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
MOCK_DIR = WEB_DIR / "mock"
HOST = "127.0.0.1"
DISCLAIMER = ("SYNTHETIC DATA - academic demonstration only. Not validated on plant telemetry; "
              "must not be used for operating, maintaining or licensing a real plant.")
DATASET_INFO = {
    "nominal": ("Nominal year", "Nominal physics, cleaning every 720 h alternating mechanical/chemical (seed 42)."),
    "stress_no_clean": ("Stress: no cleaning", "Nominal physics, never cleaned; crosses Rf, dP and BSI limits (seed 43)."),
    "stress_long_interval": ("Stress: 90-day interval", "Nominal physics, cleaning every 90 days; crosses Rf and BSI (seed 44)."),
    "mismatch_stress": ("Mismatch stress (held-out)", "Deposition drift + scale deposit + summer bloom, 60-day cleaning; "
                                                      "mechanisms the UDE lacks (seed 45)."),
}
MAX_BODY_BYTES = 64 * 1024
MAX_SERIES_POINTS = 20000


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _bad(msg):
    return ApiError(HTTPStatus.BAD_REQUEST, msg)


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------

def _int(value, name, lo, hi):
    if isinstance(value, bool):
        raise _bad(f"{name} must be an integer")
    try:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        v = int(value)
    except (TypeError, ValueError):
        raise _bad(f"{name} must be an integer")
    if not lo <= v <= hi:
        raise _bad(f"{name} must be in {lo}..{hi}")
    return v


def _float(value, name, lo, hi):
    if isinstance(value, bool):
        raise _bad(f"{name} must be a number")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise _bad(f"{name} must be a number")
    if not math.isfinite(v) or not lo <= v <= hi:
        raise _bad(f"{name} must be in {lo}..{hi}")
    return v


def _choice(value, name, options):
    if value not in options:
        raise _bad(f"{name} must be one of {list(options)}")
    return value


def _cleanings(value, origin, horizon):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 50:
        raise _bad("planned_cleanings must be a list of at most 50 {hour, type} objects")
    out = []
    for ev in value:
        if not isinstance(ev, dict):
            raise _bad("each planned cleaning must be an object {hour, type}")
        hour = _int(ev.get("hour"), "planned_cleanings.hour", origin + 1, origin + horizon)
        out.append({"hour": hour, "type": _choice(ev.get("type"), "planned_cleanings.type", ("mechanical", "chemical"))})
    return sorted(out, key=lambda e: e["hour"])


def _clean_json(x):
    """Replace NaN/inf with None so the output is strict JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    return x


# -----------------------------------------------------------------------------
# Backends
# -----------------------------------------------------------------------------

class MockBackend:
    """Canned responses with the contract's shapes (web/mock/*.json)."""

    def __init__(self, mock_dir: Path = MOCK_DIR):
        self.dir = mock_dir
        self.meta_data = self._load("meta.json")
        self.ids = {d["id"]: d for d in self.meta_data["datasets"]}

    def _load(self, name):
        path = self.dir / name
        if not path.exists():
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, f"mock file {name} missing (run python -m web.make_mock)")
        return json.loads(path.read_text())

    def _dataset(self, value):
        return self.ids[_choice(value, "dataset", tuple(self.ids))]

    def meta(self):
        return self.meta_data

    def series(self, q):
        ds = self._dataset(q.get("dataset"))
        start = _int(q.get("start", 0), "start", 0, ds["hours"] - 1)
        end = _int(q.get("end", ds["hours"] - 1), "end", start, ds["hours"] - 1)
        stride = _int(q.get("stride", 1), "stride", 1, 168)
        full = self._load(f"series_{ds['id']}.json")
        keep = [i for i, t in enumerate(full["t"]) if start <= t <= end and (t - start) % stride == 0]
        if not keep:  # mock series is pre-strided; fall back to every stored point in range
            keep = [i for i, t in enumerate(full["t"]) if start <= t <= end]
        out = {k: [full[k][i] for i in keep] for k in ("t", "T_in", "flow", "U", "dP_kpa", "Rf_true", "bsi")}
        out["cleaning_events"] = [e for e in full["cleaning_events"] if start <= e["hour"] <= end]
        return out

    def forecast(self, body):
        ds = self._dataset(body.get("dataset"))
        _int(body.get("origin_hour"), "origin_hour", 11, ds["hours"] - 1)
        _int(body.get("horizon_h", 168), "horizon_h", 1, 720)
        _choice(body.get("model", "ude"), "model", ("ude", "ode", "persistence"))
        return self._load("forecast.json")

    def decision(self, body):
        ds = self._dataset(body.get("dataset"))
        _int(body.get("origin_hour"), "origin_hour", 11, ds["hours"] - 1)
        return self._load("decision.json")

    def metrics(self):
        return self._load("metrics.json")

    def correction(self):
        return self._load("correction.json")


class RealBackend:
    """Datasets from db/, forecasts from ml/forecast.py, costs from cleaning_optimizer.simulate_policy."""

    def __init__(self):
        import ml  # noqa: F401  (MPS fallback env before torch)
        import config
        import cleaning_optimizer as co
        from ml import data as D
        from ml import forecast as F
        sys.path.insert(0, str(ROOT / "db"))
        import generate as dbgen
        self.config, self.co, self.D, self.F, self.dbgen = config, co, D, F, dbgen
        self.lock = threading.Lock()
        self.cache = {}
        self.files = {"nominal": "biofouling_synthetic_dataset.csv", "stress_no_clean": "stress_no_clean.csv",
                      "stress_long_interval": "stress_long_interval.csv", "mismatch_stress": "mismatch_stress.csv"}

    # -- helpers --
    def _df(self, ds_id):
        _choice(ds_id, "dataset", tuple(self.files))
        return self.D.load(ds_id)

    def _events(self, df, lo=None, hi=None):
        ev = df[df["cleaning_event"] == 1]
        out = [{"hour": int(h), "type": "mechanical" if t == "Mechanical_Balls" else "chemical"}
               for h, t in zip(ev["time_hours"], ev["cleaning_type"])]
        return [e for e in out if (lo is None or e["hour"] >= lo) and (hi is None or e["hour"] <= hi)]

    def _cached(self, key, fn):
        if key in self.cache:
            return self.cache[key]
        with self.lock:
            if key not in self.cache:
                self.cache[key] = fn()
        return self.cache[key]

    # -- endpoints --
    def meta(self):
        import dataclasses
        from ml import train as T
        datasets = []
        for ds_id, fname in self.files.items():
            df = self.D.load(ds_id)
            label, desc = DATASET_INFO[ds_id]
            datasets.append({"id": ds_id, "label": label, "hours": int(len(df)), "description": desc,
                             "cleaning_events": self._events(df)})
        model = {"ensemble_size": 0, "k_dep": None, "k_rem": None, "device": "cpu (inference)", "trained_on": None}
        if T.LOG_PATH.exists():
            log = json.loads(T.LOG_PATH.read_text())
            members = log.get("ude_members", [])
            model.update({
                "ensemble_size": len(members),
                "k_dep": sum(m["k_dep"] for m in members) / len(members) if members else None,
                "k_rem": sum(m["k_rem"] for m in members) / len(members) if members else None,
                "ode_k_dep": log["ode"]["k_dep"], "ode_k_rem": log["ode"]["k_rem"],
                "config_k_dep": self.config.K_DEPOSITION, "config_k_rem": self.config.K_REMOVAL,
                "device": f"trained on {log.get('device')} ({log.get('dtype')}); inference on cpu",
                "trained_on": "nominal train cycles 0-7 (hours %d-%d)" % tuple(log["splits"]["train"]),
            })
        c = self.config
        return {
            "datasets": datasets,
            "limits": {"rf_crit": c.CRITICAL_RF_M2K_W, "dp_crit_kpa": c.CRITICAL_DELTA_P_KPA, "bsi_crit": c.CRITICAL_BSI_SCORE},
            "costs": dataclasses.asdict(self.co.DEFAULT_COSTS) | {"biocide_permit_kg_per_year": c.ANNUAL_BIOCIDE_PERMIT_LIMIT_KG},
            "model": model,
            "disclaimer": DISCLAIMER,
        }

    def series(self, q):
        ds_id = q.get("dataset")
        df = self._df(ds_id)
        n = len(df)
        start = _int(q.get("start", 0), "start", 0, n - 1)
        end = _int(q.get("end", n - 1), "end", start, n - 1)
        stride = _int(q.get("stride", 1), "stride", 1, 168)
        if (end - start) // stride + 1 > MAX_SERIES_POINTS:
            raise _bad(f"too many points; increase stride (max {MAX_SERIES_POINTS})")
        sl = df.iloc[start:end + 1:stride]
        col = lambda c: [float(v) for v in sl[c]]
        return {"t": [int(v) for v in sl["time_hours"]], "T_in": col("T_cool_in_meas"), "flow": col("mass_flow_cool_meas"),
                "U": col("U_overall_meas"), "dP_kpa": col("delta_P_meas_kPa"), "Rf_true": col("Rf_fouling_resistance"),
                "bsi": col("bsi_score"), "cleaning_events": self._events(df, start, end)}

    def forecast(self, body):
        ds_id = body.get("dataset")
        df = self._df(ds_id)
        origin = _int(body.get("origin_hour"), "origin_hour", self.D.BURN_IN_HOURS - 1, len(df) - 1)
        horizon = _int(body.get("horizon_h", 168), "horizon_h", 1, 720)
        model = _choice(body.get("model", "ude"), "model", ("ude", "ode", "persistence"))
        exog = _choice(body.get("exog", "climatology"), "exog", ("climatology", "oracle"))
        offset = _float(body.get("temp_offset_c", 0.0), "temp_offset_c", -10.0, 10.0)
        planned = _cleanings(body.get("planned_cleanings"), origin, horizon)
        key = ("forecast", ds_id, origin, horizon, model, exog, offset, json.dumps(planned))
        return self._cached(key, lambda: self.F.forecast_at(df, origin, horizon, model, planned, exog, offset))

    def decision(self, body):
        ds_id = body.get("dataset")
        df = self._df(ds_id)
        origin = _int(body.get("origin_hour"), "origin_hour", self.D.BURN_IN_HOURS - 1, len(df) - 1)
        t_min = _int(body.get("t_min_days", 1), "t_min_days", 1, 90)
        t_max = _int(body.get("t_max_days", 90), "t_max_days", t_min, 90)
        eps = _float(body.get("epsilon", 0.05), "epsilon", 0.0, 1.0)
        base = self._cached(("decision", ds_id, origin, t_min, t_max), lambda: self._decision(df, ds_id, origin, t_min, t_max))
        return self._apply_epsilon(base, eps)

    def _decision(self, df, ds_id, origin, t_min, t_max):
        co, F = self.co, self.F
        offset, _, _ = self.dbgen.DATASETS[self.files[ds_id]]
        seed = self.config.DEFAULT_RANDOM_SEED + offset
        scenario = self.dbgen.MISMATCH_SCENARIO if ds_id == "mismatch_stress" else co.NOMINAL_SCENARIO
        st = F.state_at_origin(df, origin)
        n_done = int(((df["cleaning_event"] == 1) & (df["time_hours"] <= origin)).sum())

        def evaluate(policy, days):
            state = co.FoulingState(delta_f_m=st["delta"], cum_clean_hours=st["cum"], time_since_last_clean=0.0, n_cleanings=n_done)
            r = co.evaluate_policy(policy, days * 24, seed, scenario, state, origin + 1)
            keys = ("cleaning_cost", "downtime_cost", "pumping_energy_cost", "thermal_penalty_cost")
            return {"window_days": days, "total_cost": r["total_cost"], "cost_per_day": r["total_cost"] / days,
                    "cost_breakdown": {k: r[k] for k in keys} | {"cleanings_mechanical": r["cleanings_mechanical"],
                                                                "cleanings_chemical": r["cleanings_chemical"],
                                                                "biocide_kg": r["biocide_kg"]},
                    "feasible": r["is_feasible"], "hours_above_limit": r["hours_above_limit"],
                    "peak_rf": r["peak_rf"], "peak_dp_kpa": r["peak_dp_kpa"]}

        try:
            risk = {r["interval_days"]: r for r in F.risk_by_interval(df, origin, t_min, t_max, "ude")["intervals"]}
            risk_note = "risk_limit_cross = UDE-ensemble probability that Rf or dP crosses its limit before the first cleaning"
        except FileNotFoundError:
            risk, risk_note = {}, "UDE artifacts missing: risk not available (train with python -m ml.train)"
        candidates = []
        for days in range(t_min, t_max + 1):
            c = evaluate(co.fixed_interval_policy(days), max(60, days))
            c["interval_days"] = days
            c["risk_limit_cross"] = risk.get(days, {}).get("p_any")
            candidates.append(c)
        window = 90
        compared = {"fixed_30": evaluate(co.fixed_interval_policy(30), window),
                    "condition_based": evaluate(co.condition_based_policy(0.8, 0.8), window),
                    "never_clean": evaluate(co.never_clean_policy(), window)}
        notes = [
            "Costs: forward re-simulation (cleaning_optimizer.simulate_policy) of the dataset's generator from the "
            "state estimated at the origin (no truth), priced with the shared cost function.",
            "Each candidate is simulated over max(60, T) days; candidates are ranked by cost per day.",
            "compared: fixed 30 d, condition-based (0.8 x Rf/dP limits, true-state trigger) and never clean, over 90 days.",
            risk_note,
            "First-cycle risk rule is a prototype. Advisory only; operator review and plant procedures remain mandatory.",
        ]
        return {"origin_hour": origin, "state_estimate_um": st["delta"] * 1e6, "candidates": candidates,
                "compared": compared, "notes": notes}

    @staticmethod
    def _apply_epsilon(base, eps):
        cands = []
        for c in base["candidates"]:
            r = c["risk_limit_cross"]
            c = dict(c, risk_ok=(r is None or r <= eps))
            cands.append(c)
        ok = [c for c in cands if c["feasible"] and c["risk_ok"]]
        rec = min(ok, key=lambda c: c["cost_per_day"])["interval_days"] if ok else None
        notes = list(base["notes"])
        if rec is None:
            notes.insert(0, f"No candidate interval is both feasible and within risk epsilon = {eps:g}: clean as soon as practicable "
                            "and review the situation.")
        return {"candidates": cands, "recommended_interval_days": rec, "compared": base["compared"], "epsilon": eps,
                "origin_hour": base["origin_hour"], "state_estimate_um": base["state_estimate_um"], "notes": notes}

    def metrics(self):
        from ml.evaluate import METRICS_PATH
        if not METRICS_PATH.exists():
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "metrics.json missing (run python -m ml.evaluate)")
        return json.loads(METRICS_PATH.read_text())

    def correction(self):
        return self._cached(("correction",), self.F.correction_slices)


# -----------------------------------------------------------------------------
# HTTP layer
# -----------------------------------------------------------------------------

def make_handler(backend):
    static_root = STATIC_DIR.resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "FoulingDemo/1.0"

        def log_message(self, fmt, *args):  # quieter: one line per request on stderr
            sys.stderr.write("%s %s\n" % (self.command, self.path))

        def _send(self, status, payload, content_type="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(_clean_json(payload), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store" if content_type == "application/json" else "max-age=60")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._send(status, {"error": message})

        def _dispatch(self, fn):
            try:
                self._send(HTTPStatus.OK, fn())
            except ApiError as e:
                self._error(e.status, e.message)
            except FileNotFoundError as e:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(e))
            except ValueError as e:
                self._error(HTTPStatus.BAD_REQUEST, str(e))
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            routes = {"/api/meta": backend.meta, "/api/series": lambda: backend.series(q),
                      "/api/metrics": backend.metrics, "/api/correction": backend.correction}
            if url.path in routes:
                return self._dispatch(routes[url.path])
            if url.path.startswith("/api/"):
                return self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            self._static(url.path)

        def do_POST(self):
            url = urlparse(self.path)
            routes = {"/api/forecast": backend.forecast, "/api/decision": backend.decision}
            if url.path not in routes:
                return self._error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._error(HTTPStatus.BAD_REQUEST, "bad Content-Length")
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._error(HTTPStatus.BAD_REQUEST, f"JSON body required (max {MAX_BODY_BYTES} bytes)")
            try:
                body = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                return self._error(HTTPStatus.BAD_REQUEST, "body is not valid JSON")
            if not isinstance(body, dict):
                return self._error(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
            self._dispatch(lambda: routes[url.path](body))

        def _static(self, path):
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (static_root / rel).resolve()
            if not target.is_relative_to(static_root) or not target.is_file():
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript",):
                ctype += "; charset=utf-8"
            self._send(HTTPStatus.OK, target.read_bytes(), ctype)

    return Handler


def make_server(mock: bool, port: int = 8050) -> ThreadingHTTPServer:
    backend = MockBackend() if mock else RealBackend()
    httpd = ThreadingHTTPServer((HOST, port), make_handler(backend))
    httpd.daemon_threads = True
    return httpd


def main():
    ap = argparse.ArgumentParser(description="Fouling dashboard server (binds to 127.0.0.1)")
    ap.add_argument("--mock", action="store_true", help="serve canned responses from web/mock/")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    httpd = make_server(args.mock, args.port)
    print(f"{'MOCK' if args.mock else 'REAL'} mode: http://{HOST}:{httpd.server_address[1]}/  ({DISCLAIMER})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
