"""API contract tests for web/server.py in mock and real mode."""

import json
from pathlib import Path
import sys
import threading
import unittest
import urllib.error
import urllib.request

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from web import server as S

NUM = (int, float)


def _request(base, path, body=None, raw=None, method=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method or ("POST" if data is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def _json(base, path, body=None):
    status, ctype, payload = _request(base, path, body)
    return status, json.loads(payload)


class ContractMixin:
    mock = True

    @classmethod
    def setUpClass(cls):
        cls.httpd = S.make_server(cls.mock, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        status, cls.meta = _json(cls.base, "/api/meta")
        assert status == 200, cls.meta

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def assert_band(self, b, n):
        for k in ("mean", "lo", "hi"):
            self.assertIsInstance(b[k], list)
            self.assertEqual(len(b[k]), n)
            self.assertTrue(all(isinstance(v, NUM) for v in b[k]))

    def assert_ttl(self, t):
        if t is not None:
            for k in ("median", "lo", "hi"):
                self.assertIn(k, t)
                self.assertTrue(t[k] is None or isinstance(t[k], NUM))

    # ---- endpoints ----
    def test_meta(self):
        m = self.meta
        for k in ("datasets", "limits", "costs", "model", "disclaimer"):
            self.assertIn(k, m)
        self.assertIn("SYNTHETIC", m["disclaimer"])
        ids = {d["id"] for d in m["datasets"]}
        self.assertEqual(ids, {"nominal", "stress_no_clean", "stress_long_interval", "mismatch_stress"})
        for d in m["datasets"]:
            for k in ("id", "label", "hours", "description", "cleaning_events"):
                self.assertIn(k, d)
            self.assertIsInstance(d["hours"], int)
            for e in d["cleaning_events"]:
                self.assertIsInstance(e["hour"], int)
                self.assertIn(e["type"], ("mechanical", "chemical"))
        for k in ("rf_crit", "dp_crit_kpa", "bsi_crit"):
            self.assertIsInstance(m["limits"][k], NUM)
        for k in ("ensemble_size", "k_dep", "k_rem", "device", "trained_on"):
            self.assertIn(k, m["model"])

    def test_series(self):
        status, s = _json(self.base, "/api/series?dataset=nominal&start=0&end=2000&stride=24")
        self.assertEqual(status, 200, s)
        keys = ("t", "T_in", "flow", "U", "dP_kpa", "Rf_true", "bsi")
        n = len(s["t"])
        self.assertGreater(n, 10)
        for k in keys:
            self.assertEqual(len(s[k]), n, k)
            self.assertTrue(all(isinstance(v, NUM) for v in s[k]), k)
        self.assertTrue(all(0 <= t <= 2000 for t in s["t"]))
        self.assertIsInstance(s["cleaning_events"], list)

    def test_forecast(self):
        body = {"dataset": "stress_no_clean", "origin_hour": 3000, "horizon_h": 168, "model": "ude",
                "planned_cleanings": [{"hour": 3100, "type": "mechanical"}], "temp_offset_c": 0, "exog": "climatology"}
        status, f = _json(self.base, "/api/forecast", body)
        self.assertEqual(status, 200, f)
        n = len(f["hours"])
        self.assertGreater(n, 0)
        for q in ("Rf", "U", "dP_kpa"):
            self.assert_band(f[q], n)
        self.assertIn("rf", f["time_to_limit"])
        self.assertIn("dp", f["time_to_limit"])
        self.assert_ttl(f["time_to_limit"]["rf"])
        self.assert_ttl(f["time_to_limit"]["dp"])
        self.assertIn("Rf", f["truth"])
        self.assertTrue(f["truth"]["Rf"] is None or len(f["truth"]["Rf"]) == n)
        self.assertIsInstance(f["model"], str)
        self.assertIsInstance(f["notes"], list)

    def test_decision(self):
        body = {"dataset": "nominal", "origin_hour": 3000, "t_min_days": 1, "t_max_days": 90, "epsilon": 0.05}
        status, d = _json(self.base, "/api/decision", body)
        self.assertEqual(status, 200, d)
        self.assertGreater(len(d["candidates"]), 0)
        for c in d["candidates"]:
            for k in ("interval_days", "total_cost", "cost_breakdown", "feasible", "risk_limit_cross"):
                self.assertIn(k, c)
            self.assertIsInstance(c["feasible"], bool)
            self.assertTrue(c["risk_limit_cross"] is None or 0.0 <= c["risk_limit_cross"] <= 1.0)
        self.assertTrue(d["recommended_interval_days"] is None or isinstance(d["recommended_interval_days"], int))
        for k in ("fixed_30", "condition_based", "never_clean"):
            self.assertIn(k, d["compared"])
        self.assertIsInstance(d["notes"], list)

    def test_rf_table(self):
        status, t = _json(self.base, "/api/rf_table?dataset=nominal&every=72")
        self.assertEqual(status, 200, t)
        for k in ("u_clean", "k_biofilm", "dp_clean_kpa", "rf_limit", "dp_limit_kpa"):
            self.assertIsInstance(t["constants"][k], NUM)
        self.assertIsInstance(t["status_rule"], str)
        self.assertGreater(len(t["rows"]), 50)
        num_keys = ("hour", "water_temp_c", "flow_kg_s", "u_measured", "dp_measured_kpa", "rf_from_u", "thickness_from_u_um",
                    "thickness_from_dp_um", "thickness_um", "rf", "pct_of_rf_limit", "cooling_lost_pct", "extra_pressure_pct", "rf_true")
        for r in t["rows"]:
            for k in num_keys:
                self.assertIsInstance(r[k], NUM, k)
            self.assertIn(r["status"], ("Good", "Watch", "Act soon", "Over limit"))
            self.assertAlmostEqual(r["rf"], r["thickness_um"] * 1e-6 / t["constants"]["k_biofilm"], delta=1e-9)
            self.assertIsInstance(r["cleaned_since_last_row"], list)
        status, _ = _json(self.base, "/api/rf_table?dataset=nominal&every=2")
        self.assertEqual(status, 400)

    def test_decision_cost_table_fields(self):
        status, d = _json(self.base, "/api/decision", {"dataset": "nominal", "origin_hour": 3000})
        self.assertEqual(status, 200, d)
        for c in d["candidates"]:
            b = c["cost_breakdown"]
            for k in ("cleaning_cost", "downtime_cost", "pumping_energy_cost", "thermal_penalty_cost", "downtime_hours",
                      "pump_energy_kwh", "thermal_shortfall_mwh", "biocide_kg", "biocide_limit_kg", "cleanings_mechanical", "cleanings_chemical"):
                self.assertIsInstance(b[k], NUM, k)
            self.assertIn("infeasible_reason", c)
            if not c["feasible"]:
                self.assertIsInstance(c["infeasible_reason"], str)

    def test_metrics_and_correction(self):
        status, m = _json(self.base, "/api/metrics")
        self.assertEqual(status, 200, m)
        self.assertIn("datasets", m)
        status, c = _json(self.base, "/api/correction")
        self.assertEqual(status, 200, c)
        self.assertIsInstance(c["inputs"], list)
        for s in c["slices"]:
            self.assertEqual(len(s["x"]), len(s["g"]))
            self.assertIsInstance(s["name"], str)

    # ---- errors and safety ----
    def test_errors_are_json(self):
        cases = [("/api/series?dataset=nope", None, 400), ("/api/series?dataset=nominal&stride=abc", None, 400),
                 ("/api/forecast", {"dataset": "nominal", "origin_hour": -5}, 400),
                 ("/api/forecast", {"dataset": "nominal", "origin_hour": 100, "model": "lstm"}, 400),
                 ("/api/decision", {"dataset": "x", "origin_hour": 100}, 400), ("/api/nothing", None, 404)]
        for path, body, code in cases:
            with self.subTest(path=path, body=body):
                status, ctype, payload = _request(self.base, path, body)
                self.assertEqual(status, code)
                self.assertTrue(ctype.startswith("application/json"))
                self.assertIn("error", json.loads(payload))
        status, _, payload = _request(self.base, "/api/forecast", raw=b"{not json")
        self.assertEqual(status, 400)

    def test_static_and_traversal(self):
        status, ctype, _ = _request(self.base, "/")
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/html"))
        for p in ("/../server.py", "/%2e%2e/server.py", "/vendor/../../server.py", "/..%2f..%2fconfig.py"):
            with self.subTest(path=p):
                status, _, payload = _request(self.base, p)
                self.assertEqual(status, 404)
                self.assertNotIn(b"import", payload)


class TestMockContract(ContractMixin, unittest.TestCase):
    mock = True


def _artifacts_present():
    art = APP_DIR / "ml" / "artifacts"
    return (art / "ode.pt").exists() and all((art / f"ude_seed{s}.pt").exists() for s in (11, 12, 13, 14, 15)) \
        and (art / "metrics.json").exists()


@unittest.skipUnless(_artifacts_present(), "ml artifacts missing: run python -m ml.train and python -m ml.evaluate")
class TestRealContract(ContractMixin, unittest.TestCase):
    mock = False


if __name__ == "__main__":
    unittest.main()
