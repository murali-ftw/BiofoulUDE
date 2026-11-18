/* Biofouling UDE dashboard. Vanilla JS + vendored Chart.js 4.4.1 (+ zoom plugin 2.0.1).
   All data is SYNTHETIC; academic demonstration, not for plant use. */
"use strict";

// Okabe-Ito palette (colour-blind safe); series are also told apart by dash pattern and legend text.
const C = { blue: "#56B4E9", sky: "#0072B2", orange: "#E69F00", green: "#009E73", verm: "#D55E00",
            purple: "#CC79A7", yellow: "#F0E442", grey: "#9aa3ad", text: "#e6e8eb", grid: "#2c333d" };
const RF_SCALE = 1e4; // Rf shown in 1e-4 m²K/W

const S = { meta: null, dataset: "nominal", origin: 3000, series: {}, fullSeries: {}, charts: {}, cleanings: [],
            metrics: null, correction: null, loaded: {} };

// ---------------------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
};
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const fmtMoney = (v) => (v === null || v === undefined ? "–" : "$" + Math.round(v).toLocaleString());
const fmtSci = (v) => (v === null || v === undefined ? "–" : Number(v).toExponential(2));
const hToD = (h) => (h / 24).toFixed(1);

function toast(msg, kind = "error") {
  const t = el("div", { class: "toast " + kind, text: msg });
  $("toasts").append(t);
  setTimeout(() => t.remove(), kind === "error" ? 8000 : 4000);
}

async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opts);
  let data;
  try { data = await r.json(); } catch { throw new Error(`${path}: HTTP ${r.status} (non-JSON response)`); }
  if (!r.ok) throw new Error(data.error || `${path}: HTTP ${r.status}`);
  return data;
}

function badges(...kinds) {
  const map = { SYNTHETIC: "badge-synthetic", ML_FORECAST: "badge-ml", PHYSICS: "badge-physics" };
  return kinds.map((k) => el("span", { class: "badge " + map[k], title: k === "SYNTHETIC" ? "synthetic data" : k, text: k }));
}

function card(container, title, kinds, wide = false) {
  const canvas = el("canvas", { role: "img", "aria-label": title });
  const wrap = el("div", { class: "canvas-wrap" }, canvas);
  const h = el("h3", {}, title, ...badges(...kinds));
  const c = el("div", { class: "chart-card" + (wide ? " wide" : "") }, h, wrap);
  container.append(c);
  const loading = el("div", { class: "loading", text: "Loading…" });
  return { card: c, canvas, setLoading: (on) => (on ? wrap.append(loading) : loading.remove()) };
}

function destroy(key) { if (S.charts[key]) { S.charts[key].destroy(); delete S.charts[key]; } }

// ---------------------------------------------------------------------------- Chart.js setup
if (window.ChartZoom) Chart.register(window.ChartZoom);
Chart.defaults.color = C.text;
Chart.defaults.borderColor = C.grid;
Chart.defaults.font.family = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
Chart.defaults.animation = false;
Chart.defaults.plugins.legend.labels.boxWidth = 18;
// helper series that only bound a filled band are not shown in legends
Chart.defaults.plugins.legend.labels.filter = (item) => !/^(5% band|ensemble min)$/.test(item.text);
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.line.borderWidth = 1.6;

// Vertical markers: cleaning events (letter M / C) and the origin.
const markersPlugin = {
  id: "markers",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || (!opts.events && opts.origin === undefined)) return;
    const { ctx, chartArea: a, scales: { x } } = chart;
    ctx.save();
    ctx.font = "bold 10px system-ui";
    ctx.textAlign = "center";
    for (const e of opts.events || []) {
      const px = x.getPixelForValue(e.hour);
      if (px < a.left || px > a.right) continue;
      ctx.strokeStyle = C.grey; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, a.top + 12); ctx.lineTo(px, a.bottom); ctx.stroke();
      ctx.fillStyle = C.text;
      ctx.fillText(e.type === "mechanical" ? "M" : "C", px, a.top + 10);
    }
    if (opts.origin !== undefined && opts.origin !== null) {
      const px = x.getPixelForValue(opts.origin);
      if (px >= a.left && px <= a.right) {
        ctx.strokeStyle = C.yellow; ctx.setLineDash([]); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
        ctx.fillStyle = C.yellow; ctx.textAlign = "left";
        const label = opts.label || "origin";
        ctx.fillText(label, Math.min(px + 3, a.right - ctx.measureText(label).width - 2), opts.label ? a.top + 12 : a.bottom - 4);
      }
    }
    ctx.restore();
  },
};

// Shaded x-ranges with hatch patterns (infeasible / high-risk), so shading is not colour-only.
function hatch(color, dir) {
  const c = document.createElement("canvas"); c.width = c.height = 8;
  const g = c.getContext("2d");
  g.strokeStyle = color; g.lineWidth = 1.5;
  g.beginPath();
  if (dir > 0) { g.moveTo(0, 8); g.lineTo(8, 0); } else { g.moveTo(0, 0); g.lineTo(8, 8); }
  g.stroke();
  return g.createPattern(c, "repeat");
}
const shadePlugin = {
  id: "shade",
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.ranges) return;
    const { ctx, chartArea: a, scales: { x } } = chart;
    ctx.save();
    for (const r of opts.ranges) {
      const half = (x.getPixelForValue(1) - x.getPixelForValue(0)) / 2;
      const x0 = x.getPixelForValue(r.from) - half, x1 = x.getPixelForValue(r.to) + half;
      ctx.fillStyle = r.pattern;
      ctx.fillRect(Math.max(a.left, x0), a.top, Math.min(a.right, x1) - Math.max(a.left, x0), a.bottom - a.top);
    }
    ctx.restore();
  },
};
Chart.register(markersPlugin, shadePlugin);

const limitDataset = (label, xs, value) => ({
  label, data: xs.map((x) => ({ x, y: value })), borderColor: C.verm, borderDash: [8, 4], borderWidth: 1.5, pointRadius: 0, order: 0,
});

function lineChart(canvas, datasets, { xTitle = "hour", yTitle = "", events = [], origin = null, zoom = false, yMin } = {}) {
  return new Chart(canvas, {
    type: "line",
    data: { datasets },
    options: {
      parsing: true, normalized: true, maintainAspectRatio: false, interaction: { mode: "nearest", intersect: false, axis: "x" },
      scales: { x: { type: "linear", title: { display: true, text: xTitle } }, y: { title: { display: true, text: yTitle }, min: yMin } },
      plugins: {
        markers: { events, origin },
        legend: { position: "bottom" },
        tooltip: { callbacks: { title: (items) => (items.length ? `hour ${Math.round(items[0].parsed.x)} (day ${hToD(items[0].parsed.x)})` : "") } },
        zoom: zoom ? {
          zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "x", onZoomComplete: syncZoom },
          pan: { enabled: true, mode: "x", onPanComplete: syncZoom },
        } : undefined,
      },
    },
  });
}

// ---------------------------------------------------------------------------- tabs
const TABS = ["monitor", "forecast", "decision", "models", "correction", "about"];
function selectTab(name, focus = false) {
  for (const t of TABS) {
    const on = t === name;
    const tab = $("tab-" + t);
    tab.setAttribute("aria-selected", on ? "true" : "false");
    tab.tabIndex = on ? 0 : -1;
    $("panel-" + t).hidden = !on;
    if (on && focus) tab.focus();
  }
  location.hash = name;
  onTabShown(name);
}
function setupTabs() {
  for (const t of TABS) $("tab-" + t).addEventListener("click", () => selectTab(t));
  document.querySelector(".tabs").addEventListener("keydown", (ev) => {
    const i = TABS.indexOf(document.activeElement.id.replace("tab-", ""));
    if (i < 0) return;
    let j = null;
    if (ev.key === "ArrowRight") j = (i + 1) % TABS.length;
    if (ev.key === "ArrowLeft") j = (i - 1 + TABS.length) % TABS.length;
    if (ev.key === "Home") j = 0;
    if (ev.key === "End") j = TABS.length - 1;
    if (j !== null) { ev.preventDefault(); selectTab(TABS[j], true); }
  });
}
function onTabShown(name) {
  if (name === "monitor") renderMonitor();
  if (name === "models" && !S.loaded.models) loadModels();
  if (name === "correction" && !S.loaded.correction) loadCorrection();
  if (name === "forecast" && !S.loaded.forecast) { S.loaded.forecast = true; runForecast(); }
  if (name === "decision" && !S.loaded.decision) { S.loaded.decision = true; runDecision(); }
}

// ---------------------------------------------------------------------------- shared controls
function datasetMeta() { return S.meta.datasets.find((d) => d.id === S.dataset); }

function setOrigin(h, { fromSlider = false } = {}) {
  const ds = datasetMeta();
  const v = Math.max(11, Math.min(ds.hours - 1, Math.round(Number(h) || 0)));
  S.origin = v;
  $("origin").value = v; $("origin-num").value = v;
  $("origin-out").textContent = `${v} h (day ${hToD(v)})`;
  for (const k of Object.keys(S.charts)) {
    if (k.startsWith("mon-")) { S.charts[k].options.plugins.markers.origin = v; S.charts[k].update("none"); }
  }
  if (!fromSlider) return;
}

function setupShared() {
  const sel = $("dataset");
  for (const d of S.meta.datasets) sel.append(el("option", { value: d.id, text: d.label }));
  sel.addEventListener("change", () => {
    S.dataset = sel.value;
    const ds = datasetMeta();
    $("dataset-desc").textContent = ds.description;
    for (const id of ["origin", "origin-num"]) $(id).max = ds.hours - 1;
    setOrigin(Math.min(S.origin, ds.hours - 1));
    S.cleanings = []; renderCleanings();
    S.loaded.forecast = S.loaded.decision = false;
    const active = TABS.find((t) => !$("panel-" + t).hidden);
    onTabShown(active);
  });
  $("dataset-desc").textContent = datasetMeta().description;
  $("origin").addEventListener("input", (e) => setOrigin(e.target.value, { fromSlider: true }));
  $("origin").addEventListener("change", () => { S.loaded.forecast = S.loaded.decision = false; });
  $("origin-num").addEventListener("change", (e) => { setOrigin(e.target.value); S.loaded.forecast = S.loaded.decision = false; });
  setOrigin(S.origin);
}

// ---------------------------------------------------------------------------- 1 Monitor
async function getSeries(id, stride) {
  const key = id + ":" + stride;
  if (!S.series[key]) {
    const ds = S.meta.datasets.find((d) => d.id === id);
    S.series[key] = await api(`/api/series?dataset=${id}&start=0&end=${ds.hours - 1}&stride=${stride}`);
  }
  return S.series[key];
}

let monitorDrawn = null;
async function renderMonitor() {
  if (monitorDrawn === S.dataset) return;
  const box = $("monitor-charts");
  box.replaceChildren();
  for (const k of Object.keys(S.charts)) if (k.startsWith("mon-")) destroy(k);
  const specs = [
    ["mon-rf", "Fouling resistance Rf (true, evaluation only)", ["SYNTHETIC", "PHYSICS"], "Rf (1e-4 m²K/W)", "Rf_true", RF_SCALE, C.orange, S.meta.limits.rf_crit * RF_SCALE, "Rf limit"],
    ["mon-u", "Overall heat-transfer coefficient U (measured)", ["SYNTHETIC"], "U (W/m²K)", "U", 1, C.blue],
    ["mon-dp", "Tube-side pressure drop dP (measured)", ["SYNTHETIC"], "dP (kPa)", "dP_kpa", 1, C.green, S.meta.limits.dp_crit_kpa, "dP limit"],
    ["mon-t", "Cooling-water inlet temperature (measured)", ["SYNTHETIC"], "T (°C)", "T_in", 1, C.purple],
    ["mon-flow", "Cooling-water mass flow (measured)", ["SYNTHETIC"], "flow (kg/s)", "flow", 1, C.sky],
  ];
  const cards = specs.map((s) => card(box, s[1], s[2]));
  cards.forEach((c) => c.setLoading(true));
  try {
    const s = await getSeries(S.dataset, 3);
    specs.forEach((spec, i) => {
      const [key, title, , yTitle, field, scale, color, limit, limitLabel] = spec;
      const data = s.t.map((t, j) => ({ x: t, y: s[field][j] * scale }));
      const ds = [{ label: title.split(" (")[0], data, borderColor: color, borderWidth: 1 }];
      if (limit !== undefined) ds.push(limitDataset(limitLabel, [s.t[0], s.t[s.t.length - 1]], limit));
      cards[i].setLoading(false);
      S.charts[key] = lineChart(cards[i].canvas, ds, { yTitle, events: s.cleaning_events, origin: S.origin, zoom: true });
    });
    monitorDrawn = S.dataset;
  } catch (e) { cards.forEach((c) => c.setLoading(false)); toast("Monitor: " + e.message); }
}

function syncZoom({ chart }) {
  const { min, max } = chart.scales.x;
  for (const [k, ch] of Object.entries(S.charts)) {
    if (k.startsWith("mon-") && ch !== chart) { ch.options.scales.x.min = min; ch.options.scales.x.max = max; ch.update("none"); }
  }
}
function resetZoom() {
  for (const [k, ch] of Object.entries(S.charts)) {
    if (k.startsWith("mon-")) { ch.options.scales.x.min = undefined; ch.options.scales.x.max = undefined; if (ch.resetZoom) ch.resetZoom("none"); ch.update("none"); }
  }
}

// ---------------------------------------------------------------------------- 2 Forecast
function renderCleanings() {
  const ul = $("clean-list");
  ul.replaceChildren();
  S.cleanings.sort((a, b) => a.hour - b.hour);
  S.cleanings.forEach((c, i) => {
    const rm = el("button", { type: "button", "aria-label": `Remove ${c.type} cleaning at hour ${c.hour}`, text: "×" });
    rm.addEventListener("click", () => { S.cleanings.splice(i, 1); renderCleanings(); });
    ul.append(el("li", {}, `${c.type === "mechanical" ? "M" : "C"} · hour ${c.hour}`, rm));
  });
  if (!S.cleanings.length) ul.append(el("li", { text: "none (no cleaning in the horizon)" }));
}

function setupForecast() {
  const bind = (id, out, f) => { const u = () => ($(out).textContent = f($(id).value)); $(id).addEventListener("input", u); u(); };
  bind("horizon", "horizon-out", (v) => `${v} h (${hToD(v)} d)`);
  bind("toff", "toff-out", (v) => `${Number(v) > 0 ? "+" : ""}${v} °C`);
  $("clean-add").addEventListener("click", () => {
    const h = Math.round(Number($("clean-hour").value));
    const H = Number($("horizon").value);
    if (!(h > S.origin && h <= S.origin + H)) return toast(`Cleaning hour must be in ${S.origin + 1}..${S.origin + H}`);
    if (S.cleanings.some((c) => c.hour === h)) return toast("There is already a cleaning at that hour");
    S.cleanings.push({ hour: h, type: $("clean-type").value });
    renderCleanings();
  });
  $("clean-dataset").addEventListener("click", () => {
    const H = Number($("horizon").value);
    S.cleanings = datasetMeta().cleaning_events.filter((e) => e.hour > S.origin && e.hour <= S.origin + H).map((e) => ({ ...e }));
    renderCleanings();
    if (!S.cleanings.length) toast("The dataset has no cleaning inside this horizon", "info");
  });
  $("clean-clear").addEventListener("click", () => { S.cleanings = []; renderCleanings(); });
  $("run-forecast").addEventListener("click", runForecast);
  renderCleanings();
}

async function fullSeries(id) {
  if (!S.fullSeries[id]) S.fullSeries[id] = await getSeries(id, 1);
  return S.fullSeries[id];
}

function actualCrossing(s, field, limit, after) {
  for (let i = 0; i < s.t.length; i++) if (s.t[i] > after && s[field][i] > limit) return s.t[i];
  return null;
}

async function runForecast() {
  const btn = $("run-forecast");
  const H = Number($("horizon").value);
  const body = { dataset: S.dataset, origin_hour: S.origin, horizon_h: H, model: $("model").value,
                 planned_cleanings: S.cleanings.filter((c) => c.hour > S.origin && c.hour <= S.origin + H),
                 temp_offset_c: Number($("toff").value), exog: $("exog").value };
  const box = $("forecast-charts");
  box.replaceChildren();
  ["fc-rf", "fc-u", "fc-dp"].forEach(destroy);
  const kinds = body.model === "persistence" ? ["SYNTHETIC"] : ["SYNTHETIC", "ML_FORECAST", "PHYSICS"];
  const cards = [card(box, "Rf forecast", kinds), card(box, "U forecast", kinds), card(box, "dP forecast", kinds)];
  cards.forEach((c) => c.setLoading(true));
  btn.disabled = true;
  $("ttl-readout").textContent = ""; $("forecast-notes").replaceChildren();
  try {
    const [f, s] = await Promise.all([api("/api/forecast", body), fullSeries(S.dataset)]);
    const lo = Math.max(0, S.origin - 167);
    const hist = s.t.map((t, i) => [t, i]).filter(([t]) => t >= lo && t <= S.origin);
    const L = S.meta.limits;
    const label = { ude: "UDE ensemble", ode: "plain ODE", persistence: "persistence" }[f.model] || f.model;
    const fan = (q, scale, color, histField, histLabel, limit, limitLabel, truth) => {
      const xs = f.hours;
      const ds = [
        { label: "5% band", data: xs.map((x, i) => ({ x, y: f[q].lo[i] * scale })), borderColor: "transparent", pointRadius: 0 },
        { label: "90% band (5-95%)", data: xs.map((x, i) => ({ x, y: f[q].hi[i] * scale })), borderColor: "transparent",
          backgroundColor: color + "40", fill: "-1", pointRadius: 0 },
        { label: `${label} mean`, data: xs.map((x, i) => ({ x, y: f[q].mean[i] * scale })), borderColor: color, borderWidth: 2 },
        { label: histLabel, data: hist.map(([t, i]) => ({ x: t, y: s[histField][i] * scale })), borderColor: C.grey, borderWidth: 1 },
      ];
      if (truth) ds.push({ label: "true Rf (synthetic, evaluation only)", data: xs.map((x, i) => ({ x, y: truth[i] === null ? null : truth[i] * scale })),
                           borderColor: C.text, borderDash: [4, 3], borderWidth: 1.4 });
      if (limit !== undefined) ds.push(limitDataset(limitLabel, [hist.length ? hist[0][0] : xs[0], xs[xs.length - 1]], limit));
      return ds;
    };
    const evs = datasetMeta().cleaning_events.filter((e) => e.hour >= lo && e.hour <= S.origin);
    const planned = body.planned_cleanings;
    const opts = (yTitle) => ({ yTitle, events: [...evs, ...planned], origin: S.origin });
    cards.forEach((c) => c.setLoading(false));
    S.charts["fc-rf"] = lineChart(cards[0].canvas, fan("Rf", RF_SCALE, C.orange, "Rf_true", "true Rf before origin (evaluation only)", L.rf_crit * RF_SCALE, "Rf limit", f.truth.Rf), opts("Rf (1e-4 m²K/W)"));
    S.charts["fc-u"] = lineChart(cards[1].canvas, fan("U", 1, C.blue, "U", "measured U before origin"), opts("U (W/m²K)"));
    S.charts["fc-dp"] = lineChart(cards[2].canvas, fan("dP_kpa", 1, C.green, "dP_kpa", "measured dP before origin", L.dp_crit_kpa, "dP limit"), opts("dP (kPa)"));

    const ttl = (t, actual, name) => {
      const act = actual === null ? "not crossed in the dataset after the origin" : `actual crossing hour ${actual} (in ${hToD(actual - S.origin)} d)`;
      if (!t || t.median === null) {
        const frac = t ? ` (${Math.round(100 * t.fraction_crossing)}% of trajectories cross)` : "";
        return `${name}: not predicted to cross within ${H} h${frac}; ${act}.`;
      }
      const d = (v) => (v === null ? "beyond horizon" : `${hToD(v - S.origin)} d`);
      return `${name}: predicted crossing in ${d(t.median)} (5-95%: ${d(t.lo)} to ${d(t.hi)}); ${act}.`;
    };
    const aRf = actualCrossing(s, "Rf_true", L.rf_crit, S.origin);
    const aDp = actualCrossing(s, "dP_kpa", L.dp_crit_kpa, S.origin);
    $("ttl-readout").replaceChildren(el("strong", { text: "Predicted time to limit" }), el("br"),
      ttl(f.time_to_limit.rf, aRf, "Rf"), el("br"), ttl(f.time_to_limit.dp, aDp, "dP (actual from measured dP)"));
    $("forecast-notes").append(el("ul", {}, ...f.notes.map((n) => el("li", { text: n }))));
  } catch (e) { cards.forEach((c) => c.setLoading(false)); toast("Forecast: " + e.message); }
  finally { btn.disabled = false; }
}

// ---------------------------------------------------------------------------- 3 Decision
function setupDecision() {
  const u = () => ($("eps-out").textContent = Number($("eps").value).toFixed(2));
  $("eps").addEventListener("input", u); u();
  $("eps").addEventListener("change", runDecision);
  $("run-decision").addEventListener("click", runDecision);
}

async function runDecision() {
  const btn = $("run-decision");
  const tmin = Number($("tmin").value), tmax = Number($("tmax").value);
  if (!(tmin >= 1 && tmax <= 90 && tmin <= tmax)) return toast("Interval range must satisfy 1 ≤ shortest ≤ longest ≤ 90");
  const body = { dataset: S.dataset, origin_hour: S.origin, t_min_days: tmin, t_max_days: tmax, epsilon: Number($("eps").value) };
  const box = $("decision-chart-card");
  box.replaceChildren();
  destroy("dec");
  const c = card(box, "Cost per day vs cleaning interval (re-simulated from the estimated state)", ["SYNTHETIC", "PHYSICS", "ML_FORECAST"], true);
  c.card.classList.add("wide");
  c.setLoading(true);
  btn.disabled = true;
  $("decision-summary").textContent = "Evaluating intervals… (re-simulation; first run for an origin can take ~10 s)";
  try {
    const d = await api("/api/decision", body);
    c.setLoading(false);
    const xs = d.candidates.map((k) => k.interval_days);
    const per = (k, f) => k.cost_breakdown[f] / (k.window_days || 1);
    const parts = [["cleaning_cost", "cleaning", C.sky], ["downtime_cost", "downtime", C.purple],
                   ["pumping_energy_cost", "pumping energy", C.green], ["thermal_penalty_cost", "thermal penalty", C.orange]];
    const datasets = parts.map(([f, lab, col]) => ({ type: "bar", label: lab + " ($/day)", data: d.candidates.map((k) => ({ x: k.interval_days, y: per(k, f) })),
                                                    backgroundColor: col, stack: "cost", yAxisID: "y", order: 2 }));
    datasets.push({ type: "line", label: "risk of crossing a limit before T (UDE)", data: d.candidates.map((k) => ({ x: k.interval_days, y: k.risk_limit_cross })),
                    borderColor: C.verm, borderWidth: 2, pointRadius: 2, pointStyle: "triangle", yAxisID: "risk", order: 1 });
    datasets.push({ type: "line", label: `ε = ${d.epsilon}`, data: [{ x: xs[0], y: d.epsilon }, { x: xs[xs.length - 1], y: d.epsilon }],
                    borderColor: C.verm, borderDash: [2, 4], borderWidth: 1, yAxisID: "risk", order: 1 });
    const cmp = [["fixed_30", "fixed 30 d", [10, 4]], ["condition_based", "condition-based", [3, 3]], ["never_clean", "never clean", [1, 4]]];
    for (const [k, lab, dash] of cmp) {
      const v = d.compared[k];
      if (v) datasets.push({ type: "line", label: `${lab}: ${fmtMoney(v.cost_per_day)}/day${v.feasible ? "" : " (infeasible)"}`,
                             data: [{ x: xs[0], y: v.cost_per_day }, { x: xs[xs.length - 1], y: v.cost_per_day }],
                             borderColor: C.text, borderDash: dash, borderWidth: 1.2, yAxisID: "y", stack: "ref-" + k, order: 0 });
    }
    // shaded ranges
    const ranges = [];
    const runs = (pred, pattern) => {
      let start = null;
      d.candidates.forEach((k, i) => {
        if (pred(k) && start === null) start = k.interval_days;
        const next = d.candidates[i + 1];
        if (start !== null && (!next || !pred(next))) { ranges.push({ from: start, to: k.interval_days, pattern }); start = null; }
      });
    };
    runs((k) => !k.feasible, hatch("rgba(213,94,0,0.55)", 1));
    runs((k) => k.feasible && k.risk_ok === false, hatch("rgba(240,228,66,0.45)", -1));
    S.charts.dec = new Chart(c.canvas, {
      data: { datasets },
      options: {
        maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        scales: {
          x: { type: "linear", min: xs[0] - 0.5, max: xs[xs.length - 1] + 0.5, stacked: true, title: { display: true, text: "cleaning interval T (days)" },
               afterBuildTicks: (axis) => { axis.ticks = [xs[0], ...[...Array(19).keys()].map((i) => (i + 1) * 5)].filter((v, i, a) => v >= xs[0] && v <= xs[xs.length - 1] && a.indexOf(v) === i).map((value) => ({ value })); } },
          y: { stacked: true, title: { display: true, text: "cost per day ($/day)" } },
          risk: { position: "right", min: 0, max: 1, grid: { drawOnChartArea: false }, title: { display: true, text: "risk (probability)" } },
        },
        plugins: {
          shade: { ranges },
          markers: { events: [], origin: d.recommended_interval_days, label: d.recommended_interval_days === null ? null : `recommended: ${d.recommended_interval_days} d` },
          legend: { position: "bottom" },
          tooltip: { callbacks: { title: (it) => `T = ${it[0].parsed.x} d` ,
            afterTitle: (it) => { const k = d.candidates.find((q) => q.interval_days === it[0].parsed.x);
              return k ? `${k.feasible ? "feasible" : "INFEASIBLE"}; risk ${k.risk_limit_cross === null ? "n/a" : fmt(k.risk_limit_cross, 2)}${k.risk_ok === false ? " > ε" : ""}` : ""; } } },
        },
      },
    });
    const rec = d.candidates.find((k) => k.interval_days === d.recommended_interval_days);
    $("decision-summary").replaceChildren(
      el("strong", { text: d.recommended_interval_days === null ? "No interval recommended." : `Recommended interval: ${d.recommended_interval_days} days` }),
      el("br"),
      rec ? `${fmtMoney(rec.cost_per_day)}/day over ${rec.window_days} days; risk ${rec.risk_limit_cross === null ? "n/a" : fmt(rec.risk_limit_cross, 2)} ≤ ε = ${d.epsilon}; feasible.` : (d.notes[0] || ""),
      el("br"),
      `State estimated at the origin from observations only: δ ≈ ${fmt(d.state_estimate_um, 1)} µm. `,
      "Shading: ", el("strong", { text: "/// orange = infeasible" }), " (a limit or the biocide permit is exceeded), ",
      el("strong", { text: "\\\\\\ yellow = risk above ε" }), ". Yellow vertical line = recommended interval.");
    const t = el("table", {}, el("caption", { text: "Comparison strategies (simulated over 90 days from the origin)" }));
    t.append(el("thead", {}, el("tr", {}, ...["strategy", "cost per day", "total (90 d)", "feasible", "hours above a limit", "peak Rf", "peak dP (kPa)"].map((h) => el("th", { scope: "col", text: h })))));
    const tb = el("tbody");
    for (const [k, lab] of cmp) {
      const v = d.compared[k];
      if (!v) continue;
      tb.append(el("tr", {}, el("th", { scope: "row", text: lab }), el("td", { text: fmtMoney(v.cost_per_day) }), el("td", { text: fmtMoney(v.total_cost) }),
        el("td", { text: v.feasible ? "yes" : "NO" }), el("td", { text: String(v.hours_above_limit) }), el("td", { text: fmtSci(v.peak_rf) }), el("td", { text: fmt(v.peak_dp_kpa, 2) })));
    }
    t.append(tb);
    $("decision-compare").replaceChildren(el("div", { class: "table-wrap" }, t));
    $("decision-notes").replaceChildren(el("ul", {}, ...d.notes.map((n) => el("li", { text: n }))));
  } catch (e) { c.setLoading(false); $("decision-summary").textContent = ""; toast("Decision: " + e.message); }
  finally { btn.disabled = false; }
}

// ---------------------------------------------------------------------------- 4 Models
const MODEL_STYLE = { persistence: [C.grey, [2, 3], "circle"], ridge: [C.purple, [6, 3], "rect"], gbm: [C.green, [10, 3, 2, 3], "triangle"],
                      ode: [C.orange, [4, 2], "rectRot"], ude: [C.blue, [], "star"] };
const DS_LABEL = { nominal_test: "nominal test", stress_no_clean: "stress: no clean", stress_long_interval: "stress: 90-day", mismatch_stress: "mismatch (held-out)" };

async function loadModels() {
  try {
    S.metrics = await api("/api/metrics");
    S.loaded.models = true;
    renderModels();
    $("metric-q").addEventListener("change", renderModels);
    $("metric-m").addEventListener("change", renderModels);
  } catch (e) { toast("Models: " + e.message); }
}

function renderModels() {
  const M = S.metrics, q = $("metric-q").value, m = $("metric-m").value;
  const hz = M.horizons_h.map(String);
  const v = $("verdict");
  v.replaceChildren(el("strong", { text: M.mock ? "MOCK metrics (no model loaded)" : "Verdict (synthetic data; see ml/README.md)" }));
  if (M.verdict) v.append(el("ul", {}, ...M.verdict.map((s) => el("li", { text: s }))));
  const box = $("models-charts");
  box.replaceChildren();
  Object.keys(S.charts).filter((k) => k.startsWith("mod-")).forEach(destroy);
  const tables = $("metrics-tables");
  tables.replaceChildren();
  const scale = q === "rf" && m !== "r2" ? RF_SCALE : 1;
  const unit = m === "r2" ? "" : { rf: " (1e-4 m²K/W)", u: " (W/m²K)", dp: " (kPa)" }[q];
  for (const [name, res] of Object.entries(M.datasets)) {
    const models = Object.keys(res.metrics).filter((k) => MODEL_STYLE[k]);
    const c = card(box, `${DS_LABEL[name] || name}: ${m.toUpperCase()} vs horizon`, ["SYNTHETIC", "ML_FORECAST"]);
    const datasets = models.map((k) => ({ label: k, data: hz.map((h) => ({ x: Number(h), y: res.metrics[k][h][q][m] * scale })),
      borderColor: MODEL_STYLE[k][0], borderDash: MODEL_STYLE[k][1], pointStyle: MODEL_STYLE[k][2], pointRadius: 4 }));
    S.charts["mod-" + name] = new Chart(c.canvas, { type: "line", data: { datasets }, options: {
      maintainAspectRatio: false, scales: { x: { type: "logarithmic", title: { display: true, text: "horizon (h)" } }, y: { title: { display: true, text: m.toUpperCase() + unit } } },
      plugins: { legend: { position: "bottom", labels: { usePointStyle: true } } } } });

    const t = el("table", {}, el("caption", { text: `${DS_LABEL[name] || name}: ${m.toUpperCase()} of ${q}${unit} (${res.origins} origins; best per horizon underlined)` }));
    t.append(el("thead", {}, el("tr", {}, el("th", { scope: "col", text: "model" }), ...hz.map((h) => el("th", { scope: "col", text: h + " h" })))));
    const tb = el("tbody");
    const best = {};
    for (const h of hz) {
      const vals = models.map((k) => res.metrics[k][h][q][m]).filter((x) => x !== null);
      best[h] = m === "r2" ? Math.max(...vals) : Math.min(...vals);
    }
    const allModels = Object.keys(res.metrics);
    for (const k of allModels) {
      const row = el("tr", {}, el("th", { scope: "row", text: k }));
      for (const h of hz) {
        const x = res.metrics[k][h][q][m];
        const td = el("td", { text: x === null ? "–" : m === "r2" ? fmt(x, 3) : fmt(x * scale, q === "rf" ? 3 : 2) });
        if (MODEL_STYLE[k] && x === best[h]) td.className = "best";
        row.append(td);
      }
      tb.append(row);
    }
    t.append(tb);
    tables.append(el("div", { class: "table-wrap" }, t));
    if (res.coverage_90) {
      const ct = el("table", {}, el("caption", { text: `${DS_LABEL[name] || name}: empirical coverage of the 90% band for Rf (target 0.90)` }));
      ct.append(el("thead", {}, el("tr", {}, el("th", { text: "model" }), ...hz.map((h) => el("th", { text: h + " h" })))));
      const cb = el("tbody");
      for (const [k, cov] of Object.entries(res.coverage_90)) cb.append(el("tr", {}, el("th", { scope: "row", text: k }), ...hz.map((h) => el("td", { text: fmt(cov[h], 2) }))));
      ct.append(cb);
      tables.append(el("div", { class: "table-wrap" }, ct));
    }
    if (res.time_to_limit) {
      const tt = el("table", {}, el("caption", { text: `${DS_LABEL[name] || name}: time-to-limit error (mean |predicted - actual| crossing hour)` }));
      tt.append(el("thead", {}, el("tr", {}, ...["limit", "actual crossing (h)", "method", "lead 1-7 d", "lead 7-30 d", "lead 30-60 d", "all", "90% interval coverage", "never predicted"].map((h) => el("th", { text: h })))));
      const tb2 = el("tbody");
      for (const [lim, r] of Object.entries(res.time_to_limit)) {
        if (!r.actual_first_crossing_h) { tb2.append(el("tr", {}, el("th", { text: lim }), el("td", { text: "never crossed" }), el("td", { text: "–" }))); continue; }
        for (const meth of ["ude", "ode", "linear_trend_7d"]) {
          const x = r[meth];
          const cell = (b) => (x[b] && x[b].mae_h !== null ? `${fmt(x[b].mae_h, 0)} h (n=${x[b].n})` : "–");
          tb2.append(el("tr", {}, el("th", { text: lim }), el("td", { text: String(r.actual_first_crossing_h) }), el("td", { text: meth }),
            el("td", { text: cell("lead_1_7d") }), el("td", { text: cell("lead_7_30d") }), el("td", { text: cell("lead_30_60d") }), el("td", { text: cell("all") }),
            el("td", { text: x.interval_90_coverage === undefined ? "–" : fmt(x.interval_90_coverage, 2) }), el("td", { text: String(x.never_predicted) })));
        }
      }
      tt.append(tb2);
      tables.append(el("div", { class: "table-wrap" }, tt));
    }
  }
}

// ---------------------------------------------------------------------------- 5 Correction
async function loadCorrection() {
  const box = $("correction-charts");
  try {
    S.correction = await api("/api/correction");
    S.loaded.correction = true;
    box.replaceChildren();
    let maxDev = 0;
    for (const s of S.correction.slices) {
      for (const g of s.g) maxDev = Math.max(maxDev, Math.abs(g - 1));
      const c = card(box, `g vs ${s.name}`, ["ML_FORECAST"]);
      const ds = [];
      if (s.g_min) {
        ds.push({ label: "ensemble min", data: s.x.map((x, i) => ({ x, y: s.g_min[i] })), borderColor: "transparent" });
        ds.push({ label: "ensemble range", data: s.x.map((x, i) => ({ x, y: s.g_max[i] })), borderColor: "transparent", backgroundColor: C.blue + "40", fill: "-1" });
      }
      ds.push({ label: "learned g (mean)", data: s.x.map((x, i) => ({ x, y: s.g[i] })), borderColor: C.blue, borderWidth: 2 });
      if (s.generator_equivalent) ds.push({ label: "generator's nutrient factor (rescaled)", data: s.x.map((x, i) => ({ x, y: s.generator_equivalent[i] })), borderColor: C.text, borderDash: [5, 4] });
      ds.push({ label: "g = 1 (plain physics)", data: [{ x: s.x[0], y: 1 }, { x: s.x[s.x.length - 1], y: 1 }], borderColor: C.grey, borderDash: [2, 3], borderWidth: 1 });
      S.charts["cor-" + s.name] = lineChart(c.canvas, ds, { xTitle: s.name, yTitle: "g" });
    }
    const ref = S.correction.reference;
    $("correction-note").replaceChildren(
      maxDev < 0.05 ? el("strong", { text: `g stays within ±5% of 1 everywhere shown (max deviation ${fmt(maxDev * 100, 1)}%): the network adds little beyond the physics. ` })
                    : el("strong", { text: `g departs from 1 by up to ${fmt(maxDev * 100, 0)}% in the ranges shown. ` }),
      ref ? `Reference state: T ${ref.T} °C, δ ${ref.delta_um} µm, ${ref.hours_since_cleaning} h since cleaning, hour ${ref.hour}. ` : "",
      S.correction.mock ? "MOCK data." : "Season panel: the dashed line is what g would need to be for the UDE to match the generator exactly on nominal data.");
  } catch (e) { toast("Correction: " + e.message); }
}

// ---------------------------------------------------------------------------- init
async function init() {
  setupTabs();
  try {
    S.meta = await api("/api/meta");
  } catch (e) { toast("Cannot reach the API: " + e.message); return; }
  $("disclaimer").textContent = S.meta.disclaimer;
  if (/MOCK/.test(S.meta.disclaimer)) { const b = $("mode-badge"); b.hidden = false; b.textContent = "MOCK MODE: canned responses, no model"; }
  const params = new URLSearchParams(location.search);
  if (S.meta.datasets.some((d) => d.id === params.get("dataset"))) S.dataset = params.get("dataset");
  if (params.get("origin")) S.origin = Number(params.get("origin"));
  setupShared(); setupForecast(); setupDecision();
  $("dataset").value = S.dataset;
  $("reset-zoom").addEventListener("click", resetZoom);
  const start = location.hash.replace("#", "");
  selectTab(TABS.includes(start) ? start : "monitor");
}
init();
