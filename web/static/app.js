/* Biofouling UDE dashboard. Vanilla JS + vendored Chart.js 4.4.1 (+ zoom plugin 2.0.1).
   All data is SYNTHETIC; academic demonstration, not for plant use. */
"use strict";

// Okabe-Ito palette (colour-blind safe); series are also told apart by dash pattern and legend text.
const C = { blue: "#56B4E9", sky: "#0072B2", orange: "#E69F00", green: "#009E73", verm: "#D55E00",
            purple: "#CC79A7", yellow: "#F0E442", grey: "#9aa3ad", text: "#e6e8eb", grid: "#2c333d" };
const RF_SCALE = 1e4; // Rf shown in 1e-4 m²K/W

const S = { meta: null, dataset: "nominal", origin: 3000, series: {}, fullSeries: {}, charts: {}, cleanings: [],
            metrics: null, correction: null, loaded: {}, decisions: {}, rfTables: {}, overviewToken: 0 };

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
const TABS = ["overview", "monitor", "rftable", "forecast", "decision", "costs", "models", "correction", "about"];
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
  if (name === "overview") renderOverview();
  if (name === "rftable") renderRfTable();
  if (name === "costs") renderCostTable();
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
  const originChanged = () => {
    S.loaded.forecast = S.loaded.decision = false;
    const active = TABS.find((t) => !$("panel-" + t).hidden);
    if (["overview", "rftable", "costs"].includes(active)) onTabShown(active);
  };
  $("origin").addEventListener("change", originChanged);
  $("origin-num").addEventListener("change", (e) => { setOrigin(e.target.value); originChanged(); });
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
    const d = await getDecision(body);
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

// ---------------------------------------------------------------------------- shared helpers for the plain views
function getDecision(body) {
  const key = JSON.stringify(body);
  if (!S.decisions[key]) S.decisions[key] = api("/api/decision", body).catch((e) => { delete S.decisions[key]; throw e; });
  return S.decisions[key];
}
function decisionBody() {
  return { dataset: S.dataset, origin_hour: S.origin, t_min_days: Number($("tmin").value) || 1,
           t_max_days: Number($("tmax").value) || 90, epsilon: Number($("eps").value) };
}
function getRfTable(id, every) {
  const key = id + ":" + every;
  if (!S.rfTables[key]) S.rfTables[key] = api(`/api/rf_table?dataset=${id}&every=${every}`).catch((e) => { delete S.rfTables[key]; throw e; });
  return S.rfTables[key];
}
const STATUS = { "Good": ["good", "✅"], "Watch": ["watch", "👀"], "Act soon": ["soon", "⚠️"], "Over limit": ["over", "⛔"] };
function statusPill(s) { const [cls, sym] = STATUS[s] || ["watch", "•"]; return el("span", { class: "status " + cls, text: `${sym} ${s}` }); }
const STATUS_COLOR = { good: C.green, watch: C.yellow, soon: C.orange, over: C.verm };
const rf4 = (v) => (v === null || v === undefined ? "–" : (v * 1e4).toFixed(3));
const money = (v) => (v === null || v === undefined ? "–" : "$" + Math.round(v).toLocaleString());
const num = (v, d = 1) => (v === null || v === undefined ? "–" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));

function downloadCsv(filename, header, rows) {
  const esc = (v) => { const t = v === null || v === undefined ? "" : String(v); return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t; };
  const text = [header, ...rows].map((r) => r.map(esc).join(",")).join("\n");
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: filename });
  document.body.append(a); a.click(); a.remove();
}

function buildTable(groups, columns, rows, rowClass) {
  // groups: [[label, span, className]], columns: [{label, unit, cls, cell(row) -> string|Node}]
  const t = el("table");
  const thead = el("thead");
  if (groups) thead.append(el("tr", { class: "group" }, ...groups.map(([lab, span, cls]) => el("th", { colspan: String(span), scope: "colgroup", class: cls || "", text: lab }))));
  thead.append(el("tr", {}, ...columns.map((c) => { const th = el("th", { scope: "col", class: c.cls || "" }, c.label); if (c.unit) th.append(el("span", { class: "unit", text: c.unit })); if (c.title) th.title = c.title; return th; })));
  const tb = el("tbody");
  for (const r of rows) {
    const tr = el("tr", { class: rowClass ? rowClass(r) : "" });
    columns.forEach((c, i) => {
      const v = c.cell(r);
      const td = el(i === 0 ? "th" : "td", i === 0 ? { scope: "row", class: c.cls || "" } : { class: c.cls || "" });
      if (v instanceof Node) td.append(v); else td.textContent = v;
      tr.append(td);
    });
    tb.append(tr);
  }
  t.append(thead, tb);
  return t;
}

// ---------------------------------------------------------------------------- 0 Overview
async function renderOverview() {
  const token = ++S.overviewToken;
  const box = $("overview-cards");
  const ds = datasetMeta();
  $("overview-when").textContent = `Showing "today" = day ${hToD(S.origin)} (hour ${S.origin}) of "${ds.label}". Move the slider above to look at another moment.`;
  box.replaceChildren(el("p", { class: "hint", text: "Working it out…" }));
  const card = (title, big, ...rest) => el("div", { class: "card" }, el("h3", { text: title }), big, ...rest);
  try {
    const t = await getRfTable(S.dataset, 24);
    if (token !== S.overviewToken) return;
    const rows = t.rows.filter((r) => r.hour <= S.origin);
    const r = rows.length ? rows[rows.length - 1] : t.rows[0];
    const k = t.constants;
    const [cls] = STATUS[r.status];
    const pct = Math.max(0, r.pct_of_rf_limit);
    const meter = el("div", { class: "meter", role: "img", "aria-label": `${pct.toFixed(0)}% of the limit` },
      el("span", { style: `width:${Math.min(100, pct)}%;background:${STATUS_COLOR[cls]}` }));
    const lastClean = ds.cleaning_events.filter((e) => e.hour <= S.origin).pop();
    box.replaceChildren(
      card("How dirty are the tubes?", el("div", { class: "big" }, statusPill(r.status)),
        el("p", { text: `The slime layer is at ${pct.toFixed(0)}% of the safety limit (Rf = ${rf4(r.rf)} × 10⁻⁴ m²K/W; limit ${rf4(k.rf_limit)}).` }),
        meter, el("div", { class: "meter-label" }, el("span", { text: "clean" }), el("span", { text: "limit" })),
        el("p", { class: "hint", text: lastClean ? `Last cleaned on day ${hToD(lastClean.hour)} (${lastClean.type}).` : "Not cleaned yet in this data." })),
      card("Slime thickness", el("div", { class: "big", text: `${num(r.thickness_um, 0)} µm` }),
        el("p", { text: `About ${num(r.thickness_um / 70, 1)} times the thickness of a human hair (≈ 70 µm). That is thin, but slime insulates heat very well.` })),
      card("Cooling lost", el("div", { class: "big", text: `${num(r.cooling_lost_pct, 1)}%` }),
        el("p", { text: `The exchanger passes about ${num(r.cooling_lost_pct, 1)}% less heat than when it is clean (measured U ${num(r.u_measured, 0)} vs ${num(k.u_clean, 0)} W/m²K when clean).` })),
      card("Extra pump effort", el("div", { class: "big", text: `${r.extra_pressure_pct >= 0 ? "+" : ""}${num(r.extra_pressure_pct, 1)}%` }),
        el("p", { text: `The water loses ${num(r.dp_measured_kpa, 2)} kPa of pressure in the tubes, against ${num(k.dp_clean_kpa, 2)} kPa when clean (limit ${num(k.dp_limit_kpa, 0)} kPa).` })),
      card("If we don't clean", el("div", { class: "big", id: "ov-ttl", text: "…" }), el("p", { id: "ov-ttl-text", text: "Asking the forecast model…" })),
      card("Suggested cleaning plan", el("div", { class: "big", id: "ov-plan", text: "…" }), el("p", { id: "ov-plan-text", text: "Comparing 90 cleaning plans (takes a few seconds the first time)…" }),
        el("p", { class: "hint", text: "Advice only: plant staff and procedures decide." })),
    );
  } catch (e) { box.replaceChildren(); toast("Overview: " + e.message); return; }

  api("/api/forecast", { dataset: S.dataset, origin_hour: S.origin, horizon_h: 720, model: "ude", planned_cleanings: [], exog: "climatology" })
    .then((f) => {
      if (token !== S.overviewToken) return;
      const t = f.time_to_limit.rf;
      if (t && t.median !== null) {
        const d = (t.median - S.origin) / 24;
        $("ov-ttl").textContent = d < 1 ? "Within a day" : `About ${Math.round(d)} days`;
        const lo = t.lo === null ? null : Math.max(0, (t.lo - S.origin) / 24), hi = t.hi === null ? null : (t.hi - S.origin) / 24;
        $("ov-ttl-text").textContent = `until the slime passes the safety limit, if nobody cleans (likely between ${lo === null ? "?" : Math.round(lo)} and ${hi === null ? "more than 30" : Math.round(hi)} days).`;
      } else {
        $("ov-ttl").textContent = "Not within 30 days";
        $("ov-ttl-text").textContent = "The model does not expect the slime to reach the safety limit in the next 30 days, even without cleaning.";
      }
    }).catch((e) => { $("ov-ttl").textContent = "–"; $("ov-ttl-text").textContent = "Forecast unavailable: " + e.message; });

  getDecision(decisionBody()).then((d) => {
    if (token !== S.overviewToken) return;
    if (d.recommended_interval_days === null) {
      $("ov-plan").textContent = "Clean soon";
      $("ov-plan-text").textContent = "No regular cleaning plan keeps every limit and the chemical permit; clean as soon as practical and review.";
      return;
    }
    const rec = d.candidates.find((c) => c.interval_days === d.recommended_interval_days);
    const f30 = d.compared.fixed_30;
    $("ov-plan").textContent = `Every ${d.recommended_interval_days} days`;
    $("ov-plan-text").textContent = `Estimated ${money(rec.cost_per_day)} per day` +
      (f30 ? `, compared with ${money(f30.cost_per_day)} per day for cleaning every 30 days${f30.feasible ? "" : " (which would break a limit)"}.` : ".");
  }).catch((e) => { $("ov-plan").textContent = "–"; $("ov-plan-text").textContent = "Unavailable: " + e.message; });
}

// ---------------------------------------------------------------------------- Fouling (Rf) table
let rfRowsShown = [];
async function renderRfTable() {
  const every = Number($("rf-every").value);
  const box = $("rf-table-box");
  box.replaceChildren(el("p", { class: "hint", text: "Calculating…" }));
  try {
    const t = await getRfTable(S.dataset, every);
    const k = t.constants;
    $("rf-formula").replaceChildren(
      el("strong", { text: "How each row is calculated" }),
      el("ul", {},
        el("li", {}, "Measured values: the median of the last ", el("strong", { text: `${k.block_hours} hours` }), " of sensor readings (since the last cleaning)."),
        el("li", {}, el("strong", { text: "Step 1, from U: " }), el("code", { text: `Rf = 1/U − 1/${k.u_clean}` }), " , then ", el("code", { text: `thickness = Rf × ${k.k_biofilm}` }),
          ` (clean U = ${k.u_clean} W/m²K; slime conducts heat at ${k.k_biofilm} W/m·K).`),
        el("li", {}, el("strong", { text: "Step 2, from dP: " }), `find the thickness that makes the pipe-flow equations (Darcy-Weisbach with Haaland friction) give the measured dP at the measured flow: ${k.number_of_tubes} tubes, ${k.tube_inner_diameter_m * 1000} mm wide, ${k.tube_length_m} m long; clean dP = ${num(k.dp_clean_kpa, 2)} kPa.`),
        el("li", {}, el("strong", { text: "Result: " }), el("code", { text: "thickness = average of step 1 and step 2" }), " , ", el("code", { text: `calculated Rf = thickness / ${k.k_biofilm}` }),
          ` ; % of limit = Rf / ${rf4(k.rf_limit)}×10⁻⁴ ; cooling lost = 1 − U(Rf)/U_clean ; extra pressure = dP / clean dP − 1.`),
        el("li", { text: `Status: ${t.status_rule}` }),
        el("li", { text: "True Rf is the simulator's own value, shown only to check the calculation; a real plant would not know it." })));
    const attention = $("rf-filter").value === "attention";
    const rows = t.rows.filter((r) => !attention || r.status !== "Good");
    rfRowsShown = rows;
    const counts = {};
    t.rows.forEach((r) => (counts[r.status] = (counts[r.status] || 0) + 1));
    $("rf-summary").textContent = `${t.rows.length} rows (one every ${every / 24 === 1 ? "day" : every / 24 + " days"}): ` +
      Object.entries(counts).map(([s, n]) => `${n} ${s}`).join(", ") + (attention ? `. Showing ${rows.length} that need attention.` : ".") +
      " The highlighted row is \"today\".";
    const todayRow = t.rows.filter((r) => r.hour <= S.origin).pop();
    const cols = [
      { label: "Day", cell: (r) => { const f = document.createDocumentFragment(); f.append(num(r.day, 0)); if (r.cleaned_since_last_row.length) f.append(el("br"), el("span", { class: "hint", text: "🧽 cleaned (" + r.cleaned_since_last_row.map((c) => c.type[0].toUpperCase()).join("+") + ")" })); return f; } },
      { label: "Date and time", cell: (r) => r.timestamp.slice(0, 16) },
      { label: "Water temperature", unit: "°C", cell: (r) => num(r.water_temp_c, 1) },
      { label: "Water flow", unit: "kg/s", cell: (r) => num(r.flow_kg_s, 1) },
      { label: "Measured U", unit: "W/m²K", cell: (r) => num(r.u_measured, 0) },
      { label: "Measured dP", unit: "kPa", cell: (r) => num(r.dp_measured_kpa, 2) },
      { label: "Rf from U", unit: "× 10⁻⁴ m²K/W", cell: (r) => rf4(r.rf_from_u) },
      { label: "Thickness from U", unit: "µm", cell: (r) => num(r.thickness_from_u_um, 1) },
      { label: "Thickness from dP", unit: "µm", cell: (r) => num(r.thickness_from_dp_um, 1) },
      { label: "Average thickness", unit: "µm", cls: "group-result", cell: (r) => num(r.thickness_um, 1) },
      { label: "Calculated Rf", unit: "× 10⁻⁴ m²K/W", cls: "group-result", cell: (r) => rf4(r.rf) },
      { label: "% of Rf limit", unit: "%", cls: "group-result", cell: (r) => num(r.pct_of_rf_limit, 0) },
      { label: "Cooling lost", unit: "%", cls: "group-result", cell: (r) => num(r.cooling_lost_pct, 1) },
      { label: "Extra pressure", unit: "%", cls: "group-result", cell: (r) => num(r.extra_pressure_pct, 1) },
      { label: "Status", cls: "group-result", cell: (r) => statusPill(r.status) },
      { label: "True Rf (check)", unit: "× 10⁻⁴ m²K/W", cls: "group-result", cell: (r) => rf4(r.rf_true) },
    ];
    const groups = [["When", 2], ["Measured by sensors", 4], ["Step 1: from U", 2], ["Step 2: from dP", 1], ["Result", 7, "group-result"]];
    box.replaceChildren(buildTable(groups, cols, rows, (r) => (todayRow && r.hour === todayRow.hour ? "highlight" : "")));
    const hl = box.querySelector("tr.highlight");
    if (hl) box.scrollTop = Math.max(0, hl.offsetTop - box.clientHeight / 2);
  } catch (e) { box.replaceChildren(); toast("Fouling table: " + e.message); }
}

function downloadRfTable() {
  const h = ["day", "timestamp", "water_temp_c", "flow_kg_s", "u_measured_w_m2k", "dp_measured_kpa", "rf_from_u_m2k_w", "thickness_from_u_um",
             "thickness_from_dp_um", "thickness_um", "calculated_rf_m2k_w", "pct_of_rf_limit", "cooling_lost_pct", "extra_pressure_pct", "status", "true_rf_m2k_w", "cleaned"];
  downloadCsv(`fouling_table_${S.dataset}.csv`, h, rfRowsShown.map((r) => [r.day, r.timestamp, r.water_temp_c, r.flow_kg_s, r.u_measured, r.dp_measured_kpa, r.rf_from_u,
    r.thickness_from_u_um, r.thickness_from_dp_um, r.thickness_um, r.rf, r.pct_of_rf_limit, r.cooling_lost_pct, r.extra_pressure_pct, r.status, r.rf_true,
    r.cleaned_since_last_row.map((c) => c.type).join("+")]));
}

// ---------------------------------------------------------------------------- Cost table
let costRowsShown = [];
async function renderCostTable() {
  const box = $("cost-table-box");
  box.replaceChildren(el("p", { class: "hint", text: "Simulating 90 cleaning plans… (a few seconds the first time for a new \"today\")" }));
  const c = S.meta.costs, L = S.meta.limits;
  $("cost-formula").replaceChildren(
    el("strong", { text: "The cost formula" }),
    el("p", {}, el("code", { text: "Total cost = cleaning cost + downtime cost + pumping cost + heat-loss cost" })),
    el("ul", {},
      el("li", {}, el("strong", { text: "Cleaning cost" }), ` = mechanical cleanings × $${num(c.mechanical_cleaning_cost, 0)} + chemical cleanings × $${num(c.chemical_cleaning_cost, 0)} (plans alternate mechanical and chemical)`),
      el("li", {}, el("strong", { text: "Downtime cost" }), ` = number of cleanings × ${num(c.downtime_hours_per_cleaning, 0)} hours × $${num(c.downtime_cost_per_hour, 0)} per hour`),
      el("li", {}, el("strong", { text: "Pumping cost" }), ` = extra pump energy caused by the slime (kWh) × $${num(c.electricity_cost_per_kwh, 2)} per kWh`),
      el("li", {}, el("strong", { text: "Heat-loss cost" }), ` = cooling lost because of the slime (MWh) × $${num(c.thermal_penalty_cost_per_mwh, 0)} per MWh`),
      el("li", {}, el("strong", { text: "Allowed?" }), ` A plan is allowed only if Rf stays below ${rf4(L.rf_crit)}×10⁻⁴ m²K/W, dP below ${num(L.dp_crit_kpa, 0)} kPa, the severity index below ${num(L.bsi_crit, 0)}, ` +
        `and biocide use stays within the ${num(c.biocide_permit_kg_per_year ?? 250, 0)} kg per year permit (scaled to the simulated period). The risk column (from the AI model) must also be at most ε, which is set on the "When to clean" tab.`)));
  const params = [
    ["Mechanical cleaning", money(c.mechanical_cleaning_cost), "per cleaning (sponge balls; removes 90% of the slime)"],
    ["Chemical cleaning", money(c.chemical_cleaning_cost), "per cleaning (biocide shock; removes 95%)"],
    ["Downtime per cleaning", `${num(c.downtime_hours_per_cleaning, 0)} h`, "the exchanger is stopped while it is cleaned"],
    ["Cost of downtime", `${money(c.downtime_cost_per_hour)} per hour`, "lost production and staff time"],
    ["Electricity price", `$${num(c.electricity_cost_per_kwh, 2)} per kWh`, "for the extra pumping caused by the slime"],
    ["Value of lost cooling", `${money(c.thermal_penalty_cost_per_mwh)} per MWh`, "penalty for heat the exchanger could not remove"],
    ["Biocide per chemical cleaning", `${num(c.biocide_kg_per_event, 0)} kg`, "chemical released to the environment"],
    ["Biocide permit", `${num(c.biocide_permit_kg_per_year ?? 250, 0)} kg per year`, "environmental limit"],
    ["Rf limit", `${rf4(L.rf_crit)} × 10⁻⁴ m²K/W`, "fouling safety limit"],
    ["dP limit", `${num(L.dp_crit_kpa, 0)} kPa`, "pressure-drop safety limit"],
    ["Acceptable risk ε", num(Number($("eps").value), 2), "highest allowed chance (from the AI model) of passing a limit before the first cleaning"],
  ];
  $("cost-params-box").replaceChildren(el("table", {}, el("caption", { text: "Parameters used (fixed values from config.py)" }),
    el("thead", {}, el("tr", {}, ...["Parameter", "Value", "Meaning"].map((h) => el("th", { scope: "col", text: h })))),
    el("tbody", {}, ...params.map(([a, b, m]) => el("tr", {}, el("th", { scope: "row", text: a }), el("td", { text: b }), el("td", { class: "reason", text: m }))))));
  try {
    const d = await getDecision(decisionBody());
    const rec = d.recommended_interval_days;
    const allowed = (r) => r.feasible && r.risk_ok !== false;
    const why = (r) => {
      if (allowed(r)) return "✅ Yes";
      const parts = [];
      if (!r.feasible) parts.push(r.infeasible_reason || "breaks a limit");
      if (r.risk_ok === false) parts.push(`risk ${num(r.risk_limit_cross, 2)} is above ε = ${num(d.epsilon, 2)}`);
      return "⛔ No: " + parts.join("; ");
    };
    const filter = $("cost-filter").value;
    let rows = d.candidates.map((r) => ({ ...r, plan: r.interval_days === 1 ? "Every day" : `Every ${r.interval_days} days` }));
    if (filter === "allowed") rows = rows.filter(allowed);
    if (filter === "every5") rows = rows.filter((r) => r.interval_days % 5 === 0 || r.interval_days === 1 || r.interval_days === rec);
    const comp = [["fixed_30", "Fixed: every 30 days"], ["condition_based", "Clean when 80% of a limit is reached"], ["never_clean", "Never clean"]]
      .filter(([k]) => d.compared[k]).map(([k, lab]) => ({ ...d.compared[k], plan: lab, risk_limit_cross: null, risk_ok: null }));
    const b = (r, k) => (r.cost_breakdown && r.cost_breakdown[k] !== undefined ? r.cost_breakdown[k] : null);
    const cols = [
      { label: "Cleaning plan", cell: (r) => (r.interval_days === rec && !r.plan.startsWith("Fixed") ? `★ ${r.plan} (recommended)` : r.plan) },
      { label: "Simulated period", unit: "days", cell: (r) => num(r.window_days, 0) },
      { label: "Cleanings", unit: "mechanical + chemical", cell: (r) => (b(r, "cleanings_mechanical") === null ? "–" : `${b(r, "cleanings_mechanical")} + ${b(r, "cleanings_chemical")}`) },
      { label: "Cleaning cost", unit: "$", cell: (r) => money(b(r, "cleaning_cost")) },
      { label: "Hours stopped", unit: "h", cell: (r) => num(b(r, "downtime_hours"), 0) },
      { label: "Downtime cost", unit: "$", cell: (r) => money(b(r, "downtime_cost")) },
      { label: "Extra pump energy", unit: "kWh", cell: (r) => num(b(r, "pump_energy_kwh"), 0) },
      { label: "Pumping cost", unit: "$", cell: (r) => money(b(r, "pumping_energy_cost")) },
      { label: "Cooling lost", unit: "MWh", cell: (r) => num(b(r, "thermal_shortfall_mwh"), 0) },
      { label: "Heat-loss cost", unit: "$", cell: (r) => money(b(r, "thermal_penalty_cost")) },
      { label: "Biocide used / allowed", unit: "kg", cell: (r) => (b(r, "biocide_kg") === null ? "–" : `${num(b(r, "biocide_kg"), 0)} / ${num(b(r, "biocide_limit_kg"), 0)}`) },
      { label: "Highest Rf", unit: "× 10⁻⁴ m²K/W", cell: (r) => rf4(r.peak_rf) },
      { label: "Highest dP", unit: "kPa", cell: (r) => num(r.peak_dp_kpa, 2) },
      { label: "Risk of passing a limit", unit: "AI model, 0-1", cell: (r) => (r.risk_limit_cross === null || r.risk_limit_cross === undefined ? "–" : num(r.risk_limit_cross, 2)) },
      { label: "Allowed?", cls: "group-result reason", cell: (r) => why(r) },
      { label: "Total cost", unit: "$ over the period", cls: "group-result", cell: (r) => money(r.total_cost) },
      { label: "Cost per day", unit: "$ per day", cls: "group-result", cell: (r) => money(r.cost_per_day) },
    ];
    const groups = [["Plan", 3], ["Cleaning", 1], ["Downtime", 2], ["Pumping", 2], ["Cooling", 2], ["Chemical & safety", 4], ["Results", 3, "group-result"]];
    const cls = (r) => (r.interval_days === rec && r.plan.startsWith("Every") ? "recommended" : allowed(r) ? "" : "not-allowed");
    costRowsShown = [...comp, ...rows];
    const nAllowed = d.candidates.filter(allowed).length;
    $("cost-summary").textContent = rec === null
      ? `None of the ${d.candidates.length} plans is allowed from this "today". Clean as soon as practical and review.`
      : `${nAllowed} of ${d.candidates.length} plans are allowed. Cheapest allowed: every ${rec} days. The first three rows are reference strategies (simulated over 90 days).`;
    box.replaceChildren(buildTable(groups, cols, costRowsShown, cls));
  } catch (e) { box.replaceChildren(); toast("Cost table: " + e.message); }
}

function downloadCostTable() {
  const h = ["plan", "simulated_days", "cleanings_mechanical", "cleanings_chemical", "cleaning_cost", "downtime_hours", "downtime_cost", "pump_energy_kwh",
             "pumping_cost", "cooling_lost_mwh", "heat_loss_cost", "biocide_kg", "biocide_allowed_kg", "peak_rf", "peak_dp_kpa", "risk", "feasible", "reason", "total_cost", "cost_per_day"];
  downloadCsv(`cost_table_${S.dataset}_h${S.origin}.csv`, h, costRowsShown.map((r) => {
    const b = r.cost_breakdown || {};
    return [r.plan, r.window_days, b.cleanings_mechanical, b.cleanings_chemical, b.cleaning_cost, b.downtime_hours, b.downtime_cost, b.pump_energy_kwh,
            b.pumping_energy_cost, b.thermal_shortfall_mwh, b.thermal_penalty_cost, b.biocide_kg, b.biocide_limit_kg, r.peak_rf, r.peak_dp_kpa,
            r.risk_limit_cross, r.feasible, r.infeasible_reason, r.total_cost, r.cost_per_day];
  }));
}

function setupPlainViews() {
  $("rf-every").addEventListener("change", renderRfTable);
  $("rf-filter").addEventListener("change", renderRfTable);
  $("rf-download").addEventListener("click", downloadRfTable);
  $("cost-filter").addEventListener("change", renderCostTable);
  $("cost-download").addEventListener("click", downloadCostTable);
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
  setupShared(); setupForecast(); setupDecision(); setupPlainViews();
  $("dataset").value = S.dataset;
  $("reset-zoom").addEventListener("click", resetZoom);
  const start = location.hash.replace("#", "");
  selectTab(TABS.includes(start) ? start : "overview");
}
init();
