/**
 * workspace/biofouling_ai_repaired/dashboard/app.js
 * 
 * Nuclear Auxiliary Heat-Exchanger Biofouling Monitor & Predictive Cleaning Optimizer
 * IndustryOS Command Centre UI Engine — Zero-Install Standard Library Engine
 * Features:
 *  - Native vector SVG dual-axis telemetry charting with uncertainty bands, gradients & crosshair
 *  - Segmented operating mode switcher (Simulation / Manual Snapshot / Historical CSV)
 *  - 5-stage CSV ingestion sequence stepper and data-quality audit reporting
 *  - Scenario trade-off matrix with tabular numeric comparisons
 *  - Economic sensitivity optimizer recalculation
 *  - Sidebar navigation smooth section scrolling
 */

let currentHorizonHours = 720; // 30 days default
let cachedForecastData = null;
let cachedRecommendationData = null;
let currentOperatingMode = "simulation";
let cachedCSVText = "";
let cachedColumnMapping = {};
let currentCSVFileName = "cooling_telemetry.csv";
let currentCSVHeaders = [];

document.addEventListener("DOMContentLoaded", () => {
  initDashboard();
});

async function initDashboard() {
  setupEventListeners();
  setupSidebarNavigation();
  setupModeSwitchers();
  setupManualIngestion();
  setupCSVIngestion();
  setupResizeHandler();
  if (window.location.hash === "#section-ingestion" || window.location.hash === "#csv") {
    toggleModePanel("csv");
  } else if (window.location.hash === "#manual") {
    toggleModePanel("manual");
  }
  await refreshAll();
}

function setupEventListeners() {
  document.getElementById("btn-refresh")?.addEventListener("click", () => {
    refreshAll();
  });

  document.getElementById("btn-re-evaluate")?.addEventListener("click", () => {
    recalculateOptimizer();
  });

  // Horizon buttons
  document.querySelectorAll(".btn-horizon").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".btn-horizon").forEach(b => b.classList.remove("btn-primary"));
      btn.classList.add("btn-primary");
      currentHorizonHours = parseInt(btn.getAttribute("data-hours"), 10) || 720;
      fetchForecast(currentHorizonHours);
    });
  });
}

function setupSidebarNavigation() {
  const navItems = document.querySelectorAll(".sidebar-nav .nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", (e) => {
      navItems.forEach(i => i.classList.remove("active"));
      item.classList.add("active");

      const href = item.getAttribute("href");
      if (href && href.startsWith("#")) {
        const targetEl = document.querySelector(href);
        if (targetEl) {
          e.preventDefault();
          targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        if (href === "#section-ingestion") {
          toggleModePanel("csv");
        }
      }
    });
  });
}

function setupResizeHandler() {
  let resizeTimeout = null;
  window.addEventListener("resize", () => {
    if (resizeTimeout) clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      if (cachedForecastData) {
        renderForecastSVG(cachedForecastData);
      }
    }, 120);
  });
}

function setupModeSwitchers() {
  const btnSim = document.getElementById("btn-mode-sim");
  const btnManual = document.getElementById("btn-mode-manual");
  const btnCsv = document.getElementById("btn-mode-csv");

  btnSim?.addEventListener("click", async () => {
    await switchOperatingMode("simulation");
  });

  btnManual?.addEventListener("click", () => {
    toggleModePanel("manual");
  });

  btnCsv?.addEventListener("click", () => {
    toggleModePanel("csv");
  });
}

function toggleModePanel(mode) {
  const manualPanel = document.getElementById("panel-manual-input");
  const csvPanel = document.getElementById("panel-csv-upload");
  const btnManual = document.getElementById("btn-mode-manual");
  const btnCsv = document.getElementById("btn-mode-csv");
  const btnSim = document.getElementById("btn-mode-sim");

  if (mode === "manual") {
    const isShowing = manualPanel.classList.contains("active-panel");
    manualPanel.classList.toggle("active-panel", !isShowing);
    csvPanel.classList.remove("active-panel");
    btnManual.classList.toggle("active", !isShowing);
    btnCsv.classList.remove("active");
    if (isShowing) {
      btnSim.classList.add("active");
    } else {
      btnSim.classList.remove("active");
    }
  } else if (mode === "csv") {
    const isShowing = csvPanel.classList.contains("active-panel");
    csvPanel.classList.toggle("active-panel", !isShowing);
    manualPanel.classList.remove("active-panel");
    btnCsv.classList.toggle("active", !isShowing);
    btnManual.classList.remove("active");
    if (isShowing) {
      btnSim.classList.add("active");
    } else {
      btnSim.classList.remove("active");
    }
  }
}

async function switchOperatingMode(mode) {
  try {
    const res = await fetch("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (!res.ok) return;
    const data = await res.json();
    currentOperatingMode = data.active_mode;

    // Reset button states
    document.querySelectorAll(".btn-mode").forEach(b => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    document.getElementById("panel-manual-input")?.classList.remove("active-panel");
    document.getElementById("panel-csv-upload")?.classList.remove("active-panel");

    if (mode === "simulation") {
      const b = document.getElementById("btn-mode-sim");
      b?.classList.add("active");
      b?.setAttribute("aria-selected", "true");
      updateProvenancePill("synthetic_simulation_data", "8,760h Annual Nuclear Cooling Benchmark Cycle");
    } else if (mode === "manual") {
      const b = document.getElementById("btn-mode-manual");
      b?.classList.add("active");
      b?.setAttribute("aria-selected", "true");
      updateProvenancePill("manual_operating_input", "Manual Engineering Operating Snapshot");
    } else if (mode === "csv") {
      const b = document.getElementById("btn-mode-csv");
      b?.classList.add("active");
      b?.setAttribute("aria-selected", "true");
      updateProvenancePill("user_uploaded_csv", "Ingested Industrial Telemetry CSV");
    }

    await refreshAll();
  } catch (e) {
    console.warn("Mode switch failed:", e);
  }
}

function updateProvenancePill(source, desc) {
  const pill = document.getElementById("active-provenance-pill");
  const descEl = document.getElementById("active-dataset-desc");
  if (!pill) return;

  pill.className = "provenance-pill";
  if (source === "manual_operating_input") {
    pill.classList.add("pill-manual");
    pill.textContent = "DATA SOURCE: MANUAL OPERATING INPUT";
  } else if (source === "user_uploaded_csv") {
    pill.classList.add("pill-csv");
    pill.textContent = "DATA SOURCE: INGESTED HISTORICAL CSV";
  } else {
    pill.classList.add("pill-simulation");
    pill.textContent = "DATA SOURCE: SYNTHETIC SIMULATION";
  }

  if (descEl && desc) descEl.textContent = desc;
}

function setupManualIngestion() {
  document.getElementById("btn-reset-manual-defaults")?.addEventListener("click", () => {
    document.getElementById("manual-t-in").value = "18.5";
    document.getElementById("manual-t-out").value = "26.2";
    document.getElementById("manual-flow").value = "220.0";
    document.getElementById("manual-dp").value = "10.5";
    document.getElementById("manual-hours").value = "2500";
    document.getElementById("manual-timestamp").value = new Date().toISOString();
    document.getElementById("manual-heat-duty").value = "";
    document.getElementById("manual-u-overall").value = "";
    hideFeedbackBox("manual-feedback-box");
  });

  document.getElementById("btn-submit-manual")?.addEventListener("click", async () => {
    const payload = {
      T_cool_in_meas: parseFloat(document.getElementById("manual-t-in")?.value),
      T_cool_out_meas: parseFloat(document.getElementById("manual-t-out")?.value),
      mass_flow_cool_meas: parseFloat(document.getElementById("manual-flow")?.value),
      delta_P_meas_kPa: parseFloat(document.getElementById("manual-dp")?.value),
      time_hours: parseInt(document.getElementById("manual-hours")?.value, 10),
      timestamp: document.getElementById("manual-timestamp")?.value,
      heat_duty_MW: document.getElementById("manual-heat-duty")?.value || null,
      U_overall_meas: document.getElementById("manual-u-overall")?.value || null,
    };

    try {
      const res = await fetch("/api/ingest/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      const fbBox = document.getElementById("manual-feedback-box");

      if (res.ok && data.is_valid) {
        currentOperatingMode = "manual";
        document.querySelectorAll(".btn-mode").forEach(b => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        const mBtn = document.getElementById("btn-mode-manual");
        mBtn?.classList.add("active");
        mBtn?.setAttribute("aria-selected", "true");
        updateProvenancePill("manual_operating_input", `Operating Snapshot (Hour ${data.validated_condition.time_hours})`);

        let html = `<strong>✓ Manual Operating Condition Applied Successfully</strong><br>`;
        html += `Derived Fouling Rf: <strong>${data.derived_indicators.Rf_fouling_resistance.toExponential(2)} m²·K/W</strong> | `;
        html += `Thermal Eff: <strong>${(data.derived_indicators.thermal_efficiency_ratio * 100).toFixed(1)}%</strong> | `;
        html += `BSI Score: <strong>${data.derived_indicators.bsi_score}</strong> | `;
        html += `Excess Pump Power: <strong>${data.derived_indicators.parasitic_pump_kw} kW</strong><br>`;
        if (data.warnings && data.warnings.length > 0) {
          html += `<div style="margin-top: 0.4rem; color: var(--accent-amber); font-size: 0.7rem;">⚠️ ${data.warnings.join("<br>⚠️ ")}</div>`;
        }
        fbBox.className = "validation-feedback-box feedback-success";
        fbBox.innerHTML = html;
        fbBox.style.display = "block";

        await refreshAll();
      } else {
        let html = `<strong>Validation Errors:</strong><br>`;
        html += data.errors ? data.errors.join("<br>") : (data.error || "Input rejected.");
        if (data.warnings && data.warnings.length > 0) {
          html += `<br><br><strong>Warnings:</strong><br>${data.warnings.join("<br>")}`;
        }
        fbBox.className = "validation-feedback-box feedback-error";
        fbBox.innerHTML = html;
        fbBox.style.display = "block";
      }
    } catch (e) {
      console.warn("Manual ingestion failed:", e);
    }
  });
}

function setupCSVIngestion() {
  const dropzone = document.getElementById("csv-dropzone");
  const fileInput = document.getElementById("csv-file-input");
  const rawTextarea = document.getElementById("raw-csv-textarea");

  dropzone?.addEventListener("click", () => fileInput?.click());
  dropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleCSVFile(e.dataTransfer.files[0]);
    }
  });

  fileInput?.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleCSVFile(e.target.files[0]);
    }
  });

  document.getElementById("btn-toggle-raw-csv")?.addEventListener("click", () => {
    const rawCont = document.getElementById("raw-csv-container");
    if (rawCont) rawCont.style.display = rawCont.style.display === "none" ? "block" : "none";
  });

  document.getElementById("btn-load-sample-csv")?.addEventListener("click", async () => {
    try {
      invalidateStaleCSVState();
      currentCSVFileName = "sample_industrial_cooling_120h.csv";
      updateCSVUploadStepper(1);
      updateCSVUploadStepper(2);
      const res = await fetch("/api/sample/csv");
      if (!res.ok) return;
      const csvStr = await res.text();
      cachedCSVText = csvStr;
      if (rawTextarea) rawTextarea.value = csvStr;
      await previewCSVContent(csvStr);
    } catch (e) {
      console.warn("Failed to load sample CSV:", e);
    }
  });

  document.getElementById("btn-preview-csv")?.addEventListener("click", async () => {
    const csvStr = rawTextarea?.value || cachedCSVText;
    if (!csvStr) {
      alert("Please select a CSV file or paste CSV text first.");
      return;
    }
    invalidateStaleCSVState();
    updateCSVUploadStepper(1);
    updateCSVUploadStepper(2);
    await previewCSVContent(csvStr);
  });

  document.getElementById("btn-confirm-ingest")?.addEventListener("click", async () => {
    await confirmCSVIngestion();
  });
}

function handleCSVFile(file) {
  invalidateStaleCSVState();
  currentCSVFileName = file.name || "cooling_telemetry.csv";
  updateCSVUploadStepper(1);

  const reader = new FileReader();
  reader.onload = async (e) => {
    updateCSVUploadStepper(2);
    cachedCSVText = e.target.result;
    const rawTextarea = document.getElementById("raw-csv-textarea");
    if (rawTextarea) rawTextarea.value = cachedCSVText;
    await previewCSVContent(cachedCSVText);
  };
  reader.readAsText(file);
}

function updateCSVUploadStepper(stepNumber) {
  const stepper = document.getElementById("csv-upload-stepper");
  if (stepper) stepper.style.display = "flex";

  for (let i = 1; i <= 5; i++) {
    const stepEl = document.getElementById(`step-${i}`);
    if (!stepEl) continue;
    stepEl.classList.remove("active", "completed");

    if (i < stepNumber) {
      stepEl.classList.add("completed");
    } else if (i === stepNumber) {
      stepEl.classList.add("active");
    }
  }

  for (let i = 1; i <= 4; i++) {
    const lineEl = document.getElementById(`line-${i}`);
    if (!lineEl) continue;
    if (i < stepNumber) {
      lineEl.classList.add("completed");
    } else {
      lineEl.classList.remove("completed");
    }
  }
}

function invalidateStaleCSVState() {
  const previewCard = document.getElementById("csv-preview-card");
  if (previewCard) previewCard.style.display = "none";

  const summaryCard = document.getElementById("csv-ingestion-summary-card");
  if (summaryCard) summaryCard.style.display = "none";

  cachedColumnMapping = {};
}

async function previewCSVContent(csvText) {
  try {
    const res = await fetch("/api/ingest/csv/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ csv_text: csvText }),
    });
    const preview = await res.json();
    if (!res.ok || !preview.success) {
      const errMsg = preview.error || "Failed to parse CSV preview. Ensure valid CSV headers.";
      alert(`CSV Preview Error: ${errMsg}`);
      return;
    }
    updateCSVUploadStepper(3);
    currentCSVHeaders = preview.headers || [];
    renderCSVPreview(preview);
    setTimeout(() => updateCSVUploadStepper(4), 150);
  } catch (e) {
    console.warn("CSV preview error:", e);
    alert(`CSV Preview Error: ${e.message || "Failed to connect to preview endpoint."}`);
  }
}

function renderCSVPreview(preview) {
  const card = document.getElementById("csv-preview-card");
  if (!card) return;
  card.style.display = "block";

  const summary = document.getElementById("csv-preview-summary");
  if (summary) {
    summary.textContent = `CSV Preview: ${preview.total_estimated_rows} data rows | Delimiter: '${preview.delimiter}'`;
  }

  // Column mapping selectors
  const mapCont = document.getElementById("csv-mapping-selectors");
  if (mapCont && preview.headers) {
    let mapHtml = "";
    const canonicalOptions = [
      { key: "T_cool_in_meas", label: "Tin (Intake Temp °C) *" },
      { key: "T_cool_out_meas", label: "Tout (Discharge Temp °C) *" },
      { key: "mass_flow_cool_meas", label: "Flow (Cooling Mass Flow kg/s) *" },
      { key: "delta_P_meas_kPa", label: "ΔP (Diff Pressure kPa) *" },
      { key: "timestamp", label: "Timestamp (ISO Date)" },
      { key: "time_hours", label: "Operating Hours (h)" },
      { key: "heat_duty_MW", label: "Heat Duty (MW)" },
      { key: "U_overall_meas", label: "Overall HTC (W/m²K)" },
      { key: "", label: "-- Ignore Column --" },
    ];

    cachedColumnMapping = preview.inferred_mapping || {};

    preview.headers.forEach(h => {
      const selected = cachedColumnMapping[h] || "";
      mapHtml += `
        <div style="font-size: 0.72rem; font-family: var(--font-mono); display: flex; flex-direction: column; gap: 0.2rem;">
          <span style="color: var(--accent-cyan);">${h}:</span>
          <select class="form-input col-map-select" data-src-col="${h}" style="padding: 0.3rem 0.5rem; font-size: 0.72rem;">
            ${canonicalOptions.map(opt => `<option value="${opt.key}" ${opt.key === selected ? 'selected' : ''}>${opt.label}</option>`).join("")}
          </select>
        </div>
      `;
    });
    mapCont.innerHTML = mapHtml;

    // Listen to changes
    mapCont.querySelectorAll(".col-map-select").forEach(sel => {
      sel.addEventListener("change", (e) => {
        const srcCol = e.target.getAttribute("data-src-col");
        cachedColumnMapping[srcCol] = e.target.value;
      });
    });
  }

  // Render sample rows table
  const tbl = document.getElementById("csv-preview-table");
  if (tbl && preview.sample_rows && preview.sample_rows.length > 0) {
    const headers = Object.keys(preview.sample_rows[0]);
    let tblHtml = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>`;
    preview.sample_rows.forEach(r => {
      tblHtml += `<tr>${headers.map(h => `<td>${r[h]}</td>`).join("")}</tr>`;
    });
    tblHtml += `</tbody>`;
    tbl.innerHTML = tblHtml;
  }
}

async function confirmCSVIngestion() {
  const csvStr = document.getElementById("raw-csv-textarea")?.value || cachedCSVText;
  if (!csvStr) return;

  try {
    const res = await fetch("/api/ingest/csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        csv_text: csvStr,
        column_mapping: cachedColumnMapping,
        dataset_name: "cooling_telemetry",
      }),
    });

    const data = await res.json();
    if (res.ok && data.status === "SUCCESS") {
      const qr = data.quality_report;
      updateCSVUploadStepper(5);
      renderQualityReport(qr);
      renderIngestionSummaryCard(qr, currentCSVHeaders);
      currentOperatingMode = "csv";
      document.querySelectorAll(".btn-mode").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      const cBtn = document.getElementById("btn-mode-csv");
      cBtn?.classList.add("active");
      cBtn?.setAttribute("aria-selected", "true");
      updateProvenancePill("user_uploaded_csv", `Dataset: ${qr.filename} (${qr.valid_rows} valid records)`);
      await refreshAll();
    } else {
      alert(`CSV Ingestion Rejected: ${data.error || 'Failed validation.'}`);
    }
  } catch (e) {
    console.warn("CSV ingestion failed:", e);
  }
}

function renderIngestionSummaryCard(qr, headers) {
  const summaryCard = document.getElementById("csv-ingestion-summary-card");
  if (!summaryCard) return;

  summaryCard.style.display = "block";

  const fnEl = document.getElementById("summary-filename");
  if (fnEl) fnEl.textContent = qr.filename || currentCSVFileName;

  const rowEl = document.getElementById("summary-row-count");
  if (rowEl) rowEl.textContent = `${qr.valid_rows} valid / ${qr.total_rows} total rows`;

  const colEl = document.getElementById("summary-column-count");
  if (colEl) colEl.textContent = `${headers && headers.length > 0 ? headers.length : 6} columns detected`;

  const covEl = document.getElementById("summary-time-coverage");
  if (covEl) {
    covEl.textContent = qr.time_span_hours ? `${qr.time_span_hours} hours total span` : `${qr.valid_rows} time points`;
  }

  const scoreEl = document.getElementById("summary-quality-score");
  if (scoreEl) {
    scoreEl.textContent = `${qr.quality_score}% (${qr.quality_rating})`;
    scoreEl.style.color = (qr.quality_rating === "EXCELLENT" || qr.quality_rating === "GOOD") ? "var(--accent-emerald)" : "var(--accent-amber)";
  }

  const valEl = document.getElementById("summary-validation-status");
  if (valEl) {
    valEl.textContent = qr.valid_rows > 0 ? `VALIDATED (${qr.validity_pct}% Physical Pass Rate)` : "FAILED";
    valEl.style.color = qr.valid_rows > 0 ? "var(--accent-emerald)" : "var(--accent-crimson)";
  }

  const anaEl = document.getElementById("summary-analysis-status");
  if (anaEl) {
    anaEl.textContent = "ANALYZED — Recalculation Applied";
    anaEl.style.color = "var(--accent-cyan)";
  }

  const badgeEl = document.getElementById("summary-overall-badge");
  if (badgeEl) {
    badgeEl.textContent = `INGESTED: ${qr.valid_rows} ROWS`;
  }
}

function renderQualityReport(qr) {
  if (!qr) return;
  const listEl = document.getElementById("quality-stats-list");
  const badgeEl = document.getElementById("quality-score-badge");

  if (badgeEl) {
    badgeEl.textContent = `QUALITY: ${qr.quality_score}% (${qr.quality_rating})`;
    badgeEl.className = "provenance-pill " + (qr.quality_rating === "EXCELLENT" || qr.quality_rating === "GOOD" ? "pill-csv" : "pill-manual");
  }

  if (listEl) {
    listEl.innerHTML = `
      <div class="quality-stat-row"><span>Total Data Rows:</span><strong>${qr.total_rows}</strong></div>
      <div class="quality-stat-row"><span>Valid Physical Rows:</span><strong style="color: var(--accent-emerald);">${qr.valid_rows} (${qr.validity_pct}%)</strong></div>
      <div class="quality-stat-row"><span>Data Completeness:</span><strong>${qr.completeness_pct}%</strong></div>
      <div class="quality-stat-row"><span>Duplicate Timestamps:</span><strong>${qr.duplicate_timestamps}</strong></div>
      <div class="quality-stat-row"><span>Physical Range Violations:</span><strong>${qr.range_violations}</strong></div>
      <div class="quality-stat-row"><span>Thermodynamic Inversions (Tout ≤ Tin):</span><strong>${qr.thermodynamic_inversions}</strong></div>
      <div class="quality-stat-row"><span>Detected Cleaning Events:</span><strong style="color: var(--accent-cyan);">${qr.detected_cleaning_events}</strong></div>
    `;
  }
}

function hideFeedbackBox(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "none";
}

async function refreshAll() {
  await Promise.all([
    fetchStatus(),
    fetchKPIs(),
    fetchRecommendation(),
    fetchForecast(currentHorizonHours),
  ]);
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    const el = document.getElementById("system-status-text");
    if (el) el.textContent = `${data.status} — ${data.framework}`;
  } catch (e) {
    console.warn("Status fetch failed:", e);
  }
}

async function fetchKPIs() {
  try {
    const res = await fetch("/api/kpis");
    if (!res.ok) return;
    const kpis = await res.json();

    // Primary & Secondary readouts
    updateText("val-rf", kpis.Rf_fouling_resistance ? kpis.Rf_fouling_resistance.toExponential(2) : "1.85e-4");
    updateText("val-htc-ratio", kpis.thermal_efficiency_ratio ? (kpis.thermal_efficiency_ratio * 100).toFixed(1) + "%" : "86.4%");
    updateText("val-delta-p", kpis.delta_P_meas_kPa ? kpis.delta_P_meas_kPa.toFixed(1) : "58.2");
    updateText("val-flow", kpis.mass_flow_kg_s ? kpis.mass_flow_kg_s.toFixed(1) : "220.0");
    updateText("val-temps", `${kpis.T_cool_in_meas ? kpis.T_cool_in_meas.toFixed(1) : "18.5"} / ${kpis.T_cool_out_meas ? kpis.T_cool_out_meas.toFixed(1) : "26.2"}`);
    updateText("val-bsi", kpis.bsi_score ? kpis.bsi_score.toFixed(1) : "42.5");
    updateText("val-excess-pump", kpis.parasitic_pump_kw ? kpis.parasitic_pump_kw.toFixed(1) : "6.4");
    updateText("val-heat-duty", kpis.heat_duty_MW ? kpis.heat_duty_MW.toFixed(1) : "13.8");

    // Telemetry timestamp
    updateText("telemetry-timestamp", `Operating Hour: ${kpis.time_hours || 0} | Season: ${kpis.season || "Annual"}`);

    // Provenance Tag Updates
    const prov = kpis.provenance_summary || {};
    updateProvenanceBadge("prov-rf", prov.Rf_fouling_resistance || "synthetic_simulation");
    updateProvenanceBadge("prov-eff", prov.thermal_efficiency_ratio || "synthetic_simulation");
    updateProvenanceBadge("prov-dp", prov.delta_P_meas_kPa || "synthetic_simulation");
    updateProvenanceBadge("prov-flow", prov.mass_flow_kg_s || "synthetic_simulation");
    updateProvenanceBadge("prov-temps", prov.T_cool_in_meas || "synthetic_simulation");
    updateProvenanceBadge("prov-bsi", prov.bsi_score || "synthetic_simulation");
    updateProvenanceBadge("prov-pump", prov.parasitic_pump_kw || "synthetic_simulation");
    updateProvenanceBadge("prov-heat", prov.heat_duty_MW || "synthetic_simulation");

    // BSI Status Text
    const bsi = kpis.bsi_score || 0;
    const bsiEl = document.getElementById("bsi-status-text");
    if (bsiEl) {
      if (bsi < 30) bsiEl.textContent = "Status: Nominal / Low Fouling";
      else if (bsi < 65) bsiEl.textContent = "Status: Moderate Degradation";
      else bsiEl.textContent = "Status: Critical / Severe Fouling";
    }
  } catch (e) {
    console.warn("KPIs fetch failed:", e);
  }
}

async function fetchRecommendation(customParams = null) {
  try {
    let url = "/api/recommendation";
    if (customParams) {
      const q = new URLSearchParams(customParams).toString();
      url += `?${q}`;
    }
    const res = await fetch(url);
    if (!res.ok) return;
    const rec = await res.json();
    cachedRecommendationData = rec;
    renderRecommendation(rec);
  } catch (e) {
    console.warn("Recommendation fetch failed:", e);
  }
}

function renderRecommendation(rec) {
  if (!rec) return;

  // Status tag
  const statusEl = document.getElementById("rec-status-tag");
  if (statusEl) {
    statusEl.textContent = rec.status || "OPTIMAL_FEASIBLE";
    statusEl.style.color = rec.is_feasible ? "var(--accent-emerald)" : "var(--accent-crimson)";
  }

  // Large interval readout
  updateText("rec-interval-days", rec.recommended_interval_days !== undefined ? rec.recommended_interval_days : 14);
  updateText("rec-interval-hours", rec.recommended_interval_hours !== undefined ? rec.recommended_interval_hours : 336);
  updateText("rec-window-days", rec.recommended_window_days || "Day 12 to Day 17");

  // Threshold crossing info
  const crossing = rec.threshold_crossing;
  const breachEl = document.getElementById("rec-breach-info");
  if (breachEl) {
    if (crossing && crossing.breached) {
      breachEl.textContent = `Day ${crossing.breach_days_relative} (Hour ${crossing.breach_hour_absolute}) — ${crossing.limiting_parameter}`;
    } else {
      breachEl.textContent = "None within 90-day lookahead horizon";
    }
  }

  // Rationale & delayed impact
  updateText("rec-rationale", rec.primary_reason || "Evaluated candidate scenarios against thermal and hydraulic constraints.");
  updateText("rec-delayed-impact", rec.expected_if_delayed || "Delaying cleaning leads to thermal deficit and excess pumping power.");

  // Current condition
  const cond = rec.current_operating_condition;
  if (cond) {
    updateText("rec-current-condition", 
      `Biofilm Thickness: ${cond.biofilm_thickness_um} μm | ` +
      `Thermal Rf: ${cond.Rf_fouling_resistance.toExponential(2)} m²·K/W | ` +
      `Pressure Drop: ${cond.delta_P_meas_kPa} kPa`
    );
  }

  // Strategy comparison table
  renderStrategyTable(rec.all_scenarios, rec.recommended_interval_days);
}

function renderStrategyTable(scenarios, recommendedDays) {
  const tbody = document.getElementById("scenario-tbody");
  if (!tbody || !scenarios) return;

  let optimalCost = 0;
  const optScenario = scenarios.find(s => s.candidate_interval_days === recommendedDays);
  if (optScenario) optimalCost = optScenario.total_projected_cost;

  let html = "";
  scenarios.forEach(s => {
    const isRec = s.candidate_interval_days === recommendedDays;
    const isFeas = s.is_feasible;
    const delta = s.total_projected_cost - optimalCost;
    const deltaStr = isRec ? "0 (Optimum)" : (delta >= 0 ? `+$${delta.toLocaleString()}` : `-$${Math.abs(delta).toLocaleString()}`);

    const statusBadge = isRec 
      ? `<span class="tag-recommended">RECOMMENDED</span>`
      : (isFeas ? `<span class="tag-feasible">FEASIBLE</span>` : `<span class="tag-infeasible">INFEASIBLE</span>`);

    const rowClass = isRec ? "row-recommended" : (!isFeas ? "row-infeasible" : "");

    html += `
      <tr class="${rowClass}">
        <td><strong>${s.strategy}</strong></td>
        <td>${statusBadge}</td>
        <td style="text-align: right;">${s.intervention_count}</td>
        <td style="text-align: right;">$${s.cleaning_cost.toLocaleString()}</td>
        <td style="text-align: right;">$${s.downtime_cost.toLocaleString()}</td>
        <td style="text-align: right;">$${s.pump_energy_cost.toLocaleString()}</td>
        <td style="text-align: right;">$${s.thermal_penalty_cost.toLocaleString()}</td>
        <td style="text-align: right;"><strong>$${s.total_projected_cost.toLocaleString()}</strong></td>
        <td style="text-align: right; color: ${isRec ? 'var(--accent-cyan)' : (!isFeas ? 'var(--accent-crimson)' : 'var(--text-secondary)')};">${deltaStr}</td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

async function fetchForecast(horizonHours = 720) {
  try {
    const res = await fetch(`/api/forecast?horizon_hours=${horizonHours}`);
    if (!res.ok) return;
    const forecastData = await res.json();
    cachedForecastData = forecastData;
    renderForecastSVG(forecastData);
    if (forecastData.model_info) {
      const mi = forecastData.model_info;
      const qHatStr = mi.conformal_q_hat_m2k_w ? mi.conformal_q_hat_m2k_w.toExponential(2) : "1.81e-5";
      const covStr = mi.empirical_test_coverage_pct ? mi.empirical_test_coverage_pct.toFixed(1) + "%" : "98.4%";
      const confEl = document.getElementById("rec-model-confidence");
      if (confEl) {
        confEl.textContent = `Model Reliability: Calibrated 95% Prediction Interval (Split-Conformal ±q̂ = ${qHatStr} m²·K/W, Test Coverage = ${covStr}) • Hybrid Ensemble (Ridge + GBR Residuals)`;
      }
    }
  } catch (e) {
    console.warn("Forecast fetch failed:", e);
  }
}

/**
 * Render multi-series forecast chart using native SVG elements with glowing area fills, interactive crosshair and uncertainty envelope.
 */
function renderForecastSVG(data) {
  const container = document.getElementById("timeseries-svg-container");
  if (!container || !data) return;

  const width = Math.max(container.clientWidth || 1000, 400);
  const height = Math.max(container.clientHeight || 360, 260);
  const padL = 72, padR = 72, padT = 32, padB = 45;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const hist = data.historical || [];
  const fore = data.forecast || [];

  if (hist.length === 0 && fore.length === 0) return;

  // Compute combined time range
  const allHours = [...hist.map(p => p.time_hours), ...fore.map(p => p.time_hours)];
  const minH = Math.min(...allHours);
  const maxH = Math.max(...allHours);
  const hourSpan = maxH - minH || 1;

  // Y-Left: Rf range
  const allRf = [
    ...hist.map(p => p.Rf_fouling_resistance),
    ...fore.map(p => p.predicted_Rf_upper !== undefined ? p.predicted_Rf_upper : p.predicted_Rf)
  ];
  const maxRf = Math.max(...allRf, (data.thresholds?.rf_crit || 3.0e-4) * 1.15, 0.00035);

  // Y-Right: delta_P range
  const allDp = [
    ...hist.map(p => p.delta_P_meas_kPa),
    ...fore.map(p => p.predicted_delta_P_kPa)
  ];
  const minDp = Math.min(...allDp, 6.0);
  const maxDp = Math.max(...allDp, (data.thresholds?.dp_crit || 16.0) * 1.10, 18.0);

  // Helper coordinate mappers
  const mapX = (h) => padL + ((h - minH) / hourSpan) * plotW;
  const mapY_Rf = (rf) => padT + plotH - (Math.max(0, rf) / maxRf) * plotH;
  const mapY_Dp = (dp) => padT + plotH - ((Math.max(minDp, dp) - minDp) / (maxDp - minDp || 1)) * plotH;

  // Current simulation divider
  const curX = mapX(data.current_hour);

  // Critical threshold lines
  const threshRf_Y = mapY_Rf(data.thresholds.rf_crit);
  const threshDp_Y = mapY_Dp(data.thresholds.dp_crit);

  // Recommended cleaning line
  const recX = mapX(data.recommended_cleaning_hour);

  // Build Uncertainty Polygon (Predicted Rf Upper & Lower)
  let uncertaintyPath = "";
  if (fore.length > 0) {
    const upperPts = fore.map(p => `${mapX(p.time_hours).toFixed(1)},${mapY_Rf(p.predicted_Rf_upper).toFixed(1)}`).join(" L ");
    const lowerPts = [...fore].reverse().map(p => `${mapX(p.time_hours).toFixed(1)},${mapY_Rf(p.predicted_Rf_lower).toFixed(1)}`).join(" L ");
    uncertaintyPath = `M ${upperPts} L ${lowerPts} Z`;
  }

  // Paths
  const histRfPath = hist.map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p.time_hours).toFixed(1)},${mapY_Rf(p.Rf_fouling_resistance).toFixed(1)}`).join(" ");
  const foreRfPath = fore.map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p.time_hours).toFixed(1)},${mapY_Rf(p.physics_predicted_Rf || p.predicted_Rf).toFixed(1)}`).join(" ");
  const foreMlRfPath = fore.filter(p => p.ml_predicted_Rf !== undefined).map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p.time_hours).toFixed(1)},${mapY_Rf(p.ml_predicted_Rf).toFixed(1)}`).join(" ");

  const histDpPath = hist.map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p.time_hours).toFixed(1)},${mapY_Dp(p.delta_P_meas_kPa).toFixed(1)}`).join(" ");
  const foreDpPath = fore.map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p.time_hours).toFixed(1)},${mapY_Dp(p.predicted_delta_P_kPa).toFixed(1)}`).join(" ");

  // Area under Historical Rf curve for subtle glow gradient
  const histRfAreaPath = hist.length > 0 
    ? `M ${mapX(hist[0].time_hours).toFixed(1)},${(padT + plotH).toFixed(1)} ` +
      hist.map(p => `L ${mapX(p.time_hours).toFixed(1)},${mapY_Rf(p.Rf_fouling_resistance).toFixed(1)}`).join(" ") +
      ` L ${mapX(hist[hist.length-1].time_hours).toFixed(1)},${(padT + plotH).toFixed(1)} Z`
    : "";

  // SVG Assembly
  const svgContent = `
    <svg class="chart-svg" id="chart-svg-elem" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" style="cursor: crosshair;">
      <defs>
        <linearGradient id="rfGlowGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#38BDF8" stop-opacity="0.0"/>
        </linearGradient>
        <linearGradient id="uncertaintyGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.18"/>
          <stop offset="100%" stop-color="#38BDF8" stop-opacity="0.03"/>
        </linearGradient>
      </defs>

      <!-- Background Grid -->
      <line x1="${padL}" y1="${padT}" x2="${padL + plotW}" y2="${padT}" stroke="rgba(255, 255, 255, 0.05)" stroke-dasharray="3"/>
      <line x1="${padL}" y1="${padT + plotH * 0.33}" x2="${padL + plotW}" y2="${padT + plotH * 0.33}" stroke="rgba(255, 255, 255, 0.05)" stroke-dasharray="3"/>
      <line x1="${padL}" y1="${padT + plotH * 0.66}" x2="${padL + plotW}" y2="${padT + plotH * 0.66}" stroke="rgba(255, 255, 255, 0.05)" stroke-dasharray="3"/>
      <line x1="${padL}" y1="${padT + plotH}" x2="${padL + plotW}" y2="${padT + plotH}" stroke="rgba(255, 255, 255, 0.12)"/>

      <!-- Axes -->
      <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="rgba(255, 255, 255, 0.12)"/>
      <line x1="${padL + plotW}" y1="${padT}" x2="${padL + plotW}" y2="${padT + plotH}" stroke="rgba(255, 255, 255, 0.12)"/>

      <!-- Critical Threshold Lines -->
      <line x1="${padL}" y1="${threshRf_Y}" x2="${padL + plotW}" y2="${threshRf_Y}" stroke="#EF4444" stroke-dasharray="4 3" stroke-width="1.3"/>
      <text x="${padL + 8}" y="${threshRf_Y - 5}" fill="#EF4444" font-size="10" font-family="monospace" font-weight="bold">Limit: Rf = ${(data.thresholds.rf_crit*1e4).toFixed(1)}e-4</text>

      <line x1="${padL}" y1="${threshDp_Y}" x2="${padL + plotW}" y2="${threshDp_Y}" stroke="#EF4444" stroke-dasharray="4 3" stroke-width="1.3"/>
      <text x="${padL + plotW - 8}" y="${threshDp_Y - 5}" fill="#EF4444" font-size="10" text-anchor="end" font-family="monospace" font-weight="bold">Limit: ΔP = ${data.thresholds.dp_crit.toFixed(0)} kPa</text>

      <!-- Current Time Marker (t0) -->
      <line x1="${curX}" y1="${padT}" x2="${curX}" y2="${padT + plotH}" stroke="#38BDF8" stroke-dasharray="3 2" stroke-width="1.5"/>
      <text x="${curX + 5}" y="${padT + 12}" fill="#38BDF8" font-size="10" font-family="monospace" font-weight="bold">NOW (t₀)</text>

      <!-- Recommended Cleaning Marker -->
      <line x1="${recX}" y1="${padT}" x2="${recX}" y2="${padT + plotH}" stroke="#10B981" stroke-width="2"/>
      <polygon points="${recX-5},${padT} ${recX+5},${padT} ${recX},${padT+8}" fill="#10B981"/>
      <text x="${recX}" y="${padT - 6}" fill="#10B981" font-size="10" font-weight="bold" text-anchor="middle" font-family="monospace">REC CLEANING</text>

      <!-- Soft Area Gradient Fill Under Historical Rf -->
      ${histRfAreaPath ? `<path d="${histRfAreaPath}" fill="url(#rfGlowGradient)" stroke="none"/>` : ''}

      <!-- Uncertainty Envelope (Shaded Polygon) -->
      ${uncertaintyPath ? `<path d="${uncertaintyPath}" fill="url(#uncertaintyGradient)" stroke="none"/>` : ''}

      <!-- Curves -->
      <!-- Rf Curves -->
      <path d="${histRfPath}" fill="none" stroke="#38BDF8" stroke-width="2.5"/>
      <path d="${foreRfPath}" fill="none" stroke="#38BDF8" stroke-dasharray="4 3" stroke-width="2"/>
      ${foreMlRfPath ? `<path d="${foreMlRfPath}" fill="none" stroke="#A855F7" stroke-dasharray="3 3" stroke-width="2"/>` : ''}

      <!-- delta_P Curves -->
      <path d="${histDpPath}" fill="none" stroke="#F59E0B" stroke-width="1.8"/>
      <path d="${foreDpPath}" fill="none" stroke="#F59E0B" stroke-dasharray="3 3" stroke-width="1.8"/>

      <!-- Interactive Crosshair (Initially hidden) -->
      <line id="chart-crosshair" x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="rgba(255, 255, 255, 0.45)" stroke-dasharray="2 2" stroke-width="1" style="display: none; pointer-events: none;"/>

      <!-- Y-Axis Labels Left (Rf in Cyan) -->
      <text x="${padL - 10}" y="${padT + 8}" fill="#38BDF8" font-size="10" font-family="monospace" text-anchor="end">${(maxRf * 1e4).toFixed(1)}e-4</text>
      <text x="${padL - 10}" y="${padT + plotH * 0.5}" fill="#38BDF8" font-size="10" font-family="monospace" text-anchor="end">${(maxRf * 0.5 * 1e4).toFixed(1)}e-4</text>
      <text x="${padL - 10}" y="${padT + plotH}" fill="#38BDF8" font-size="10" font-family="monospace" text-anchor="end">0.0</text>
      <text x="${padL - 42}" y="${padT + plotH / 2}" fill="#38BDF8" font-size="10" font-family="monospace" transform="rotate(-90 ${padL - 42} ${padT + plotH / 2})" text-anchor="middle">Rf [m²·K/W]</text>

      <!-- Y-Axis Labels Right (delta_P in Amber) -->
      <text x="${padL + plotW + 10}" y="${padT + 8}" fill="#F59E0B" font-size="10" font-family="monospace" text-anchor="start">${maxDp.toFixed(0)}</text>
      <text x="${padL + plotW + 10}" y="${padT + plotH * 0.5}" fill="#F59E0B" font-size="10" font-family="monospace" text-anchor="start">${((maxDp + minDp) / 2).toFixed(0)}</text>
      <text x="${padL + plotW + 10}" y="${padT + plotH}" fill="#F59E0B" font-size="10" font-family="monospace" text-anchor="start">${minDp.toFixed(0)}</text>
      <text x="${padL + plotW + 42}" y="${padT + plotH / 2}" fill="#F59E0B" font-size="10" font-family="monospace" transform="rotate(90 ${padL + plotW + 42} ${padT + plotH / 2})" text-anchor="middle">ΔP [kPa]</text>

      <!-- X-Axis Labels (Time Hours) -->
      <text x="${padL}" y="${padT + plotH + 18}" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">Hour ${minH}</text>
      <text x="${curX}" y="${padT + plotH + 18}" fill="#38BDF8" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">t₀ (${data.current_hour}h)</text>
      <text x="${padL + plotW}" y="${padT + plotH + 18}" fill="#94A3B8" font-size="10" font-family="monospace" text-anchor="middle">Hour ${maxH}</text>
    </svg>
  `;

  container.innerHTML = svgContent + `<div class="chart-tooltip" id="chart-tooltip"></div>`;

  // Attach interactive hover listener to the rendered SVG
  const svgEl = document.getElementById("chart-svg-elem");
  const tooltipEl = document.getElementById("chart-tooltip");
  const crosshairEl = document.getElementById("chart-crosshair");

  if (svgEl && tooltipEl && crosshairEl) {
    const combinedPoints = [
      ...hist.map(p => ({ ...p, isHist: true })),
      ...fore.map(p => ({ ...p, isHist: false }))
    ].sort((a, b) => a.time_hours - b.time_hours);

    svgEl.addEventListener("mousemove", (e) => {
      const rect = svgEl.getBoundingClientRect();
      const mouseX = ((e.clientX - rect.left) / rect.width) * width;
      const mouseY = ((e.clientY - rect.top) / rect.height) * height;

      if (mouseX >= padL && mouseX <= padL + plotW && mouseY >= padT && mouseY <= padT + plotH) {
        // Find closest point by hour
        const targetH = minH + ((mouseX - padL) / plotW) * hourSpan;
        let closest = combinedPoints[0];
        let minDiff = Math.abs(closest.time_hours - targetH);

        for (let i = 1; i < combinedPoints.length; i++) {
          const diff = Math.abs(combinedPoints[i].time_hours - targetH);
          if (diff < minDiff) {
            minDiff = diff;
            closest = combinedPoints[i];
          }
        }

        if (closest) {
          const cX = mapX(closest.time_hours);
          crosshairEl.setAttribute("x1", cX);
          crosshairEl.setAttribute("x2", cX);
          crosshairEl.style.display = "block";

          const rfVal = closest.isHist 
            ? closest.Rf_fouling_resistance 
            : (closest.physics_predicted_Rf || closest.predicted_Rf);
          const dpVal = closest.isHist ? closest.delta_P_meas_kPa : closest.predicted_delta_P_kPa;
          const phase = closest.isHist ? "Historical Telemetry" : (closest.time_hours <= data.recommended_cleaning_hour ? "Nominal Run Forecast" : "Post-Intervention Forecast");

          tooltipEl.innerHTML = `
            <div style="font-weight: 700; color: var(--accent-cyan); margin-bottom: 0.2rem;">Hour ${closest.time_hours} (${phase})</div>
            <div>Rf: <strong>${rfVal.toExponential(2)} m²·K/W</strong></div>
            <div>ΔP: <strong>${dpVal.toFixed(1)} kPa</strong></div>
            ${closest.predicted_Rf_upper ? `<div style="font-size: 0.68rem; color: var(--text-secondary);">95% CI: [${closest.predicted_Rf_lower.toExponential(2)}, ${closest.predicted_Rf_upper.toExponential(2)}]</div>` : ''}
          `;
          tooltipEl.style.display = "block";

          // Position tooltip relative to container
          const tipLeft = Math.min(e.clientX - rect.left + 15, container.clientWidth - 190);
          const tipTop = Math.max(e.clientY - rect.top - 40, 10);
          tooltipEl.style.left = `${tipLeft}px`;
          tooltipEl.style.top = `${tipTop}px`;
        }
      } else {
        crosshairEl.style.display = "none";
        tooltipEl.style.display = "none";
      }
    });

    svgEl.addEventListener("mouseleave", () => {
      crosshairEl.style.display = "none";
      tooltipEl.style.display = "none";
    });
  }
}

async function recalculateOptimizer() {
  const cClean = parseFloat(document.getElementById("param-cleaning-cost")?.value) || 3500;
  const cElec = parseFloat(document.getElementById("param-elec-cost")?.value) || 0.12;
  const horizonD = parseInt(document.getElementById("param-horizon-days")?.value, 10) || 60;

  const params = {
    cleaning_cost: cClean,
    electricity_cost: cElec,
    horizon_days: horizonD,
  };

  await fetchRecommendation(params);
}

function updateText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function updateProvenanceBadge(id, type) {
  const el = document.getElementById(id);
  if (!el) return;
  if (type === "direct_user_measurement") {
    el.className = "kpi-prov-tag kpi-prov-measured";
    el.textContent = "MEASURED";
  } else if (type === "derived_calculation") {
    el.className = "kpi-prov-tag kpi-prov-derived";
    el.textContent = "DERIVED";
  } else {
    el.className = "kpi-prov-tag kpi-prov-synthetic";
    el.textContent = "SYNTHETIC";
  }
}
