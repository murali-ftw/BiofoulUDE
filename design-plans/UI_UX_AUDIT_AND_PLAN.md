# Biofouling AI Monitor — UI/UX Audit & Redesign Implementation Plan

**Project:** Nuclear Auxiliary Heat-Exchanger Biofouling Monitor & Predictive Cleaning Optimizer  
**Location:** `workspace/biofouling_ai_repaired/`  
**Date:** 2026-11-18  
**Design Lead / Engineering Role:** Antigravity AI UI/UX Engineering Lead  
**Governing Skills:** `improve-ui`, `ui-ux-pro-max`, `frontend-design`

---

## 1. Executive Summary & Design Scope

The Biofouling AI Heat Exchanger decision-support application is a specialized engineering telemetry and predictive maintenance platform. It models biofilm growth kinetics (Ratkowsky cardinal model), continuum hydraulics (Darcy-Weisbach & Haaland), thermal fouling resistance ($R_f$), and condition-based cleaning optimization (CBO) across three operating modes: **Simulation Mode**, **Manual Input Mode**, and **Historical CSV Upload Mode**.

The existing frontend uses **vanilla HTML5, CSS3, and JavaScript** with an integrated zero-install Python HTTP server (`dashboard_server.py`) delivering native SVG time-series charts, KPI displays, and interactive scenario matrices.

This plan details the UI/UX audit findings and establishes an actionable, non-destructive redesign that elevates the interface into a **world-class nuclear & industrial telemetry console** while strictly preserving 100% of the underlying backend APIs, calculations, and data contracts.

---

## 2. UI/UX Audit Findings (`improve-ui` Protocol)

### Finding 1: Unstructured KPI Grid & Cognitive Load
- **Current State:** 8 KPI cards are arranged in an unvaried 4×2 grid. There is no visual grouping between thermal metrics ($R_f$, $U/U_0$), hydraulic metrics ($\Delta P$, mass flow), operational state ($T_{in}/T_{out}$, Heat Duty), and composite risk indicators (BSI, Parasitic Pump kW).
- **Usability Impact:** Operators scanning for thermal limits cannot distinguish primary constraint variables from secondary operational telemetry at a glance.
- **Solution:** Group metrics into distinct telemetry clusters with subtle group headers and colored indicator pips:
  1. *Thermal Degradation & Resistance* ($R_f$, $U/U_0$, Heat Duty)
  2. *Hydraulic Dynamics & Power* ($\Delta P$, Mass Flow, Parasitic Pump kW)
  3. *Thermal Boundary & Severity* (Coolant $T_{in}/T_{out}$, BSI Score with calibrated color stages)

### Finding 2: Recommendation Panel Visual Hierarchy
- **Current State:** Section C places the recommended interval in a callout, but the threshold breach warning, recommended window, and economic trade-off rationale have uneven visual weight. The model confidence indicator is a small footnote.
- **Usability Impact:** When an operational threshold breach is predicted, it should immediately command attention with an industrial warning/critical banner without obscuring the economic optimum.
- **Solution:** Restructure the recommendation header into an industrial dispatch card with a high-contrast interval counter, distinct feasibility badge, split window/breach cards, and structured rationale breakdown.

### Finding 3: Typography & Tabular Figure Alignment
- **Current State:** Monospace font is used sporadically (`var(--font-mono)`), but standard sans-serif numbers without `tabular-nums` cause jitter during refresh and misalignment in the Scenario Comparison table.
- **Usability Impact:** Scientific notation (e.g. $1.85 \times 10^{-4}$ vs $3.00 \times 10^{-4}$ $\text{m}^2\cdot\text{K/W}$) and financial values ($\$41,505$ vs $\$44,489$) are harder to compare rapidly.
- **Solution:** Enforce `font-variant-numeric: tabular-nums` across all metric readouts, table cells, and chart coordinates. Standardize typography scale: Inter / Segoe UI for UI structure, and JetBrains Mono / SF Mono for telemetry values.

### Finding 4: Mode Switcher & Ingestion Workflow Polish
- **Current State:** The 3 mode switcher buttons look like plain rectangular buttons. In Manual mode, 8 form fields are in a flat grid. In CSV mode, the 5-stage stepper and summary cards are functional but visual styling can be significantly sharpened.
- **Usability Impact:** Operating mode context is critical because single snapshots cannot infer transient growth rates. The active data provenance must be unmistakably clear.
- **Solution:** Style the mode switcher as a segmented industrial control bar. Group manual inputs into *Direct Measurements* vs *Calculated Overrides*. Refine the 5-stage stepper with animated pulse states and crisp status badges.

### Finding 5: Chart Vector Ergonomics & Legend Legibility
- **Current State:** The native SVG chart accurately renders dual Y-axes (left: $R_f$, right: $\Delta P$), shaded 95% conformal prediction intervals, and threshold limits. However, the horizontal legend items are dense, and the crosshair hover interaction can be made more responsive and intuitive.
- **Solution:** Enhance grid line subtleness, sharpen axes tick marks, group legend keys into *Thermal ($R_f$)*, *Hydraulic ($\Delta P$)*, and *Decision Boundaries*, and add clean hover focus highlights.

### Finding 6: Responsive Layout & Mobile/Tablet Scaling
- **Current State:** The 1540px container shrinks, but tables require better touch-friendly horizontal scroll indicators, and the bottom sustainability/tuning grid needs explicit breakpoints at 1024px and 768px.
- **Solution:** Implement a modern CSS grid layout with fluid breakpoints: 1 column on mobile (<768px), 2 columns on tablet (<1024px), 4 columns on desktop ($\ge 1024$px).

---

## 3. Chosen Design Direction: "Industrial Obsidian Engineering Console"

Using `ui-ux-pro-max` search results (`dark-mode-oled` + `data-dense-dashboard`):

### Color System & Semantic Tokens
```css
:root {
  /* Surface Layers */
  --bg-canvas: #07090E;         /* Deepest obsidian background */
  --bg-surface-1: #0D1117;      /* Primary container surface */
  --bg-surface-2: #131922;      /* Elevated panels & cards */
  --bg-surface-3: #1B2330;      /* Hover states & inset headers */
  
  /* Borders */
  --border-subtle: #1E2633;     /* Structural grid lines */
  --border-medium: #2B3545;     /* Card outlines */
  --border-accent: rgba(0, 229, 255, 0.35); /* Focus & highlight */
  
  /* Typography */
  --text-main: #F1F5F9;         /* Slate 100 - Primary readouts */
  --text-muted: #94A3B8;        /* Slate 400 - Secondary labels */
  --text-faint: #64748B;        /* Slate 500 - Units and helper notes */
  
  /* Semantic Telemetry Accents */
  --accent-cyan: #00E5FF;       /* Primary hydraulic & active telemetry */
  --accent-emerald: #10B981;    /* Optimal recommendation & compliant */
  --accent-amber: #F59E0B;      /* Differential pressure & warning */
  --accent-crimson: #EF4444;    /* Critical threshold breach & infeasible */
  --accent-purple: #A855F7;     /* ML Ensemble forecast & conformal bounds */
}
```

### Aesthetic Guidelines (`frontend-design` skill):
1. **No Generic AI Tells:** Zero purple-to-pink decorative gradients, zero excessive glowing card borders, zero bloated padding, zero empty space.
2. **Industrial Density:** Information-dense, crisp 1px borders, high-contrast typography, clear engineering units on every number.
3. **Strict Truth in Data:** Every metric labeled with exact physical units ($\text{m}^2\cdot\text{K/W}$, $\text{kPa}$, $\text{kg/s}$, $\text{MW}$, $\text{kW}$, $\text{t CO}_2$). Data provenance explicitly tagged (`MEASURED`, `DERIVED`, `SYNTHETIC`).

---

## 4. Implementation Plan & File Matrix

| File | Changes Planned | Constraints |
|---|---|---|
| `dashboard/index.html` | Refine semantic HTML structure; add telemetry group badges; upgrade mode selector into segmented bar; polish manual input field groupings; enhance recommendation header layout; add responsive table wrappers. | **Preserve all existing element IDs** used by `app.js` (`val-rf`, `val-delta-p`, `rec-interval-days`, `scenario-tbody`, `timeseries-svg-container`, etc.). |
| `dashboard/style.css` | Complete CSS redesign using Industrial Obsidian tokens; modern flex/grid architecture; tabular figure typography; responsive breakpoints at 1200px, 1024px, 768px, 480px; polished stepper and summary card states. | Keep self-contained vanilla CSS. Zero external frameworks. |
| `dashboard/app.js` | Enhance SVG chart rendering (sharper coordinate axes, cleaner uncertainty polygon, refined hover crosshair and tooltips); smooth transitions; keep all API endpoints and data bindings intact. | **Zero changes to API contracts or data structures.** |

---

## 5. Verification & Safety Protocol

1. **Unit & Integration Tests:** Run `python -m unittest discover -s tests -p 'test_*.py'` (70 tests must pass).
2. **Independent Acceptance Evaluator:** Run `run_evaluation.py --target-dir workspace/biofouling_ai_repaired` (25/25 tests must pass).
3. **Server Endpoint Check:** Start `dashboard_server.py` on test port, probe `/`, `/api/status`, `/api/kpis`, `/api/recommendation`, `/api/forecast`.
4. **Visual & Responsive Verification:** Verify on desktop (1440px), tablet (1024px), and mobile (375px).
5. **Frozen Evaluator SHA-256 Hashes:** Verify all 4 hashes remain unchanged.
