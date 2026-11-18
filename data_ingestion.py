#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
workspace/biofouling_ai_repaired/data_ingestion.py

External Operating Data Ingestion & Quality Validation Engine.
Supports:
  1. Manual operating condition snapshot ingestion with strict physical range validation.
  2. Historical CSV dataset upload, preview, flexible column mapping, and multi-factor quality scoring.
  3. Physical fouling indicator derivation without fabricating missing observations.
  4. Explicit provenance tracking (Direct Measurement, Derived, Physics Prediction, ML Prediction, Synthetic).
"""

import csv
from datetime import datetime
import io
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import config
import fouling_indicators
import sustainability_metrics

# Directory for storing locally ingested user datasets (never transmitted externally)
USER_UPLOADS_DIR = Path(__file__).parent / "data" / "user_uploads"
USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Physical Limits & Operational Boundaries for Validation
# =============================================================================
PHYSICAL_RANGES = {
    "T_cool_in_meas": {
        "min": 2.0,
        "max": 45.0,
        "unit": "°C",
        "description": "Cooling water intake temperature",
        "nominal": 18.5,
    },
    "T_cool_out_meas": {
        "min": 5.0,
        "max": 65.0,
        "unit": "°C",
        "description": "Cooling water discharge temperature",
        "nominal": 26.2,
    },
    "mass_flow_cool_meas": {
        "min": 20.0,
        "max": 500.0,
        "unit": "kg/s",
        "description": "Cooling water mass flow rate",
        "nominal": 220.0,
    },
    "delta_P_meas_kPa": {
        "min": 1.0,
        "max": 100.0,
        "unit": "kPa",
        "description": "Tube-bundle differential pressure drop",
        "nominal": 7.91,
    },
    "time_hours": {
        "min": 0.0,
        "max": 200000.0,
        "unit": "hours",
        "description": "Cumulative operating hours",
        "nominal": 1000.0,
    },
    "heat_duty_MW": {
        "min": 0.1,
        "max": 60.0,
        "unit": "MW",
        "description": "Exchanged thermal heat duty",
        "nominal": 13.8,
    },
    "U_overall_meas": {
        "min": 100.0,
        "max": 3500.0,
        "unit": "W/(m²·K)",
        "description": "Overall heat transfer coefficient",
        "nominal": 2200.0,
    },
}

# Standard canonical column aliases for automatic mapping
COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp": [
        "timestamp", "time", "date", "datetime", "date_time", "record_time", "utc_time", "ts",
    ],
    "time_hours": [
        "time_hours", "hours", "hour", "operating_hours", "op_hours", "operating_time_h",
        "sim_h", "cum_hours", "time_h", "operating_time", "cum_time_hours",
    ],
    "T_cool_in_meas": [
        "t_cool_in_meas", "t_in", "tin", "t_cool_in", "inlet_temp", "inlet_temperature",
        "water_inlet_c", "temp_in", "temp_in_c", "t_in_c", "tin_c", "cooling_inlet_temp",
        "inlet_water_temp", "t_cool_inlet", "t1", "t_inlet", "t_cool_in_c",
    ],
    "T_cool_out_meas": [
        "t_cool_out_meas", "t_out", "tout", "t_cool_out", "outlet_temp", "outlet_temperature",
        "water_outlet_c", "temp_out", "temp_out_c", "t_out_c", "tout_c", "cooling_outlet_temp",
        "outlet_water_temp", "t_cool_outlet", "t2", "t_outlet", "t_cool_out_c",
    ],
    "mass_flow_cool_meas": [
        "mass_flow_cool_meas", "flow", "mass_flow", "flow_rate", "cooling_flow", "m_dot",
        "mass_flow_kg_s", "flow_kg_s", "cooling_water_mass_flow_meas", "cooling_water_flow",
        "water_flow", "mass_flow_rate", "coolant_flow", "m_flow", "mflow", "flow_rate_kg_s",
    ],
    "delta_P_meas_kPa": [
        "delta_p_meas_kpa", "delta_p", "dp", "dp_kpa", "pressure_drop", "diff_pressure_kpa",
        "delta_p_kpa", "dp_meas", "pressure_difference", "differential_pressure", "delta_pressure",
        "pressure_drop_kpa", "diff_pressure", "dp_loss_kpa",
    ],
    "heat_duty_MW": [
        "heat_duty_mw", "heat_duty", "duty_mw", "q_mw", "thermal_duty_mw", "heat_load_mw",
        "thermal_duty", "heat_load", "q_thermal", "q_dot", "heat_exchange_mw",
    ],
    "U_overall_meas": [
        "u_overall_meas", "u_overall", "u_meas", "overall_htc", "u_w_m2k", "htc_meas",
        "heat_transfer_coeff", "overall_u", "htc", "u_value", "u_overall_w_m2k",
    ],
    "cleaning_event": [
        "cleaning_event", "cleaning", "cleaned", "clean_flag", "is_cleaned", "cleaning_action",
        "wash_event", "clean_event", "maintenance_cleaning",
    ],
}


def normalize_header(col_name: str) -> str:
    """
    Robust header normalization:
    - Strips UTF-8 BOM, whitespace, quotes
    - Converts to lowercase
    - Strips parenthetical / bracketed units, e.g. '(°C)', '[kPa]', '(kg/s)', '[MW]'
    - Replaces whitespace and non-alphanumeric punctuation with underscores
    - Strips trailing unit suffixes: _c, _degc, _kpa, _kg_s, _kgs, _mw, _w_m2k, _wm2k, _h, _hours, _meas
    - Deduplicates multiple consecutive underscores
    """
    import re
    if col_name is None:
        return ""
    s = str(col_name).lstrip("\ufeff").strip().lower()
    # Remove bracketed units
    s = re.sub(r"[\(\[\{].*?[\)\]\}]", "", s).strip()
    # Replace non-alphanumeric (except underscore) with underscore
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    # Deduplicate underscores
    s = re.sub(r"_+", "_", s).strip("_")
    # Also strip common trailing unit suffixes if they exist
    s_no_unit = re.sub(
        r"_(c|degc|deg_c|kpa|pa|bar|kg_s|kgs|kg_per_s|mw|kw|w|w_m2k|wm2k|w_per_m2k|m2k_w|h|hours|hr|hrs|meas|deg)$",
        "",
        s,
    ).strip("_")
    return s_no_unit if s_no_unit else s


# =============================================================================
# Manual Operating Snapshot Validation & Indicator Derivation
# =============================================================================
def validate_manual_snapshot(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single manual operating measurement snapshot and derive physical fouling indicators.
    
    Returns structured result with:
      - is_valid / valid: bool
      - errors: List of critical validation errors
      - warnings: List of cautionary engineering warnings
      - validated_condition: Validated operating measurements with units
      - derived_indicators / indicators: Calculated physical indicators (Rf, U, BSI, P_pump)
      - provenance: Explicit source tagging per variable
    """
    errors: List[str] = []
    warnings: List[str] = []
    validated: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}

    # Normalize input dictionary using COLUMN_ALIASES
    raw_dict = dict(input_dict) if input_dict else {}
    normalized_input: Dict[str, Any] = {}
    for k, v in raw_dict.items():
        norm_k = str(k).strip().lower().replace(" ", "_").replace("-", "_")
        matched = False
        for canon, aliases in COLUMN_ALIASES.items():
            if norm_k == canon.lower() or norm_k in aliases:
                normalized_input[canon] = v
                matched = True
                break
        if not matched:
            normalized_input[k] = v

    # 1. Parse & validate required numeric measurements
    req_fields = ["T_cool_in_meas", "T_cool_out_meas", "mass_flow_cool_meas", "delta_P_meas_kPa"]
    for field in req_fields:
        if field not in normalized_input or normalized_input[field] is None or str(normalized_input[field]).strip() == "":
            errors.append(f"Missing required measurement field: '{field}' ({PHYSICAL_RANGES[field]['description']}).")
            continue
        try:
            val = float(normalized_input[field])
            limits = PHYSICAL_RANGES[field]
            if val < limits["min"] or val > limits["max"]:
                errors.append(
                    f"Out of physical range for '{field}': {val} {limits['unit']}. "
                    f"Expected between {limits['min']} and {limits['max']} {limits['unit']}."
                )
            else:
                validated[field] = val
                provenance[field] = "direct_user_measurement"
        except (ValueError, TypeError):
            errors.append(f"Invalid non-numeric value for '{field}': '{normalized_input[field]}'.")

    # Mirror mass flow key for downstream convenience
    if "mass_flow_cool_meas" in validated:
        validated["mass_flow_kg_s"] = validated["mass_flow_cool_meas"]
        provenance["mass_flow_kg_s"] = "direct_user_measurement"

    # 2. Thermodynamic consistency constraint: T_out must be strictly greater than T_in
    if "T_cool_in_meas" in validated and "T_cool_out_meas" in validated:
        delta_T = validated["T_cool_out_meas"] - validated["T_cool_in_meas"]
        if delta_T <= 0.0:
            errors.append(
                f"Thermodynamic violation: Discharge temperature ({validated['T_cool_out_meas']}°C) "
                f"must strictly exceed inlet temperature ({validated['T_cool_in_meas']}°C). ΔT = {delta_T:.2f}°C."
            )
        elif delta_T < 1.0:
            warnings.append(
                f"Low temperature differential ΔT = {delta_T:.2f}°C. Heat transfer may be minimal or sensor resolution limited."
            )
        elif delta_T > 25.0:
            warnings.append(
                f"High temperature differential ΔT = {delta_T:.2f}°C exceeds typical nuclear secondary cooling design."
            )

    # 3. Operating hours & timestamp validation
    hours_val = normalized_input.get("time_hours")
    if hours_val is not None and str(hours_val).strip() != "":
        try:
            h = float(hours_val)
            if h < 0.0:
                errors.append(f"Operating hours cannot be negative: {h} h.")
            else:
                validated["time_hours"] = int(h)
                provenance["time_hours"] = "direct_user_measurement"
        except (ValueError, TypeError):
            errors.append(f"Invalid non-numeric operating hours: '{hours_val}'.")
    else:
        validated["time_hours"] = 1000
        provenance["time_hours"] = "derived_calculation"
        warnings.append("Operating hours not specified. Defaulted to nominal 1,000 h.")

    ts_val = normalized_input.get("timestamp")
    if ts_val and str(ts_val).strip() != "":
        try:
            # Test datetime parseability
            dt_str = str(ts_val).strip().replace("Z", "")
            # Try ISO formats
            parsed_dt = datetime.fromisoformat(dt_str)
            validated["timestamp"] = parsed_dt.isoformat()
            provenance["timestamp"] = "direct_user_measurement"
        except Exception:
            # Fallback string representation
            validated["timestamp"] = str(ts_val).strip()
            provenance["timestamp"] = "direct_user_measurement"
            warnings.append(f"Timestamp '{ts_val}' could not be strictly parsed as ISO-8601; stored as raw string.")
    else:
        validated["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        provenance["timestamp"] = "derived_calculation"
        warnings.append("Timestamp omitted; recorded current UTC time.")

    # 4. Mandatory snapshot caveat warning
    warnings.append(
        "Single snapshot operating input: Instantaneous fouling resistance and hydraulic penalties "
        "are physically evaluated, but transient biofouling growth rate (dRf/dt) cannot be inferred from a single point. "
        "Forward projections utilize calibrated biophysical kinetics initialized at this operating state."
    )

    if len(errors) > 0:
        return {
            "is_valid": False,
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "validated_condition": {},
            "derived_indicators": {},
            "indicators": {},
            "provenance": provenance,
            "data_source_mode": "manual_operating_input",
            "nuclear_safety_notice": "Academic Demonstration Prototype — User Provided Operating Data",
        }

    # 5. Physical indicator derivations
    m_dot = validated["mass_flow_cool_meas"]
    T_in = validated["T_cool_in_meas"]
    T_out = validated["T_cool_out_meas"]
    dp_meas = validated["delta_P_meas_kPa"]
    clean_dp = config.CLEAN_PRESSURE_DROP_KPA
    area_m2 = config.TOTAL_HEAT_TRANSFER_AREA_M2
    cp = config.WATER_SPECIFIC_HEAT_J_KG_K

    # Heat duty (MW)
    if "heat_duty_MW" in input_dict and input_dict["heat_duty_MW"] is not None and str(input_dict["heat_duty_MW"]).strip() != "":
        try:
            q_mw = float(input_dict["heat_duty_MW"])
            validated["heat_duty_MW"] = round(q_mw, 3)
            provenance["heat_duty_MW"] = "direct_user_measurement"
        except ValueError:
            q_mw = (m_dot * cp * max(0.0, T_out - T_in)) / 1e6
            validated["heat_duty_MW"] = round(q_mw, 3)
            provenance["heat_duty_MW"] = "derived_calculation"
    else:
        q_mw = (m_dot * cp * max(0.0, T_out - T_in)) / 1e6
        validated["heat_duty_MW"] = round(q_mw, 3)
        provenance["heat_duty_MW"] = "derived_calculation"

    u_clean = config.U_CLEAN_DESIGN_W_M2K

    # Overall HTC U (W/(m²·K)) and Thermal Fouling Resistance Rf [m²·K/W]
    has_user_u = False
    if "U_overall_meas" in input_dict and input_dict["U_overall_meas"] is not None and str(input_dict["U_overall_meas"]).strip() != "":
        try:
            u_meas = float(input_dict["U_overall_meas"])
            if u_meas > 0:
                validated["U_overall_meas"] = round(u_meas, 1)
                provenance["U_overall_meas"] = "direct_user_measurement"
                has_user_u = True
        except ValueError:
            pass

    if has_user_u:
        if u_meas < u_clean:
            rf_calc = (1.0 / u_meas) - (1.0 / u_clean)
        else:
            rf_calc = 0.0
        rf_calc = max(0.0, min(0.001, rf_calc))
        delta_f_m = rf_calc * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK
        biofilm_um = delta_f_m * 1e6
        provenance["Rf_fouling_resistance"] = "derived_calculation"
        provenance["biofilm_thickness_um"] = "derived_calculation"
    else:
        # User omitted U_overall_meas: invert Darcy-Weisbach continuum mechanics from measured delta-P
        delta_f_m = fouling_indicators.invert_biofilm_thickness_from_dp(dp_meas, mass_flow_kg_s=m_dot)
        biofilm_um = delta_f_m * 1e6
        rf_calc = delta_f_m / config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK
        rf_calc = max(0.0, min(0.001, rf_calc))
        if rf_calc > 0:
            u_meas = u_clean / (1.0 + u_clean * rf_calc)
        else:
            u_meas = u_clean
        u_meas = max(200.0, min(u_clean * 1.05, u_meas))
        validated["U_overall_meas"] = round(u_meas, 1)
        provenance["U_overall_meas"] = "derived_calculation"
        provenance["Rf_fouling_resistance"] = "derived_calculation"
        provenance["biofilm_thickness_um"] = "derived_calculation"

    # Thermal efficiency ratio
    eff_ratio = fouling_indicators.compute_thermal_efficiency_ratio(u_meas, u_clean)

    # Biofouling Severity Index (BSI)
    bsi = fouling_indicators.compute_biofouling_severity_index(rf_calc, dp_meas / clean_dp)

    # Parasitic pumping power kW
    vol_flow = m_dot / config.WATER_DENSITY_KG_M3
    excess_dp = max(0.0, dp_meas - clean_dp)
    p_pump_kw = sustainability_metrics.compute_parasitic_pumping_power_kw(vol_flow, excess_dp)

    derived = {
        "Rf_fouling_resistance": round(rf_calc, 7),
        "thermal_efficiency_ratio": round(eff_ratio, 4),
        "U_overall_meas": round(u_meas, 1),
        "U_overall_W_m2K": round(u_meas, 1),
        "biofilm_thickness_um": round(biofilm_um, 2),
        "bsi_score": round(bsi, 1),
        "clean_pressure_drop_kPa": round(clean_dp, 2),
        "delta_P_ratio": round(dp_meas / clean_dp, 2),
        "excess_pressure_drop_kPa": round(excess_dp, 2),
        "parasitic_pump_kw": round(p_pump_kw, 2),
        "heat_duty_MW": validated.get("heat_duty_MW", round(q_mw, 3)),
        "provenance_summary": {
            "Rf_fouling_resistance": provenance.get("Rf_fouling_resistance", "derived_calculation"),
            "thermal_efficiency_ratio": "derived_calculation",
            "biofilm_thickness_um": provenance.get("biofilm_thickness_um", "derived_calculation"),
            "bsi_score": "derived_calculation",
            "parasitic_pump_kw": "derived_calculation",
        },
    }

    # Combine provenance dictionaries so full variable lineage is accessible
    full_provenance = dict(provenance)
    full_provenance.update(derived.get("provenance_summary", {}))

    return {
        "is_valid": True,
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "validated_condition": validated,
        "derived_indicators": derived,
        "indicators": derived,
        "provenance": full_provenance,
        "data_source_mode": "manual_operating_input",
        "nuclear_safety_notice": "Academic Demonstration Prototype — User Provided Operating Data",
    }


# =============================================================================
# Historical CSV Ingestion, Preview, Mapping & Quality Scoring
# =============================================================================
class CSVDataIngestor:
    """
    Robust CSV ingestion engine supporting format auto-detection, schema mapping,
    temporal verification, outlier detection, and comprehensive data quality reporting.
    """

    @staticmethod
    def detect_delimiter(sample_text: str) -> str:
        """
        Robustly detect whether CSV uses comma, semicolon, or tab.
        Prioritizes the header row delimiter structure and ignores colons from timestamps.
        """
        if not sample_text:
            return ","
        clean_text = sample_text.lstrip("\ufeff")
        first_line = clean_text.splitlines()[0] if clean_text.splitlines() else ""
        if first_line:
            c_comma = first_line.count(",")
            c_semi = first_line.count(";")
            c_tab = first_line.count("\t")
            if c_semi > c_comma and c_semi >= c_tab:
                return ";"
            if c_tab > c_comma and c_tab > c_semi:
                return "\t"
            if c_comma > 0:
                return ","
        # Fallback to sniffer restricted to standard CSV delimiters
        try:
            dialect = csv.Sniffer().sniff(clean_text[:4096], delimiters=[",", ";", "\t"])
            return dialect.delimiter
        except Exception:
            return ","

    @classmethod
    def generate_preview(cls, csv_text: str, max_rows: int = 5) -> Dict[str, Any]:
        """
        Parse first N rows of uploaded CSV and infer canonical column mappings.
        """
        if not csv_text or not csv_text.strip():
            return {
                "success": False,
                "error": "Uploaded CSV file content is empty.",
            }

        clean_text = csv_text.lstrip("\ufeff")
        delim = cls.detect_delimiter(clean_text)
        try:
            df_preview = pd.read_csv(io.StringIO(clean_text), sep=delim, nrows=max_rows)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse CSV format with delimiter '{delim}': {e}",
            }

        headers = list(df_preview.columns)
        inferred_mapping = {}

        # Heuristic matching of headers to canonical columns:
        # Step 1: Match exact canonical name or exact alias
        # Step 2: Fall back to unit-stripped normalized matching
        for col in headers:
            norm_col = str(col).lstrip("\ufeff").strip().lower().replace(" ", "_").replace("-", "_")
            norm_col_clean = normalize_header(col)
            matched = False
            for canonical, aliases in COLUMN_ALIASES.items():
                if norm_col == canonical.lower() or norm_col in aliases:
                    inferred_mapping[col] = canonical
                    matched = True
                    break
            if not matched:
                for canonical, aliases in COLUMN_ALIASES.items():
                    if norm_col_clean == canonical.lower() or norm_col_clean in aliases:
                        inferred_mapping[col] = canonical
                        break

        # Count total rows safely
        lines = [line for line in clean_text.splitlines() if line.strip()]
        total_data_rows = max(0, len(lines) - 1)

        sample_records = df_preview.fillna("").to_dict(orient="records")

        req_cols = [
            "T_cool_in_meas",
            "T_cool_out_meas",
            "mass_flow_cool_meas",
            "delta_P_meas_kPa",
        ]
        mapped_canonicals = set(inferred_mapping.values())
        missing_mandatory = [c for c in req_cols if c not in mapped_canonicals]

        return {
            "success": True,
            "delimiter": delim,
            "total_estimated_rows": total_data_rows,
            "row_count": total_data_rows,
            "headers": headers,
            "sample_rows": sample_records,
            "preview_rows": sample_records,
            "inferred_mapping": inferred_mapping,
            "suggested_mapping": inferred_mapping,
            "required_canonical_columns": req_cols,
            "missing_mandatory_columns": missing_mandatory,
            "recommended_canonical_columns": [
                "timestamp",
                "time_hours",
            ],
        }

    @classmethod
    def ingest_and_validate(
        cls,
        csv_text: str,
        column_mapping: Optional[Dict[str, str]] = None,
        dataset_name: str = "custom_operating_telemetry",
    ) -> Dict[str, Any]:
        """
        Full parsing, schema normalization, temporal integrity check,
        outlier detection, and data quality scoring for uploaded CSV datasets.
        """
        if not csv_text or not csv_text.strip():
            return {
                "success": False,
                "error": "CSV content is empty.",
                "quality_report": None,
            }

        clean_text = csv_text.lstrip("\ufeff")
        delim = cls.detect_delimiter(clean_text)
        try:
            raw_df = pd.read_csv(io.StringIO(clean_text), sep=delim)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read CSV text with delimiter '{delim}': {e}",
                "quality_report": None,
            }

        # Clean raw headers of BOM if present
        raw_df.columns = [str(c).lstrip("\ufeff").strip() for c in raw_df.columns]

        # Apply column mapping
        if column_mapping:
            rename_dict = {}
            for orig, target in column_mapping.items():
                clean_orig = str(orig).lstrip("\ufeff").strip()
                if clean_orig in raw_df.columns and target:
                    rename_dict[clean_orig] = target
            df = raw_df.rename(columns=rename_dict)
        else:
            # Auto-infer mapping
            preview = cls.generate_preview(clean_text)
            df = raw_df.rename(columns=preview.get("inferred_mapping", {}))

        total_rows = len(df)
        if total_rows == 0:
            return {
                "success": False,
                "error": "CSV contains no data rows.",
                "quality_report": None,
            }

        anomalies: List[Dict[str, Any]] = []
        warnings: List[str] = []

        # 1. Check required canonical columns
        req_cols = ["T_cool_in_meas", "T_cool_out_meas", "mass_flow_cool_meas", "delta_P_meas_kPa"]
        missing_req = [c for c in req_cols if c not in df.columns]
        if missing_req:
            found_cols = list(raw_df.columns)
            return {
                "success": False,
                "error": (
                    f"Missing mandatory operating columns after mapping: {missing_req}. "
                    f"Detected columns in file: {found_cols}. "
                    f"Please map your columns to required canonical fields: {req_cols}."
                ),
                "quality_report": {
                    "total_rows": total_rows,
                    "valid_rows": 0,
                    "quality_rating": "REJECTED",
                    "quality_score": 0.0,
                    "missing_mandatory_columns": missing_req,
                    "detected_columns": found_cols,
                },
            }

        # 2. Check and normalize temporal ordering
        has_timestamp = "timestamp" in df.columns
        has_hours = "time_hours" in df.columns

        duplicate_timestamps = 0
        out_of_order_count = 0

        if has_timestamp:
            try:
                # Convert timestamps to datetime
                ts_series = pd.to_datetime(df["timestamp"], errors="coerce")
                null_ts = ts_series.isna().sum()
                if null_ts > 0:
                    warnings.append(f"{null_ts} timestamp entries could not be parsed as valid dates.")

                # Check duplicates
                dup_mask = ts_series.duplicated(keep=False)
                duplicate_timestamps = int(dup_mask.sum())
                if duplicate_timestamps > 0:
                    anomalies.append({
                        "type": "duplicate_timestamps",
                        "count": duplicate_timestamps,
                        "description": f"Found {duplicate_timestamps} duplicate timestamps.",
                    })

                # Check ordering
                valid_ts = ts_series.dropna()
                if not valid_ts.is_monotonic_increasing:
                    out_of_order_count = int((valid_ts.diff().dt.total_seconds() < 0).sum())
                    warnings.append(f"Timestamps are not strictly ascending ({out_of_order_count} time inversions detected).")

                # If time_hours missing, derive from timestamp delta
                if not has_hours and len(valid_ts) > 0:
                    t0 = valid_ts.iloc[0]
                    df["time_hours"] = ((valid_ts - t0).dt.total_seconds() / 3600.0).round().astype(int)
                    has_hours = True
            except Exception as ex:
                warnings.append(f"Timestamp parsing error: {ex}")

        if has_hours:
            try:
                hours_series = pd.to_numeric(df["time_hours"], errors="coerce")
                if not has_timestamp:
                    dup_mask = hours_series.duplicated()
                    duplicate_timestamps = int(dup_mask.sum())
                    if duplicate_timestamps > 0:
                        anomalies.append({
                            "type": "duplicate_time_hours",
                            "count": duplicate_timestamps,
                            "description": f"Found {duplicate_timestamps} duplicate time_hours.",
                        })
                if not hours_series.dropna().is_monotonic_increasing:
                    inv = int((hours_series.dropna().diff() < 0).sum())
                    out_of_order_count = max(out_of_order_count, inv)
                    warnings.append(f"time_hours column contains {inv} backward step inversions.")
            except Exception:
                pass
        else:
            # Synthetic hour indexing if both missing
            df["time_hours"] = np.arange(len(df))
            warnings.append("Neither timestamp nor time_hours was provided. Synthesized incremental hourly index.")

        if not has_timestamp:
            df["timestamp"] = [f"Day {h//24:02d} Hour {h%24:02d}:00" for h in df["time_hours"]]

        # 3. Missing Value Analysis (Never silently fabricate)
        missing_cells_per_col = df[req_cols].isna().sum().to_dict()
        total_missing_cells = int(df[req_cols].isna().sum().sum())
        rows_with_missing = int(df[req_cols].isna().any(axis=1).sum())

        if total_missing_cells > 0:
            anomalies.append({
                "type": "missing_observations",
                "missing_cells_by_column": missing_cells_per_col,
                "rows_affected": rows_with_missing,
                "description": f"{rows_with_missing} rows contain missing values in core telemetry fields.",
            })

        # 4. Outlier & Physical Plausibility Detection
        range_violations = 0
        thermo_violations = 0
        valid_row_mask = np.ones(total_rows, dtype=bool)

        for col in req_cols:
            limits = PHYSICAL_RANGES[col]
            s = pd.to_numeric(df[col], errors="coerce")
            viol = (s < limits["min"]) | (s > limits["max"]) | s.isna()
            cnt = int(viol.sum())
            if cnt > 0:
                range_violations += cnt
                valid_row_mask &= (~viol)
                anomalies.append({
                    "type": f"range_outlier_{col}",
                    "column": col,
                    "count": cnt,
                    "limits": f"[{limits['min']}, {limits['max']}] {limits['unit']}",
                    "description": f"{cnt} observations outside physical limits for {col}.",
                })

        # Check Tout > Tin
        tin_s = pd.to_numeric(df["T_cool_in_meas"], errors="coerce")
        tout_s = pd.to_numeric(df["T_cool_out_meas"], errors="coerce")
        thermo_viol_mask = (tout_s <= tin_s) & tin_s.notna() & tout_s.notna()
        thermo_violations = int(thermo_viol_mask.sum())
        if thermo_violations > 0:
            valid_row_mask &= (~thermo_viol_mask)
            anomalies.append({
                "type": "thermodynamic_inversion",
                "count": thermo_violations,
                "description": f"{thermo_violations} rows where discharge temp Tout <= intake temp Tin.",
            })

        valid_rows_count = int(valid_row_mask.sum())

        # 5. Calculate Comprehensive Data Quality Score (0 to 100%)
        # Deduct penalties for missing cells, range violations, thermo inversions, duplicate timestamps
        completeness_ratio = max(0.0, 1.0 - (total_missing_cells / float(total_rows * len(req_cols))))
        validity_ratio = float(valid_rows_count) / float(total_rows) if total_rows > 0 else 0.0
        temporal_ratio = max(0.0, 1.0 - ((duplicate_timestamps + out_of_order_count) / float(total_rows)))

        quality_score = (completeness_ratio * 40.0) + (validity_ratio * 45.0) + (temporal_ratio * 15.0)
        # If all rows are physically invalid, quality score must be zero
        if valid_rows_count == 0 or validity_ratio == 0.0:
            quality_score = 0.0
            rating = "REJECTED"
        elif quality_score >= 90.0:
            rating = "EXCELLENT"
        elif quality_score >= 75.0:
            rating = "GOOD"
        elif quality_score >= 50.0:
            rating = "DEGRADED"
        else:
            rating = "REJECTED"

        quality_score = round(max(0.0, min(100.0, quality_score)), 1)

        if rating == "REJECTED" or valid_rows_count < 2 or quality_score < 50.0:
            quality_report = {
                "dataset_id": "rejected_upload",
                "filename": dataset_name,
                "total_rows": total_rows,
                "valid_rows": valid_rows_count,
                "invalid_rows": total_rows - valid_rows_count,
                "completeness_pct": round(completeness_ratio * 100.0, 1),
                "completeness_score": round(completeness_ratio * 100.0, 1),
                "validity_pct": round(validity_ratio * 100.0, 1),
                "missing_data_count": total_missing_cells,
                "missing_summary": missing_cells_per_col,
                "duplicate_timestamps": duplicate_timestamps,
                "duplicate_timestamp_count": duplicate_timestamps,
                "out_of_order_count": out_of_order_count,
                "temporal_anomalies_detected": (duplicate_timestamps > 0) or (out_of_order_count > 0),
                "range_violations": range_violations,
                "thermodynamic_inversions": thermo_violations,
                "detected_cleaning_events": 0,
                "quality_score": quality_score,
                "quality_rating": rating,
                "anomalies": anomalies,
                "warnings": warnings,
                "data_provenance": "user_uploaded_csv",
                "nuclear_safety_notice": "Academic Demonstration Prototype — User Ingested Telemetry (REJECTED)",
            }
            return {
                "success": False,
                "error": f"Uploaded dataset quality score ({quality_score}%) is REJECTED. Minimum 50.0% required.",
                "dataframe": None,
                "quality_report": quality_report,
                "quality_assessment": quality_report,
                "ml_sufficient_history": False,
                "warnings": warnings,
            }

        # 6. Physical Indicator Computation on Valid Rows
        clean_dp = config.CLEAN_PRESSURE_DROP_KPA
        u_clean = config.U_CLEAN_DESIGN_W_M2K
        area_m2 = config.TOTAL_HEAT_TRANSFER_AREA_M2
        cp = config.WATER_SPECIFIC_HEAT_J_KG_K

        processed_df = df.copy()

        # Numeric conversions
        for col in req_cols:
            processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

        delta_T = processed_df["T_cool_out_meas"] - processed_df["T_cool_in_meas"]
        processed_df["heat_duty_MW"] = (processed_df["mass_flow_cool_meas"] * cp * delta_T) / 1e6
        processed_df["heat_duty_MW"] = processed_df["heat_duty_MW"].round(3)

        delta_T_lmtd = 12.0
        u_calc = (processed_df["heat_duty_MW"] * 1e6) / (area_m2 * delta_T_lmtd)
        processed_df["U_overall_meas"] = u_calc.clip(lower=200.0, upper=u_clean * 1.05).round(1)

        # Thermal resistance Rf
        rf_vals = np.where(
            processed_df["U_overall_meas"] < u_clean,
            (1.0 / processed_df["U_overall_meas"]) - (1.0 / u_clean),
            0.0
        )
        processed_df["Rf_fouling_resistance"] = np.clip(rf_vals, a_min=0.0, a_max=0.001).round(7)
        processed_df["thermal_efficiency_ratio"] = (processed_df["U_overall_meas"] / u_clean).clip(lower=0.0, upper=1.0).round(4)

        # Biofilm thickness
        delta_f = processed_df["Rf_fouling_resistance"] * config.BIOFILM_THERMAL_CONDUCTIVITY_W_MK
        processed_df["biofilm_thickness_um"] = (delta_f * 1e6).round(2)

        # BSI score
        dp_ratio = processed_df["delta_P_meas_kPa"] / clean_dp
        rf_ratio = processed_df["Rf_fouling_resistance"] / 0.0005
        bsi_vals = (0.60 * np.clip(dp_ratio - 1.0, a_min=0.0, a_max=3.0) / 3.0 + 0.40 * np.clip(rf_ratio, a_min=0.0, a_max=1.0)) * 100.0
        processed_df["bsi_score"] = np.clip(bsi_vals, a_min=0.0, a_max=100.0).round(1)

        # Parasitic pump power
        vol_flows = processed_df["mass_flow_cool_meas"] / config.WATER_DENSITY_KG_M3
        excess_dps = np.maximum(0.0, processed_df["delta_P_meas_kPa"] - clean_dp)
        p_pumps = [
            sustainability_metrics.compute_parasitic_pumping_power_kw(vf, edp)
            for vf, edp in zip(vol_flows, excess_dps)
        ]
        processed_df["parasitic_pump_kw"] = np.round(p_pumps, 2)
        processed_df["data_source"] = "user_uploaded_csv"

        # 7. Identify potential historical cleaning events
        rf_diffs = processed_df["Rf_fouling_resistance"].diff()
        dp_diffs = processed_df["delta_P_meas_kPa"].diff()
        cleaning_event_mask = (rf_diffs < -2.0e-5) | (dp_diffs < -1.5)
        processed_df["cleaning_event"] = cleaning_event_mask.astype(int)
        detected_cleaning_count = int(cleaning_event_mask.sum())

        # 8. Save sanitized dataset locally
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in dataset_name if c.isalnum() or c in ("_", "-")).strip() or "telemetry"
        file_id = f"{safe_name}_{timestamp_str}"
        saved_csv_path = USER_UPLOADS_DIR / f"{file_id}.csv"
        saved_meta_path = USER_UPLOADS_DIR / f"{file_id}.json"

        processed_df.to_csv(saved_csv_path, index=False)

        ml_sufficient = len(processed_df) >= 25
        if not ml_sufficient:
            warnings.append(
                "Dataset has < 25 rows (insufficient 24h lag/rolling window for ML ensemble). "
                "Physics ODE model will be used for forecasting."
            )

        quality_report = {
            "dataset_id": file_id,
            "filename": saved_csv_path.name,
            "total_rows": total_rows,
            "valid_rows": valid_rows_count,
            "invalid_rows": total_rows - valid_rows_count,
            "completeness_pct": round(completeness_ratio * 100.0, 1),
            "completeness_score": round(completeness_ratio * 100.0, 1),
            "validity_pct": round(validity_ratio * 100.0, 1),
            "missing_data_count": total_missing_cells,
            "missing_summary": missing_cells_per_col,
            "duplicate_timestamps": duplicate_timestamps,
            "duplicate_timestamp_count": duplicate_timestamps,
            "out_of_order_count": out_of_order_count,
            "temporal_anomalies_detected": (duplicate_timestamps > 0) or (out_of_order_count > 0),
            "range_violations": range_violations,
            "thermodynamic_inversions": thermo_violations,
            "detected_cleaning_events": detected_cleaning_count,
            "quality_score": quality_score,
            "quality_rating": rating,
            "anomalies": anomalies,
            "warnings": warnings,
            "data_provenance": "user_uploaded_csv",
            "nuclear_safety_notice": "Academic Demonstration Prototype — User Ingested Telemetry",
        }

        with open(saved_meta_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, indent=2)

        return {
            "success": True,
            "error": None,
            "dataframe": processed_df,
            "quality_report": quality_report,
            "quality_assessment": quality_report,
            "ml_sufficient_history": ml_sufficient,
            "warnings": warnings,
            "saved_csv_path": str(saved_csv_path),
        }
