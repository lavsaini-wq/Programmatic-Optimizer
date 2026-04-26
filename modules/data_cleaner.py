"""
Data cleaning utilities.

This module standardizes uploaded report data so the rest of the app
(KPI calculations, optimization rules, DeepSeek summaries) can rely on
clean, predictable columns and types.

Key responsibilities:
- Standardize column names (lowercase, snake_case, trimmed).
- Drop blank and duplicate rows.
- Convert currency-like strings ("$1,234.50") to floats.
- Convert percentage strings ("12.4%") to numeric ratios (0.124).
- Force ZIP codes into 5-digit text (preserves leading zeros).
- Provide a per-file Data QA summary the user can review.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column-name standardization
# ---------------------------------------------------------------------------
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with cleaned column names (snake_case)."""
    if df is None or df.empty:
        return df

    cleaned = df.copy()
    new_cols = []
    for col in cleaned.columns:
        name = str(col).strip().lower()
        # Replace any non-alphanumeric with underscore, collapse repeats
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        new_cols.append(name)
    cleaned.columns = new_cols
    return cleaned


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------
def _to_numeric_currency(value) -> Optional[float]:
    """Strip $ , space and convert to float; return None on failure."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    # Handle parentheses for negatives e.g. (123.45)
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _to_numeric_percent(value) -> Optional[float]:
    """Convert "12.4%" or 12.4 to 0.124. Numeric > 1.5 assumed to already be %."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value) / 100.0 if float(value) > 1.5 else float(value)
    s = str(value).strip()
    if not s:
        return None
    has_pct = s.endswith("%")
    s = s.replace("%", "").replace(",", "").strip()
    try:
        num = float(s)
    except ValueError:
        return None
    if has_pct or num > 1.5:
        return num / 100.0
    return num


def _to_zip5(value) -> Optional[str]:
    """Coerce a value into a 5-character ZIP code with leading zeros."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    # Strip ZIP+4 suffix
    s = s.split("-")[0].split(".")[0]
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    if len(digits) > 5:
        digits = digits[-5:]
    return digits.zfill(5)


# ---------------------------------------------------------------------------
# Field classification (used for type coercion below)
# ---------------------------------------------------------------------------
CURRENCY_FIELDS = {
    "spend", "budget", "cpa", "cpm", "cpc", "floor_cpm", "bid",
    "base_bid", "max_bid", "target_cpa", "target_cpm",
}
PERCENT_FIELDS = {
    "ctr", "cvr", "viewability", "out_of_geo", "ivt", "pacing",
    "win_rate", "viewability_gap", "out_of_geo_gap", "ivt_gap",
}
INTEGER_FIELDS = {"impressions", "clicks", "conversions"}
ZIP_FIELDS = {"zip", "zip_code", "postal_code"}


def coerce_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply currency / percent / int / zip coercions for known field names."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in list(out.columns):
        if col in CURRENCY_FIELDS:
            out[col] = out[col].map(_to_numeric_currency)
        elif col in PERCENT_FIELDS:
            out[col] = out[col].map(_to_numeric_percent)
        elif col in INTEGER_FIELDS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif col in ZIP_FIELDS:
            out[col] = out[col].map(_to_zip5)
    return out


# ---------------------------------------------------------------------------
# Apply user column mapping
# ---------------------------------------------------------------------------
def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename source columns to standard field names per the mapping dict.

    mapping: {standard_field: source_column_name}
    Source columns not referenced in the mapping are kept as-is.
    """
    if df is None or df.empty or not mapping:
        return df
    rename = {src: std for std, src in mapping.items() if src and src in df.columns}
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------------
def clean_dataframe(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Run the full cleaning pipeline on a single DataFrame."""
    if df is None or df.empty:
        return df

    out = standardize_column_names(df)
    if mapping:
        out = apply_mapping(out, mapping)
    out = coerce_known_columns(out)

    # Drop fully blank rows
    out = out.dropna(how="all")
    # Drop exact duplicate rows
    out = out.drop_duplicates()
    out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Data QA summary
# ---------------------------------------------------------------------------
def summarize_qa(name: str, df: pd.DataFrame) -> Dict:
    """Build a QA summary record for a given cleaned DataFrame."""
    if df is None or df.empty:
        return {
            "report": name,
            "rows": 0,
            "columns": 0,
            "missing_pct": 0.0,
            "duplicate_rows_remaining": 0,
            "notes": "Empty dataset",
        }
    missing = float(df.isna().sum().sum())
    cells = float(df.size) if df.size else 1.0
    return {
        "report": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_pct": round(100.0 * missing / cells, 2),
        "duplicate_rows_remaining": int(df.duplicated().sum()),
        "notes": "OK",
    }


def build_qa_dataframe(reports: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a single QA summary DataFrame across all uploaded reports."""
    rows: List[Dict] = []
    for name, df in reports.items():
        rows.append(summarize_qa(name, df))
    return pd.DataFrame(rows)
