"""
KPI calculation utilities.

These functions take *cleaned* DataFrames (see modules.data_cleaner) and
produce KPI-enriched DataFrames + a campaign-level KPI summary.

All functions are read-only with respect to inputs (we always copy).
Nothing in this module talks to a DSP or to DeepSeek.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------
def _safe_div(num, denom):
    num = pd.to_numeric(num, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    return np.where((denom == 0) | denom.isna(), np.nan, num / denom)


# ---------------------------------------------------------------------------
# Safe scalar helpers — never raise on missing columns / NaN / dirty strings
# ---------------------------------------------------------------------------
def _safe_numeric_series(df, column_name) -> pd.Series:
    """Return a numeric Series from df[column_name], tolerant of:
    - missing column
    - duplicate column names (DataFrame instead of Series)
    - currency, comma, percent, whitespace
    - blank / "nan" / "None" / "N/A" / "-" sentinels
    - infinities
    """
    if df is None or len(df) == 0 or column_name not in df.columns:
        return pd.Series(dtype="float64")

    series = df[column_name]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace(["", "nan", "NaN", "None", "N/A", "NA", "-", "--"], np.nan)
    )

    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
    return numeric_series


def safe_sum(df, column_name) -> float:
    """Sum a numeric column safely; missing/invalid become 0."""
    series = _safe_numeric_series(df, column_name)
    if series.empty:
        return 0.0
    return float(series.fillna(0).sum())


def safe_int_sum(df, column_name) -> int:
    """Sum a numeric column and return an integer."""
    return int(round(safe_sum(df, column_name)))


def safe_avg(df, column_name) -> float:
    """Mean of a numeric column; ignores NaN; returns 0.0 if empty."""
    series = _safe_numeric_series(df, column_name).dropna()
    if series.empty:
        return 0.0
    return float(series.mean())


def safe_divide(numerator, denominator) -> float:
    """Divide two scalars safely; 0 denominator or invalid → 0.0."""
    try:
        n = float(numerator or 0)
        d = float(denominator or 0)
        if d == 0:
            return 0.0
        return n / d
    except Exception:
        return 0.0


def _to_dt(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Campaign-level KPIs
# ---------------------------------------------------------------------------
def calculate_campaign_kpis(
    df: pd.DataFrame,
    flight_start: Optional[date] = None,
    flight_end: Optional[date] = None,
    pacing_goal: float = 1.0,
) -> pd.DataFrame:
    """Add pacing, CPM/CTR/CVR/CPA, and gap columns to a campaign report."""
    if df is None or df.empty:
        return df

    out = df.copy()

    # Recompute base KPIs only when source columns exist
    if {"spend", "impressions"}.issubset(out.columns):
        out["cpm_calc"] = _safe_div(out["spend"], out["impressions"]) * 1000.0
    if {"clicks", "impressions"}.issubset(out.columns):
        out["ctr_calc"] = _safe_div(out["clicks"], out["impressions"])
    if {"conversions", "clicks"}.issubset(out.columns):
        out["cvr_calc"] = _safe_div(out["conversions"], out["clicks"])
    if {"spend", "conversions"}.issubset(out.columns):
        out["cpa_calc"] = _safe_div(out["spend"], out["conversions"])

    # Pacing math (campaign level)
    fs = _to_dt(flight_start)
    fe = _to_dt(flight_end)
    today = datetime.combine(date.today(), datetime.min.time())

    if "budget" in out.columns and "spend" in out.columns:
        out["budget_remaining"] = pd.to_numeric(out["budget"], errors="coerce") \
            - pd.to_numeric(out["spend"], errors="coerce")

        if fs and fe and fe > fs:
            total_days = max(1, (fe - fs).days + 1)
            elapsed = max(0, min(total_days, (today - fs).days + 1))
            expected_spend = pd.to_numeric(out["budget"], errors="coerce") * (
                elapsed / total_days
            )
            out["expected_spend_to_date"] = expected_spend
            out["pacing_calc"] = _safe_div(out["spend"], expected_spend)

            days_remaining = max(1, (fe - today).days)
            out["days_remaining"] = days_remaining
            out["daily_required_spend"] = _safe_div(
                out["budget_remaining"], pd.Series([days_remaining] * len(out))
            )
        else:
            out["pacing_calc"] = pd.to_numeric(out.get("pacing"), errors="coerce")

    # If user-provided pacing exists and our calc is missing, fall back
    if "pacing_calc" not in out.columns and "pacing" in out.columns:
        out["pacing_calc"] = pd.to_numeric(out["pacing"], errors="coerce")

    # Gap columns (computed against thresholds at rule-time, but useful to expose)
    return out


# ---------------------------------------------------------------------------
# Site / app KPIs
# ---------------------------------------------------------------------------
def calculate_site_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute CPA, CTR, viewability gap on a site/app report."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if {"spend", "conversions"}.issubset(out.columns):
        out["cpa_calc"] = _safe_div(out["spend"], out["conversions"])
    if {"clicks", "impressions"}.issubset(out.columns):
        out["ctr_calc"] = _safe_div(out["clicks"], out["impressions"])
    if {"spend", "impressions"}.issubset(out.columns):
        out["cpm_calc"] = _safe_div(out["spend"], out["impressions"]) * 1000.0
    return out


# ---------------------------------------------------------------------------
# Zip KPIs
# ---------------------------------------------------------------------------
def calculate_zip_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute CPA per ZIP."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if {"spend", "conversions"}.issubset(out.columns):
        out["cpa_calc"] = _safe_div(out["spend"], out["conversions"])
    return out


# ---------------------------------------------------------------------------
# PMP KPIs
# ---------------------------------------------------------------------------
def calculate_pmp_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CPM, win-rate gap, and floor-vs-bid mismatch."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if {"spend", "impressions"}.issubset(out.columns):
        out["cpm_calc"] = _safe_div(out["spend"], out["impressions"]) * 1000.0
    if {"floor_cpm", "bid"}.issubset(out.columns):
        out["floor_minus_bid"] = pd.to_numeric(out["floor_cpm"], errors="coerce") \
            - pd.to_numeric(out["bid"], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Aggregate "summary" used by DeepSeek prompt + dashboard
# ---------------------------------------------------------------------------
def build_kpi_summary(
    campaign_df: Optional[pd.DataFrame],
    benchmarks: Dict,
) -> Dict:
    """Return a small dict that summarizes overall campaign performance."""
    summary = {
        "total_spend": 0.0,
        "total_impressions": 0,
        "total_clicks": 0,
        "total_conversions": 0,
        "blended_cpa": None,
        "blended_cpm": None,
        "blended_ctr": None,
        "blended_cvr": None,
        "avg_pacing": None,
        "avg_viewability": None,
        "avg_out_of_geo": None,
        "avg_ivt": None,
        "benchmarks": benchmarks,
    }
    if campaign_df is None or campaign_df.empty:
        return summary

    df = campaign_df

    total_spend = safe_sum(df, "spend")
    total_impressions = safe_sum(df, "impressions")
    total_clicks = safe_sum(df, "clicks")
    total_conversions = safe_sum(df, "conversions")
    total_budget = safe_sum(df, "budget")

    summary["total_spend"] = round(total_spend, 2)
    summary["total_impressions"] = safe_int_sum(df, "impressions")
    summary["total_clicks"] = safe_int_sum(df, "clicks")
    summary["total_conversions"] = safe_int_sum(df, "conversions")
    summary["total_budget"] = round(total_budget, 2)

    cpa = safe_divide(total_spend, total_conversions)
    cpm = safe_divide(total_spend, total_impressions) * 1000.0
    ctr = safe_divide(total_clicks, total_impressions)
    cvr = safe_divide(total_conversions, total_clicks)

    summary["blended_cpa"] = round(cpa, 2) if total_conversions > 0 else None
    summary["blended_cpm"] = round(cpm, 2) if total_impressions > 0 else None
    summary["blended_ctr"] = round(ctr, 4) if total_impressions > 0 else None
    summary["blended_cvr"] = round(cvr, 4) if total_clicks > 0 else None

    pacing_col = "pacing_calc" if "pacing_calc" in df.columns else "pacing"
    pacing_avg = safe_avg(df, pacing_col)
    summary["avg_pacing"] = round(pacing_avg, 4) if pacing_col in df.columns else None
    summary["avg_viewability"] = round(safe_avg(df, "viewability"), 4) if "viewability" in df.columns else None
    summary["avg_out_of_geo"] = round(safe_avg(df, "out_of_geo"), 4) if "out_of_geo" in df.columns else None
    summary["avg_ivt"] = round(safe_avg(df, "ivt"), 4) if "ivt" in df.columns else None

    return summary
