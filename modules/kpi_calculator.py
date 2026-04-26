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
    totals = {
        "spend": float(pd.to_numeric(df.get("spend"), errors="coerce").sum() or 0.0),
        "impressions": int(pd.to_numeric(df.get("impressions"), errors="coerce").sum() or 0),
        "clicks": int(pd.to_numeric(df.get("clicks"), errors="coerce").sum() or 0),
        "conversions": int(pd.to_numeric(df.get("conversions"), errors="coerce").sum() or 0),
    }
    summary["total_spend"] = round(totals["spend"], 2)
    summary["total_impressions"] = totals["impressions"]
    summary["total_clicks"] = totals["clicks"]
    summary["total_conversions"] = totals["conversions"]

    if totals["conversions"] > 0:
        summary["blended_cpa"] = round(totals["spend"] / totals["conversions"], 2)
    if totals["impressions"] > 0:
        summary["blended_cpm"] = round(totals["spend"] / totals["impressions"] * 1000.0, 2)
        summary["blended_ctr"] = round(totals["clicks"] / totals["impressions"], 4)
    if totals["clicks"] > 0:
        summary["blended_cvr"] = round(totals["conversions"] / totals["clicks"], 4)

    for key, src in [
        ("avg_pacing", "pacing_calc" if "pacing_calc" in df.columns else "pacing"),
        ("avg_viewability", "viewability"),
        ("avg_out_of_geo", "out_of_geo"),
        ("avg_ivt", "ivt"),
    ]:
        if src and src in df.columns:
            val = pd.to_numeric(df[src], errors="coerce").mean()
            summary[key] = round(float(val), 4) if pd.notna(val) else None

    return summary
