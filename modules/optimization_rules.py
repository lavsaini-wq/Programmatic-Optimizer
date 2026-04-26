"""
Rule-based optimization engine.

These functions take cleaned + KPI-enriched DataFrames and return
*recommendations* (suggestions only — never applied to a DSP).

Rule categories:
- Campaign health classification
- Site exclusion candidates
- Zip removal candidates
- Zip expansion candidates
- PMP deal review flags
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Campaign health
# ---------------------------------------------------------------------------
def classify_campaign_health(
    campaign_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    """Add health_status + reason columns to the campaign-level frame."""
    if campaign_df is None or campaign_df.empty:
        return campaign_df

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    df = campaign_df.copy()
    pacing = pd.to_numeric(
        df["pacing_calc"] if "pacing_calc" in df.columns else df.get("pacing"),
        errors="coerce",
    )
    cpa = pd.to_numeric(
        df["cpa_calc"] if "cpa_calc" in df.columns else df.get("cpa"),
        errors="coerce",
    )

    statuses, reasons = [], []
    for p, c in zip(pacing, cpa):
        if pd.isna(p):
            statuses.append("Unknown")
            reasons.append("Pacing data missing")
            continue
        cpa_above = (not pd.isna(c)) and cpa_benchmark > 0 and c > cpa_benchmark
        cpa_below = (not pd.isna(c)) and cpa_benchmark > 0 and c <= cpa_benchmark

        if p < 0.9 and cpa_below:
            statuses.append("Underpacing but efficient")
            reasons.append(
                f"Pacing {p:.0%} below 90% but CPA {c:.2f} <= benchmark {cpa_benchmark:.2f}"
            )
        elif p < 0.9 and cpa_above:
            statuses.append("Underpacing and inefficient")
            reasons.append(
                f"Pacing {p:.0%} below 90% and CPA {c:.2f} > benchmark {cpa_benchmark:.2f}"
            )
        elif 0.9 <= p <= 1.1 and cpa_below:
            statuses.append("Healthy")
            reasons.append(
                f"Pacing {p:.0%} on target and CPA {c:.2f} within benchmark"
            )
        elif p > 1.1 and cpa_above:
            statuses.append("Overspending and inefficient")
            reasons.append(
                f"Pacing {p:.0%} above 110% and CPA {c:.2f} > benchmark {cpa_benchmark:.2f}"
            )
        elif p > 1.1 and cpa_below:
            statuses.append("Overspending but efficient")
            reasons.append(
                f"Pacing {p:.0%} above 110% but CPA {c:.2f} within benchmark"
            )
        else:
            statuses.append("Review")
            reasons.append("Mixed signals — review manually")

    df["health_status"] = statuses
    df["health_reason"] = reasons
    return df


# ---------------------------------------------------------------------------
# Site exclusion candidates
# ---------------------------------------------------------------------------
def site_exclusion_candidates(
    site_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    """Flag sites/apps that meet *all* the exclusion criteria."""
    if site_df is None or site_df.empty:
        return pd.DataFrame()

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    view_goal = float(benchmarks.get("viewability_goal") or 0)
    geo_thresh = float(benchmarks.get("out_of_geo_threshold") or 0)
    ivt_thresh = float(benchmarks.get("ivt_threshold") or 0)
    min_spend = float(benchmarks.get("min_spend_threshold") or 0)

    df = site_df.copy()
    spend = pd.to_numeric(df.get("spend"), errors="coerce").fillna(0)
    conv = pd.to_numeric(df.get("conversions"), errors="coerce").fillna(0)
    cpa = pd.to_numeric(
        df["cpa_calc"] if "cpa_calc" in df.columns else df.get("cpa"),
        errors="coerce",
    )
    view = pd.to_numeric(df.get("viewability"), errors="coerce")
    geo = pd.to_numeric(df.get("out_of_geo"), errors="coerce")
    ivt = pd.to_numeric(df.get("ivt"), errors="coerce")

    cond_spend = spend > min_spend
    cond_perf = (conv == 0) | ((cpa_benchmark > 0) & (cpa > 2 * cpa_benchmark))
    cond_quality = (
        ((view_goal > 0) & view.lt(view_goal))
        | ((geo_thresh > 0) & geo.gt(geo_thresh))
        | ((ivt_thresh > 0) & ivt.gt(ivt_thresh))
    )

    mask = cond_spend & cond_perf & cond_quality.fillna(False)
    candidates = df[mask].copy()
    if candidates.empty:
        return candidates

    reasons = []
    for _, row in candidates.iterrows():
        bits = []
        if pd.notna(row.get("conversions")) and row.get("conversions", 0) == 0:
            bits.append("0 conversions")
        if cpa_benchmark > 0 and pd.notna(row.get("cpa_calc", row.get("cpa"))):
            v = row.get("cpa_calc", row.get("cpa"))
            if v > 2 * cpa_benchmark:
                bits.append(f"CPA {v:.2f} > 2x benchmark")
        if view_goal > 0 and pd.notna(row.get("viewability")) and row["viewability"] < view_goal:
            bits.append(f"viewability {row['viewability']:.0%} < goal")
        if geo_thresh > 0 and pd.notna(row.get("out_of_geo")) and row["out_of_geo"] > geo_thresh:
            bits.append(f"out-of-geo {row['out_of_geo']:.0%} > threshold")
        if ivt_thresh > 0 and pd.notna(row.get("ivt")) and row["ivt"] > ivt_thresh:
            bits.append(f"IVT {row['ivt']:.0%} > threshold")
        reasons.append("; ".join(bits) or "Meets exclusion criteria")

    candidates["recommendation"] = "Exclude (review first)"
    candidates["reason"] = reasons
    return candidates


# ---------------------------------------------------------------------------
# Zip removal / expansion candidates
# ---------------------------------------------------------------------------
def zip_removal_candidates(
    zip_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    if zip_df is None or zip_df.empty:
        return pd.DataFrame()

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    geo_thresh = float(benchmarks.get("out_of_geo_threshold") or 0)
    min_spend = float(benchmarks.get("min_spend_threshold") or 0)

    df = zip_df.copy()
    spend = pd.to_numeric(df.get("spend"), errors="coerce").fillna(0)
    conv = pd.to_numeric(df.get("conversions"), errors="coerce").fillna(0)
    cpa = pd.to_numeric(
        df["cpa_calc"] if "cpa_calc" in df.columns else df.get("cpa"),
        errors="coerce",
    )
    geo = pd.to_numeric(df.get("out_of_geo"), errors="coerce").fillna(0)
    impressions = pd.to_numeric(df.get("impressions"), errors="coerce").fillna(0)
    clicks = pd.to_numeric(df.get("clicks"), errors="coerce").fillna(0)

    cond_spend = spend > min_spend
    cond_perf = (conv == 0) | ((cpa_benchmark > 0) & (cpa > cpa_benchmark))
    cond_geo = (geo_thresh > 0) & (geo > geo_thresh)
    cond_volume = (impressions >= 1000) | (clicks >= 50)

    mask = cond_spend & cond_perf & cond_geo & cond_volume
    out = df[mask].copy()
    if out.empty:
        return out
    out["recommendation"] = "Remove or reduce ZIP"
    out["reason"] = "Spend with poor CPA/conv and high out-of-geo"
    return out


def zip_expansion_candidates(
    approved_zip_df: pd.DataFrame,
    current_zip_df: pd.DataFrame,
    campaign_health_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    """Suggest ZIPs to add when the campaign is underpacing but efficient."""
    if approved_zip_df is None or approved_zip_df.empty:
        return pd.DataFrame()

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    view_goal = float(benchmarks.get("viewability_goal") or 0)
    geo_thresh = float(benchmarks.get("out_of_geo_threshold") or 0)

    if campaign_health_df is None or campaign_health_df.empty:
        return pd.DataFrame()

    pacing = pd.to_numeric(campaign_health_df.get("pacing_calc"), errors="coerce")
    cpa = pd.to_numeric(campaign_health_df.get("cpa_calc"), errors="coerce")
    view = pd.to_numeric(campaign_health_df.get("viewability"), errors="coerce")
    geo = pd.to_numeric(campaign_health_df.get("out_of_geo"), errors="coerce")

    underpacing = (pacing < 0.9).any() if len(pacing) else False
    cpa_ok = ((cpa <= cpa_benchmark) | cpa.isna()).all() if cpa_benchmark > 0 else True
    view_ok = ((view >= view_goal) | view.isna()).all() if view_goal > 0 else True
    geo_ok = ((geo <= geo_thresh) | geo.isna()).all() if geo_thresh > 0 else True

    if not (underpacing and cpa_ok and view_ok and geo_ok):
        return pd.DataFrame()

    approved = approved_zip_df.copy()
    if "zip" not in approved.columns:
        return pd.DataFrame()

    used_zips = set()
    if current_zip_df is not None and not current_zip_df.empty and "zip" in current_zip_df.columns:
        used_zips = set(current_zip_df["zip"].dropna().astype(str))

    new = approved[~approved["zip"].astype(str).isin(used_zips)].copy()
    if new.empty:
        return new
    new["recommendation"] = "Add to active targeting"
    new["reason"] = "Campaign underpacing while CPA, viewability and geo are healthy"
    return new


# ---------------------------------------------------------------------------
# PMP review
# ---------------------------------------------------------------------------
def pmp_review(
    pmp_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    if pmp_df is None or pmp_df.empty:
        return pd.DataFrame()

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    view_goal = float(benchmarks.get("viewability_goal") or 0)
    min_spend = float(benchmarks.get("min_spend_threshold") or 0)

    df = pmp_df.copy()
    spend = pd.to_numeric(df.get("spend"), errors="coerce").fillna(0)
    bid = pd.to_numeric(df.get("bid"), errors="coerce")
    floor = pd.to_numeric(df.get("floor_cpm"), errors="coerce")
    win = pd.to_numeric(df.get("win_rate"), errors="coerce")
    cpa = pd.to_numeric(df.get("cpa"), errors="coerce")
    view = pd.to_numeric(df.get("viewability"), errors="coerce")

    flags: List[str] = []
    recs: List[str] = []
    for i in range(len(df)):
        bits = []
        if pd.notna(floor.iloc[i]) and pd.notna(bid.iloc[i]) and floor.iloc[i] > bid.iloc[i]:
            bits.append("floor CPM > bid")
        if pd.notna(win.iloc[i]) and win.iloc[i] < 0.10:
            bits.append(f"low win rate {win.iloc[i]:.0%}")
        if min_spend > 0 and spend.iloc[i] < min_spend * 0.25:
            bits.append("very low spend")
        if cpa_benchmark > 0 and pd.notna(cpa.iloc[i]) and cpa.iloc[i] > cpa_benchmark:
            bits.append(f"CPA {cpa.iloc[i]:.2f} > benchmark")
        if view_goal > 0 and pd.notna(view.iloc[i]) and view.iloc[i] < view_goal:
            bits.append(f"viewability {view.iloc[i]:.0%} < goal")
        flags.append("; ".join(bits) if bits else "")
        if not bits:
            if pd.notna(cpa.iloc[i]) and cpa_benchmark > 0 and cpa.iloc[i] <= cpa_benchmark and spend.iloc[i] < min_spend:
                recs.append("Strong KPI but low scale — consider raising bid or budget")
            else:
                recs.append("No action")
        else:
            recs.append("Review deal — see flagged issues")

    df["flags"] = flags
    df["recommendation"] = recs
    return df


# ---------------------------------------------------------------------------
# Strong / poor zip helpers (used in KPI tab)
# ---------------------------------------------------------------------------
def strong_zips(zip_df: pd.DataFrame, benchmarks: Dict) -> pd.DataFrame:
    if zip_df is None or zip_df.empty:
        return pd.DataFrame()
    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    df = zip_df.copy()
    cpa = pd.to_numeric(df.get("cpa_calc", df.get("cpa")), errors="coerce")
    conv = pd.to_numeric(df.get("conversions"), errors="coerce").fillna(0)
    if cpa_benchmark <= 0:
        return pd.DataFrame()
    return df[(conv > 0) & (cpa <= cpa_benchmark)].copy()


def poor_zips(zip_df: pd.DataFrame, benchmarks: Dict) -> pd.DataFrame:
    if zip_df is None or zip_df.empty:
        return pd.DataFrame()
    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    df = zip_df.copy()
    cpa = pd.to_numeric(df.get("cpa_calc", df.get("cpa")), errors="coerce")
    conv = pd.to_numeric(df.get("conversions"), errors="coerce").fillna(0)
    return df[(conv == 0) | ((cpa_benchmark > 0) & (cpa > 2 * cpa_benchmark))].copy()


# ---------------------------------------------------------------------------
# Pacing analysis convenience frame
# ---------------------------------------------------------------------------
def build_pacing_table(campaign_df: pd.DataFrame) -> pd.DataFrame:
    if campaign_df is None or campaign_df.empty:
        return pd.DataFrame()
    cols = [
        c for c in [
            "campaign_id", "campaign_name", "budget", "spend",
            "expected_spend_to_date", "budget_remaining",
            "days_remaining", "daily_required_spend", "pacing_calc",
        ] if c in campaign_df.columns
    ]
    if not cols:
        return pd.DataFrame()
    return campaign_df[cols].copy()
