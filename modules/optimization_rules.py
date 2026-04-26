"""
Rule-based optimization engine.

These functions take cleaned + KPI-enriched DataFrames and return
*recommendations* (suggestions only — never applied to a DSP).

Rule categories:
- Campaign health classification + underpacing diagnosis
- Site exclusion candidates with a confidence score
- Zip removal candidates and zip expansion candidates (separated)
- PMP deal review with floor-mismatch severity
- Final recommendation prioritization (High / Medium / Low)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal math helpers
# ---------------------------------------------------------------------------
def _num(series_or_value):
    return pd.to_numeric(series_or_value, errors="coerce")


def _pick(df: pd.DataFrame, *names) -> pd.Series:
    """Return the first column in df that matches one of names, else NaN series."""
    for n in names:
        if n in df.columns:
            return _num(df[n])
    return pd.Series([np.nan] * len(df))


# ---------------------------------------------------------------------------
# Campaign health + underpacing diagnosis
# ---------------------------------------------------------------------------
def _diagnose_underpacing(row: pd.Series, benchmarks: Dict) -> str:
    """Return a short diagnosis string explaining WHY a campaign is underpacing."""
    bits: List[str] = []

    bid = pd.to_numeric(row.get("max_bid", row.get("base_bid", np.nan)), errors="coerce")
    cpm = pd.to_numeric(row.get("cpm_calc", row.get("cpm", np.nan)), errors="coerce")
    win = pd.to_numeric(row.get("win_rate", np.nan), errors="coerce")
    freq = pd.to_numeric(row.get("frequency", np.nan), errors="coerce")
    impressions = pd.to_numeric(row.get("impressions", np.nan), errors="coerce")
    spend = pd.to_numeric(row.get("spend", np.nan), errors="coerce")
    budget = pd.to_numeric(row.get("budget", np.nan), errors="coerce")
    daily_req = pd.to_numeric(row.get("daily_required_spend", np.nan), errors="coerce")

    if pd.notna(win) and win < 0.10:
        bits.append(f"low win rate ({win:.0%})")
    if pd.notna(cpm) and pd.notna(bid) and bid > 0 and cpm < 0.6 * bid:
        bits.append("clearing CPM well below max bid (room to bid up)")
    if pd.notna(freq) and freq >= 6:
        bits.append(f"high frequency ({freq:.1f}) — audience may be tapped out")
    if pd.notna(impressions) and impressions < 10000:
        bits.append("very low impression volume (audience or supply too narrow)")
    if pd.notna(daily_req) and pd.notna(spend) and pd.notna(budget) and budget > 0:
        if daily_req > (spend / max(1, (row.get("days_remaining") or 1))) * 1.5:
            bits.append("daily required spend much higher than recent run rate")
    if not bits:
        bits.append("Cause unclear — check supply, audience size, dayparting, and bid")
    return "; ".join(bits)


def classify_campaign_health(
    campaign_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    """Add health_status, health_reason and underpacing_diagnosis columns."""
    if campaign_df is None or campaign_df.empty:
        return campaign_df

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    df = campaign_df.copy()
    pacing = _pick(df, "pacing_calc", "pacing")
    cpa = _pick(df, "cpa_calc", "cpa")

    statuses, reasons, diagnoses = [], [], []
    for i in range(len(df)):
        p = pacing.iloc[i] if i < len(pacing) else np.nan
        c = cpa.iloc[i] if i < len(cpa) else np.nan
        row = df.iloc[i]

        if pd.isna(p):
            statuses.append("Unknown")
            reasons.append("Pacing data missing")
            diagnoses.append("Cannot diagnose without pacing data")
            continue
        cpa_above = (not pd.isna(c)) and cpa_benchmark > 0 and c > cpa_benchmark
        cpa_below = (not pd.isna(c)) and cpa_benchmark > 0 and c <= cpa_benchmark

        if p < 0.9 and cpa_below:
            statuses.append("Underpacing but efficient")
            reasons.append(
                f"Pacing {p:.0%} below 90% but CPA {c:.2f} <= benchmark {cpa_benchmark:.2f}"
            )
            diagnoses.append(_diagnose_underpacing(row, benchmarks))
        elif p < 0.9 and cpa_above:
            statuses.append("Underpacing and inefficient")
            reasons.append(
                f"Pacing {p:.0%} below 90% and CPA {c:.2f} > benchmark {cpa_benchmark:.2f}"
            )
            diagnoses.append(_diagnose_underpacing(row, benchmarks))
        elif p < 0.9:
            statuses.append("Underpacing")
            reasons.append(f"Pacing {p:.0%} below 90% (CPA unknown)")
            diagnoses.append(_diagnose_underpacing(row, benchmarks))
        elif 0.9 <= p <= 1.1 and cpa_below:
            statuses.append("Healthy")
            reasons.append(f"Pacing {p:.0%} on target and CPA {c:.2f} within benchmark")
            diagnoses.append("")
        elif p > 1.1 and cpa_above:
            statuses.append("Overspending and inefficient")
            reasons.append(
                f"Pacing {p:.0%} above 110% and CPA {c:.2f} > benchmark {cpa_benchmark:.2f}"
            )
            diagnoses.append("")
        elif p > 1.1 and cpa_below:
            statuses.append("Overspending but efficient")
            reasons.append(f"Pacing {p:.0%} above 110% but CPA {c:.2f} within benchmark")
            diagnoses.append("")
        else:
            statuses.append("Review")
            reasons.append("Mixed signals — review manually")
            diagnoses.append("")

    df["health_status"] = statuses
    df["health_reason"] = reasons
    df["underpacing_diagnosis"] = diagnoses
    return df


# ---------------------------------------------------------------------------
# Site exclusion candidates with confidence scoring
# ---------------------------------------------------------------------------
def _site_confidence(row: pd.Series, benchmarks: Dict) -> float:
    """Score 0..1 indicating how confident we are in excluding this site.

    Drivers: spend volume vs threshold, impression volume, severity of
    quality breaches (viewability / geo / IVT), CPA delta vs benchmark.
    """
    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    view_goal = float(benchmarks.get("viewability_goal") or 0)
    geo_thresh = float(benchmarks.get("out_of_geo_threshold") or 0)
    ivt_thresh = float(benchmarks.get("ivt_threshold") or 0)
    min_spend = float(benchmarks.get("min_spend_threshold") or 0) or 1.0

    spend = float(pd.to_numeric(row.get("spend", 0), errors="coerce") or 0)
    impressions = float(pd.to_numeric(row.get("impressions", 0), errors="coerce") or 0)
    conv = float(pd.to_numeric(row.get("conversions", 0), errors="coerce") or 0)
    cpa = pd.to_numeric(row.get("cpa_calc", row.get("cpa")), errors="coerce")
    view = pd.to_numeric(row.get("viewability"), errors="coerce")
    geo = pd.to_numeric(row.get("out_of_geo"), errors="coerce")
    ivt = pd.to_numeric(row.get("ivt"), errors="coerce")

    # Volume confidence (more spend / impressions => more confidence)
    spend_ratio = min(spend / min_spend, 4.0) / 4.0           # 0..1
    impr_ratio = min(impressions / 25000.0, 1.0)              # 0..1
    volume = 0.6 * spend_ratio + 0.4 * impr_ratio

    # Performance signal strength
    perf = 0.0
    if conv == 0 and spend > min_spend:
        perf = max(perf, min(spend / (3 * min_spend), 1.0))
    if cpa_benchmark > 0 and pd.notna(cpa):
        if cpa > 3 * cpa_benchmark:
            perf = max(perf, 1.0)
        elif cpa > 2 * cpa_benchmark:
            perf = max(perf, 0.7)
        elif cpa > 1.5 * cpa_benchmark:
            perf = max(perf, 0.4)

    # Quality signal strength
    quality = 0.0
    if view_goal > 0 and pd.notna(view):
        gap = view_goal - view
        if gap > 0:
            quality = max(quality, min(gap / max(view_goal, 0.01), 1.0))
    if geo_thresh > 0 and pd.notna(geo):
        excess = geo - geo_thresh
        if excess > 0:
            quality = max(quality, min(excess / max(geo_thresh, 0.01), 1.0))
    if ivt_thresh > 0 and pd.notna(ivt):
        excess = ivt - ivt_thresh
        if excess > 0:
            quality = max(quality, min(excess / max(ivt_thresh, 0.005), 1.0))

    score = 0.45 * volume + 0.35 * perf + 0.20 * quality
    return round(min(max(score, 0.0), 1.0), 2)


def site_exclusion_candidates(
    site_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    if site_df is None or site_df.empty:
        return pd.DataFrame()

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    view_goal = float(benchmarks.get("viewability_goal") or 0)
    geo_thresh = float(benchmarks.get("out_of_geo_threshold") or 0)
    ivt_thresh = float(benchmarks.get("ivt_threshold") or 0)
    min_spend = float(benchmarks.get("min_spend_threshold") or 0)

    df = site_df.copy()
    spend = _num(df.get("spend")).fillna(0)
    conv = _num(df.get("conversions")).fillna(0)
    cpa = _pick(df, "cpa_calc", "cpa")
    view = _num(df.get("viewability"))
    geo = _num(df.get("out_of_geo"))
    ivt = _num(df.get("ivt"))

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

    reasons, scores, priorities = [], [], []
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
        score = _site_confidence(row, benchmarks)
        scores.append(score)
        priorities.append("High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low")

    candidates["recommendation"] = "Exclude (review first)"
    candidates["reason"] = reasons
    candidates["confidence"] = scores
    candidates["priority"] = priorities
    candidates = candidates.sort_values(["confidence"], ascending=False)
    return candidates.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Zip removal vs expansion (clearly separated)
# ---------------------------------------------------------------------------
def zip_removal_candidates(
    zip_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    """ZIPs we want to *remove or reduce* — wasted spend + geo leakage."""
    if zip_df is None or zip_df.empty:
        return pd.DataFrame()

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    geo_thresh = float(benchmarks.get("out_of_geo_threshold") or 0)
    min_spend = float(benchmarks.get("min_spend_threshold") or 0)

    df = zip_df.copy()
    spend = _num(df.get("spend")).fillna(0)
    conv = _num(df.get("conversions")).fillna(0)
    cpa = _pick(df, "cpa_calc", "cpa")
    geo = _num(df.get("out_of_geo")).fillna(0)
    impressions = _num(df.get("impressions")).fillna(0)
    clicks = _num(df.get("clicks")).fillna(0)

    cond_spend = spend > min_spend
    cond_perf = (conv == 0) | ((cpa_benchmark > 0) & (cpa > cpa_benchmark))
    cond_geo = (geo_thresh > 0) & (geo > geo_thresh)
    cond_volume = (impressions >= 1000) | (clicks >= 50)

    mask = cond_spend & cond_perf & cond_geo & cond_volume
    out = df[mask].copy()
    if out.empty:
        return out

    # Confidence based on geo excess + spend
    confidences, priorities, reasons = [], [], []
    for _, row in out.iterrows():
        s = float(row.get("spend") or 0)
        g = float(row.get("out_of_geo") or 0)
        c = pd.to_numeric(row.get("cpa_calc", row.get("cpa")), errors="coerce")

        spend_factor = min(s / max(min_spend, 1.0), 4.0) / 4.0
        geo_factor = min((g - geo_thresh) / max(geo_thresh, 0.01), 1.0) if geo_thresh else 0
        cpa_factor = 0.0
        if cpa_benchmark > 0 and pd.notna(c) and c > cpa_benchmark:
            cpa_factor = min((c - cpa_benchmark) / cpa_benchmark, 1.0)
        score = round(0.45 * spend_factor + 0.35 * geo_factor + 0.20 * cpa_factor, 2)
        confidences.append(score)
        priorities.append("High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low")
        bits = []
        if float(row.get("conversions") or 0) == 0:
            bits.append("0 conversions")
        if cpa_benchmark > 0 and pd.notna(c) and c > cpa_benchmark:
            bits.append(f"CPA {c:.2f} > benchmark")
        if g > geo_thresh:
            bits.append(f"out-of-geo {g:.0%} > threshold")
        reasons.append("; ".join(bits) or "Meets removal criteria")

    out["recommendation"] = "Remove or reduce ZIP"
    out["reason"] = reasons
    out["confidence"] = confidences
    out["priority"] = priorities
    return out.sort_values(["confidence"], ascending=False).reset_index(drop=True)


def zip_expansion_candidates(
    approved_zip_df: pd.DataFrame,
    current_zip_df: pd.DataFrame,
    campaign_health_df: pd.DataFrame,
    benchmarks: Dict,
) -> pd.DataFrame:
    """ZIPs we want to *add*.

    Only triggers when the campaign is underpacing AND quality is healthy
    AND the campaign-level spend is large enough to indicate the issue is
    not just "we have no data yet". This guards against expanding into
    unknown ZIPs based on noise.
    """
    if approved_zip_df is None or approved_zip_df.empty:
        return pd.DataFrame()
    if "zip" not in approved_zip_df.columns:
        return pd.DataFrame()
    if campaign_health_df is None or campaign_health_df.empty:
        return pd.DataFrame()

    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    view_goal = float(benchmarks.get("viewability_goal") or 0)
    geo_thresh = float(benchmarks.get("out_of_geo_threshold") or 0)
    min_spend = float(benchmarks.get("min_spend_threshold") or 0)

    pacing = _pick(campaign_health_df, "pacing_calc", "pacing")
    cpa = _pick(campaign_health_df, "cpa_calc", "cpa")
    view = _num(campaign_health_df.get("viewability"))
    geo = _num(campaign_health_df.get("out_of_geo"))
    spend = _num(campaign_health_df.get("spend")).fillna(0)

    underpacing = (pacing.dropna() < 0.9).any() if not pacing.dropna().empty else False
    cpa_ok = ((cpa <= cpa_benchmark) | cpa.isna()).all() if cpa_benchmark > 0 else True
    view_ok = ((view >= view_goal) | view.isna()).all() if view_goal > 0 else True
    geo_ok = ((geo <= geo_thresh) | geo.isna()).all() if geo_thresh > 0 else True
    has_signal = float(spend.sum()) >= max(min_spend, 1.0) * 4

    if not (underpacing and cpa_ok and view_ok and geo_ok and has_signal):
        return pd.DataFrame()

    used_zips: set = set()
    if current_zip_df is not None and not current_zip_df.empty and "zip" in current_zip_df.columns:
        used_zips = set(current_zip_df["zip"].dropna().astype(str))

    new = approved_zip_df[~approved_zip_df["zip"].astype(str).isin(used_zips)].copy()
    if new.empty:
        return new

    # Lower-confidence by nature; this is a hypothesis, not a cleanup.
    deficit = max(0.0, 1.0 - float(pacing.dropna().mean() or 1.0))
    base_score = round(min(0.4 + 1.5 * deficit, 0.85), 2)
    new["recommendation"] = "Add to active targeting"
    new["reason"] = (
        "Campaign underpacing while CPA, viewability and geo are healthy — "
        "approved ZIPs not currently in rotation"
    )
    new["confidence"] = base_score
    new["priority"] = (
        "High" if base_score >= 0.7 else "Medium" if base_score >= 0.4 else "Low"
    )
    return new.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PMP review with floor mismatch severity
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
    spend = _num(df.get("spend")).fillna(0)
    bid = _num(df.get("bid"))
    floor = _num(df.get("floor_cpm"))
    win = _num(df.get("win_rate"))
    cpa = _num(df.get("cpa"))
    view = _num(df.get("viewability"))

    flags: List[str] = []
    recs: List[str] = []
    severities: List[str] = []
    floor_gap_pct: List[Optional[float]] = []
    confidences: List[float] = []
    priorities: List[str] = []

    for i in range(len(df)):
        bits = []
        sev_score = 0.0  # 0..1, drives severity + priority

        # Floor mismatch
        gap_pct: Optional[float] = None
        if pd.notna(floor.iloc[i]) and pd.notna(bid.iloc[i]) and bid.iloc[i] > 0:
            gap = (floor.iloc[i] - bid.iloc[i]) / bid.iloc[i]
            gap_pct = round(float(gap), 4)
            if gap > 0.20:
                bits.append(f"floor CPM {gap:.0%} above bid")
                sev_score = max(sev_score, 1.0)
            elif gap > 0.05:
                bits.append(f"floor CPM {gap:.0%} above bid")
                sev_score = max(sev_score, 0.7)
            elif gap > -0.05:
                bits.append("floor CPM near bid (thin headroom)")
                sev_score = max(sev_score, 0.4)
        floor_gap_pct.append(gap_pct)

        # Low win rate
        if pd.notna(win.iloc[i]):
            w = win.iloc[i]
            if w < 0.05:
                bits.append(f"very low win rate {w:.0%}")
                sev_score = max(sev_score, 0.9)
            elif w < 0.10:
                bits.append(f"low win rate {w:.0%}")
                sev_score = max(sev_score, 0.6)

        # Very low spend (deal not delivering)
        if min_spend > 0 and spend.iloc[i] < min_spend * 0.25:
            bits.append("very low spend")
            sev_score = max(sev_score, 0.5)

        # CPA poor
        if cpa_benchmark > 0 and pd.notna(cpa.iloc[i]) and cpa.iloc[i] > cpa_benchmark:
            ratio = cpa.iloc[i] / cpa_benchmark
            bits.append(f"CPA {cpa.iloc[i]:.2f} {ratio:.1f}x benchmark")
            sev_score = max(sev_score, min(0.4 + 0.3 * (ratio - 1), 1.0))

        # Viewability
        if view_goal > 0 and pd.notna(view.iloc[i]) and view.iloc[i] < view_goal:
            bits.append(f"viewability {view.iloc[i]:.0%} < goal")
            sev_score = max(sev_score, 0.5)

        flags.append("; ".join(bits))

        # Recommendation logic
        good_kpi = cpa_benchmark > 0 and pd.notna(cpa.iloc[i]) and cpa.iloc[i] <= cpa_benchmark
        low_scale = min_spend > 0 and spend.iloc[i] < min_spend
        if not bits:
            if good_kpi and low_scale:
                recs.append("Strong KPI but low scale — propose higher bid or budget")
                sev_score = 0.4
            else:
                recs.append("No action")
        else:
            if pd.notna(floor.iloc[i]) and pd.notna(bid.iloc[i]) and floor.iloc[i] > bid.iloc[i]:
                recs.append("Renegotiate floor or pause this deal — bid cannot clear")
            elif pd.notna(win.iloc[i]) and win.iloc[i] < 0.05:
                recs.append("Investigate creative/format compatibility and bid strategy")
            else:
                recs.append("Review deal — see flagged issues")

        severities.append(
            "High" if sev_score >= 0.75 else "Medium" if sev_score >= 0.4 else "Low"
        )
        confidences.append(round(sev_score, 2))
        priorities.append(severities[-1])

    df["floor_gap_pct"] = floor_gap_pct
    df["flags"] = flags
    df["recommendation"] = recs
    df["severity"] = severities
    df["confidence"] = confidences
    df["priority"] = priorities
    return df.sort_values(["confidence"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strong / poor zip helpers (used in KPI tab)
# ---------------------------------------------------------------------------
def strong_zips(zip_df: pd.DataFrame, benchmarks: Dict) -> pd.DataFrame:
    if zip_df is None or zip_df.empty:
        return pd.DataFrame()
    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    df = zip_df.copy()
    cpa = _pick(df, "cpa_calc", "cpa")
    conv = _num(df.get("conversions")).fillna(0)
    if cpa_benchmark <= 0:
        return pd.DataFrame()
    return df[(conv > 0) & (cpa <= cpa_benchmark)].copy()


def poor_zips(zip_df: pd.DataFrame, benchmarks: Dict) -> pd.DataFrame:
    if zip_df is None or zip_df.empty:
        return pd.DataFrame()
    cpa_benchmark = float(benchmarks.get("cpa_benchmark") or 0)
    df = zip_df.copy()
    cpa = _pick(df, "cpa_calc", "cpa")
    conv = _num(df.get("conversions")).fillna(0)
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
            "underpacing_diagnosis",
        ] if c in campaign_df.columns
    ]
    if not cols:
        return pd.DataFrame()
    return campaign_df[cols].copy()


# ---------------------------------------------------------------------------
# Final recommendation prioritization
# ---------------------------------------------------------------------------
def prioritize_recommendations(
    health_df: Optional[pd.DataFrame],
    site_recs: Optional[pd.DataFrame],
    zip_remove: Optional[pd.DataFrame],
    zip_add: Optional[pd.DataFrame],
    pmp_review_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a single ranked recommendation table with priority + confidence."""
    rows: List[Dict] = []

    # Campaign-level
    if health_df is not None and not health_df.empty:
        for _, row in health_df.iterrows():
            status = row.get("health_status")
            if status in (None, "Healthy"):
                continue
            priority = {
                "Underpacing and inefficient": "High",
                "Overspending and inefficient": "High",
                "Underpacing": "Medium",
                "Underpacing but efficient": "Medium",
                "Overspending but efficient": "Medium",
                "Review": "Low",
                "Unknown": "Low",
            }.get(status, "Medium")
            rows.append({
                "area": "campaign",
                "subject": row.get("campaign_name") or row.get("campaign_id") or "Campaign",
                "action": f"{status} — review pacing/CPA strategy",
                "rationale": row.get("health_reason", ""),
                "evidence": row.get("underpacing_diagnosis", ""),
                "confidence": 0.8 if status != "Unknown" else 0.2,
                "priority": priority,
            })

    # Site
    if site_recs is not None and not site_recs.empty:
        for _, row in site_recs.iterrows():
            rows.append({
                "area": "site",
                "subject": row.get("site") or row.get("app") or "Site",
                "action": row.get("recommendation", "Exclude"),
                "rationale": row.get("reason", ""),
                "evidence": (
                    f"spend={row.get('spend')} conv={row.get('conversions')} "
                    f"view={row.get('viewability')} geo={row.get('out_of_geo')}"
                ),
                "confidence": float(row.get("confidence") or 0),
                "priority": row.get("priority", "Medium"),
            })

    # Zip remove
    if zip_remove is not None and not zip_remove.empty:
        for _, row in zip_remove.iterrows():
            rows.append({
                "area": "zip",
                "subject": f"ZIP {row.get('zip')}",
                "action": row.get("recommendation", "Remove or reduce"),
                "rationale": row.get("reason", ""),
                "evidence": (
                    f"spend={row.get('spend')} conv={row.get('conversions')} "
                    f"geo={row.get('out_of_geo')}"
                ),
                "confidence": float(row.get("confidence") or 0),
                "priority": row.get("priority", "Medium"),
            })

    # Zip add
    if zip_add is not None and not zip_add.empty:
        sample = zip_add.head(20)
        rows.append({
            "area": "zip",
            "subject": f"{len(zip_add)} approved ZIPs",
            "action": "Add to active targeting (review list)",
            "rationale": (
                "Campaign underpacing with healthy quality — expand into "
                "approved ZIPs to capture pacing"
            ),
            "evidence": ", ".join(sample["zip"].astype(str).tolist()),
            "confidence": float(sample["confidence"].iloc[0]) if "confidence" in sample.columns else 0.5,
            "priority": sample["priority"].iloc[0] if "priority" in sample.columns else "Medium",
        })

    # PMP
    if pmp_review_df is not None and not pmp_review_df.empty:
        for _, row in pmp_review_df.iterrows():
            if row.get("recommendation") in (None, "No action"):
                continue
            rows.append({
                "area": "pmp",
                "subject": row.get("deal_name") or row.get("deal_id") or "PMP deal",
                "action": row.get("recommendation"),
                "rationale": row.get("flags", ""),
                "evidence": (
                    f"floor={row.get('floor_cpm')} bid={row.get('bid')} "
                    f"win_rate={row.get('win_rate')} cpa={row.get('cpa')}"
                ),
                "confidence": float(row.get("confidence") or 0),
                "priority": row.get("priority", "Medium"),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "priority", "area", "subject", "action", "rationale", "evidence", "confidence",
        ])

    out = pd.DataFrame(rows)
    pri_order = {"High": 0, "Medium": 1, "Low": 2}
    out["priority_rank"] = out["priority"].map(lambda p: pri_order.get(p, 3))
    out = out.sort_values(["priority_rank", "confidence"], ascending=[True, False])
    out = out.drop(columns=["priority_rank"])
    return out.reset_index(drop=True)
