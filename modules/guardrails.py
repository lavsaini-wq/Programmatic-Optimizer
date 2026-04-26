"""
Guardrails for the recommendation engine.

This module returns a list of "do not change" rules that the app must
always show alongside any recommendation, and helpers that turn the
current pipeline state into context-specific warnings.

The point is to keep a human in the loop on anything risky — even when
the AI summary returns confidently.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


GUARDRAILS: List[dict] = [
    {
        "category": "Auto-changes",
        "rule": "Do not pause campaigns automatically",
        "why": "Pausing without context can break learning, lose pacing on key flights, or violate IO commitments. Always confirm with the trader.",
    },
    {
        "category": "Auto-changes",
        "rule": "Do not change budgets automatically",
        "why": "Budget changes alter pacing curves and can trip pacing models. Get explicit approval before adjusting.",
    },
    {
        "category": "Auto-changes",
        "rule": "Do not raise bids automatically — propose, then review",
        "why": "Bid increases on broken deals or noisy ZIPs amplify waste. Confirm the diagnosis before executing.",
    },
    {
        "category": "Brand safety",
        "rule": "Do not remove DV / IAS / viewability / brand-safety rails without written approval",
        "why": "These protect the brand and the measurement setup. Removing them silently is a major risk.",
    },
    {
        "category": "Brand safety",
        "rule": "Do not loosen geo, frequency, or audience caps to chase pacing",
        "why": "These were set in the build doc for a reason. Loosening them often shifts spend to low-quality inventory.",
    },
    {
        "category": "Data quality",
        "rule": "Do not exclude sites or ZIPs that lack enough data",
        "why": "Exclusions on thin data throw away high-potential inventory. Honor the minimum spend / impression thresholds.",
    },
    {
        "category": "Data quality",
        "rule": "Do not act on KPIs that are missing or look like coercion errors",
        "why": "If a column failed to parse (e.g. text in a numeric field) the resulting metric is unreliable. Re-check the source file first.",
    },
    {
        "category": "Evidence",
        "rule": "Do not use web case studies as stronger evidence than campaign data",
        "why": "Case studies are anecdotal and context-specific. They are useful as supporting context, not as a primary signal.",
    },
    {
        "category": "Process",
        "rule": "Do not bypass the build doc constraints",
        "why": "Frequency caps, geo, dayparting and audience rules in the build doc reflect client commitments. Honor them.",
    },
    {
        "category": "Process",
        "rule": "All recommendations require a human reviewer",
        "why": "This app produces suggestions only. No DSP changes are made automatically.",
    },
]


def get_do_not_change_df() -> pd.DataFrame:
    """Return guardrails as a Pandas DataFrame (used in dashboard + Excel)."""
    return pd.DataFrame(GUARDRAILS)


def context_warnings(
    *,
    qa_df: Optional[pd.DataFrame],
    site_recs: Optional[pd.DataFrame],
    zip_remove: Optional[pd.DataFrame],
    zip_add: Optional[pd.DataFrame],
    pmp_review_df: Optional[pd.DataFrame],
    final_recommendation: Optional[Dict],
) -> List[str]:
    """Build context-specific warnings to show next to the static guardrails."""
    warnings: List[str] = []

    if qa_df is not None and not qa_df.empty and "missing_pct" in qa_df.columns:
        bad = qa_df[qa_df["missing_pct"] >= 25]
        for _, row in bad.iterrows():
            warnings.append(
                f"⚠️ {row['report']} has {row['missing_pct']}% missing values — "
                "verify the source file before acting on its recommendations."
            )
        miss_col = qa_df[qa_df.get("missing_required_fields", "").astype(str).str.len() > 0] \
            if "missing_required_fields" in qa_df.columns else pd.DataFrame()
        for _, row in miss_col.iterrows():
            warnings.append(
                f"⚠️ {row['report']} is missing required field(s): "
                f"{row['missing_required_fields']}"
            )

    def _low_conf_count(df, threshold=0.4):
        if df is None or df.empty or "confidence" not in df.columns:
            return 0
        return int((df["confidence"] < threshold).sum())

    n = _low_conf_count(site_recs)
    if n:
        warnings.append(
            f"⚠️ {n} site exclusion(s) are low confidence (<40%) — re-check before excluding."
        )
    n = _low_conf_count(zip_remove)
    if n:
        warnings.append(
            f"⚠️ {n} ZIP removal(s) are low confidence — confirm the geo signal first."
        )
    if zip_add is not None and not zip_add.empty:
        warnings.append(
            "ℹ️ ZIP expansion is a hypothesis based on pacing + healthy quality — "
            "start with a small allocation and monitor."
        )
    if pmp_review_df is not None and not pmp_review_df.empty and "severity" in pmp_review_df.columns:
        high = int((pmp_review_df["severity"] == "High").sum())
        if high:
            warnings.append(
                f"🚨 {high} PMP deal(s) flagged High severity — likely floor mismatch "
                "or unwinnable; renegotiate before increasing spend."
            )

    if final_recommendation:
        score = float(final_recommendation.get("confidence_score") or 0)
        if score < 0.4:
            warnings.append(
                "⚠️ AI confidence is low — treat the AI summary as a draft and rely on "
                "the rule-based tables."
            )

    return warnings
