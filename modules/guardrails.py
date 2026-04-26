"""
Guardrails for the recommendation engine.

This module is intentionally not data-driven — it returns a fixed list of
"do not change" rules that the app must always show alongside any
recommendation. The point is to keep a human in the loop on anything
risky.
"""

from __future__ import annotations

from typing import List

import pandas as pd


GUARDRAILS: List[dict] = [
    {
        "rule": "Do not pause campaigns automatically",
        "why": "Pausing a campaign without context can break learning, "
               "lose pacing on key flights, or violate IO commitments. "
               "Always confirm with the trader.",
    },
    {
        "rule": "Do not change budgets automatically",
        "why": "Budget changes can alter pacing curves and trip pacing "
               "models. Get explicit approval before adjusting.",
    },
    {
        "rule": "Do not remove DV / viewability / brand-safety rails without approval",
        "why": "These protect the brand and the measurement setup. "
               "Removing them silently is a major risk.",
    },
    {
        "rule": "Do not exclude sites or ZIPs that lack enough data",
        "why": "Exclusions on thin data can throw away high-potential "
               "inventory. Apply minimum spend / impression thresholds.",
    },
    {
        "rule": "Do not use web case studies as stronger evidence than campaign data",
        "why": "Case studies are anecdotal and context-specific. They "
               "are useful as supporting context, not as a primary signal.",
    },
    {
        "rule": "Do not bypass the build doc constraints",
        "why": "Frequency caps, geo, dayparting and audience rules in "
               "the build doc reflect client commitments. Honor them.",
    },
    {
        "rule": "All recommendations require a human reviewer",
        "why": "This app produces suggestions only. No DSP changes are "
               "made automatically.",
    },
]


def get_do_not_change_df() -> pd.DataFrame:
    """Return guardrails as a Pandas DataFrame (used in dashboard + Excel)."""
    return pd.DataFrame(GUARDRAILS)
