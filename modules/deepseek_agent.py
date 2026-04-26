"""
DeepSeek reasoning layer.

Sends a *summary* of campaign performance + rule-based findings to the
DeepSeek chat completions API and asks for a structured recommendation
JSON document. We never send raw row-level data — only digests.

Configuration:
- DEEPSEEK_API_KEY     (required)
- DEEPSEEK_BASE_URL    (default https://api.deepseek.com)
- DEEPSEEK_MODEL       (default deepseek-v4-flash)

The OpenAI-compatible Python client is used to talk to DeepSeek.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - import error surfaced at runtime
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Whitelisted columns we are willing to send for each section. Anything
# else is dropped before serialization. Raw row-level reports are NEVER
# sent — we send aggregated summaries + top flagged rows only.
SAFE_COLUMNS: Dict[str, List[str]] = {
    "health": [
        "campaign_id", "campaign_name", "health_status", "health_reason",
        "underpacing_diagnosis", "pacing_calc", "cpa_calc", "cpm_calc",
        "viewability", "out_of_geo", "ivt", "spend", "budget",
        "budget_remaining", "daily_required_spend",
    ],
    "site": [
        "site", "app", "spend", "impressions", "conversions", "cpa_calc",
        "viewability", "out_of_geo", "ivt", "recommendation", "reason",
        "confidence", "priority",
    ],
    "zip_remove": [
        "zip", "dma", "spend", "impressions", "conversions", "cpa_calc",
        "out_of_geo", "recommendation", "reason", "confidence", "priority",
    ],
    "zip_add": ["zip", "dma", "recommendation", "reason", "confidence", "priority"],
    "pmp": [
        "deal_id", "deal_name", "publisher", "floor_cpm", "bid", "spend",
        "impressions", "win_rate", "cpa", "viewability", "floor_gap_pct",
        "flags", "recommendation", "severity", "confidence", "priority",
    ],
    "priorities": ["priority", "area", "subject", "action", "rationale", "confidence"],
}

# Hard limits on how many rows we'll send per section. Keeps the prompt
# small and ensures we are sending top *flagged* rows, not raw reports.
ROW_LIMITS: Dict[str, int] = {
    "health": 20,
    "site": 15,
    "zip_remove": 15,
    "zip_add": 15,
    "pmp": 15,
    "priorities": 25,
}


def _safe_records(
    df: Optional[pd.DataFrame],
    section: str,
) -> List[Dict]:
    """Return at most N JSON-safe records using only whitelisted columns."""
    if df is None or df.empty:
        return []
    cols = [c for c in SAFE_COLUMNS.get(section, []) if c in df.columns]
    if not cols:
        return []
    limit = ROW_LIMITS.get(section, 15)
    safe = df[cols].head(limit).copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].astype(str)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")


def build_summary_payload(
    campaign_summary: Dict,
    health_df: Optional[pd.DataFrame],
    site_recs: Optional[pd.DataFrame],
    zip_recs_remove: Optional[pd.DataFrame],
    zip_recs_add: Optional[pd.DataFrame],
    pmp_review_df: Optional[pd.DataFrame],
    prioritized_df: Optional[pd.DataFrame],
    build_doc_text: str,
    exclusion_summary: Dict,
    approved_zip_summary: Dict,
    qa_summary: Optional[List[Dict]],
    case_studies: Optional[str],
) -> Dict:
    """Assemble the JSON-friendly payload sent to DeepSeek.

    Only summaries + a small number of top *flagged* rows are sent —
    never the raw uploaded reports.
    """
    return {
        "kpi_summary": campaign_summary,
        "data_qa_summary": (qa_summary or [])[:10],
        "campaign_health_top": _safe_records(health_df, "health"),
        "site_recommendation_candidates_top": _safe_records(site_recs, "site"),
        "zip_remove_candidates_top": _safe_records(zip_recs_remove, "zip_remove"),
        "zip_add_candidates_top": _safe_records(zip_recs_add, "zip_add"),
        "pmp_findings_top": _safe_records(pmp_review_df, "pmp"),
        "prioritized_recommendations_top": _safe_records(prioritized_df, "priorities"),
        "build_doc_constraints": (build_doc_text or "")[:4000],
        "exclusion_list_summary": exclusion_summary,
        "approved_zip_list_summary": approved_zip_summary,
        "optional_case_study_summaries": (case_studies or "")[:2000],
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a senior programmatic advertising optimization analyst. "
    "You produce recommendations for a human reviewer. You never instruct "
    "the user to make automatic DSP changes, pause campaigns, change "
    "budgets, or remove brand-safety / verification settings. "
    "You ONLY receive aggregated metrics and the top flagged rows — "
    "never raw row-level reports. Use the rule-based findings as the "
    "primary evidence; treat web case studies as weaker, supporting "
    "evidence only. Be specific, cite the relevant numbers, and when "
    "data is missing or low-confidence, say so explicitly instead of "
    "guessing. Down-weight any recommendation whose rule-based "
    "confidence is below 0.4."
)

USER_INSTRUCTIONS = """\
Using the data below, produce a single JSON object with these keys:

- campaign_id (string or null)
- campaign_name (string or null)
- health_status (one of: "Healthy", "Underpacing", "Underpacing but efficient",
  "Underpacing and inefficient", "Overspending and inefficient",
  "Overspending but efficient", "Review", "Unknown")
- confidence_score (float between 0 and 1 indicating how much you trust
  the recommendation given data quality and volume)
- executive_summary (2-4 sentence plain-English overview)
- top_issues (array of short strings)
- recommendations (array of objects with these string fields:
  area (one of "campaign", "site", "zip", "pmp", "creative",
  "measurement", "audience"), priority ("High" | "Medium" | "Low"),
  action, rationale, evidence). Sort the array High → Low. Use the
  rule-based "prioritized_recommendations_top" as your primary input
  for this list.
- do_not_change (array of short strings — guardrails the human reviewer
  should preserve, e.g. brand safety, viewability rails)
- human_next_steps (array of short strings the analyst should do next,
  ordered most-important first)

Return ONLY valid JSON. No markdown fences, no prose outside the JSON.
"""


# ---------------------------------------------------------------------------
# DeepSeek call
# ---------------------------------------------------------------------------
def generate_recommendation(payload: Dict) -> Dict:
    """Call DeepSeek and return a parsed recommendation dict.

    On error (missing key, network error, bad JSON) we return a fallback
    dict that still has the required keys so the rest of the app keeps
    working.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()

    if not api_key:
        return _fallback("DEEPSEEK_API_KEY is not set. Add it to enable AI summaries.")

    if OpenAI is None:
        return _fallback("openai package is not installed.")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        USER_INSTRUCTIONS
                        + "\n\nDATA:\n"
                        + json.dumps(payload, default=str)[:60000]
                    ),
                },
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        return _parse_json(text)
    except Exception as exc:  # noqa: BLE001
        # Some DeepSeek model variants don't accept response_format; retry once
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            USER_INSTRUCTIONS
                            + "\n\nDATA:\n"
                            + json.dumps(payload, default=str)[:60000]
                        ),
                    },
                ],
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
            return _parse_json(text)
        except Exception as exc2:  # noqa: BLE001
            return _fallback(f"DeepSeek API error: {exc2 or exc}")


def _parse_json(text: str) -> Dict:
    """Try hard to parse a JSON object from a model response."""
    if not text:
        return _fallback("Empty response from DeepSeek")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip fenced code blocks
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to grab the first {...} block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return _fallback("Could not parse JSON from DeepSeek response")
        else:
            return _fallback("Could not parse JSON from DeepSeek response")
    return _ensure_keys(data)


def _ensure_keys(data: Dict) -> Dict:
    template = {
        "campaign_id": None,
        "campaign_name": None,
        "health_status": "Review",
        "confidence_score": 0.5,
        "executive_summary": "",
        "top_issues": [],
        "recommendations": [],
        "do_not_change": [],
        "human_next_steps": [],
    }
    for key, default in template.items():
        if key not in data or data[key] is None:
            data[key] = default
    return data


def _fallback(msg: str) -> Dict:
    return {
        "campaign_id": None,
        "campaign_name": None,
        "health_status": "Unknown",
        "confidence_score": 0.0,
        "executive_summary": (
            "AI summary unavailable; see rule-based findings in the other tabs."
        ),
        "top_issues": [msg],
        "recommendations": [],
        "do_not_change": [
            "Do not pause campaigns automatically",
            "Do not change budgets automatically",
            "Do not remove brand-safety / verification settings without human review",
        ],
        "human_next_steps": [
            "Review the rule-based recommendations in the other tabs",
            "Re-run the AI summary once the API key is configured",
        ],
        "error": msg,
    }
