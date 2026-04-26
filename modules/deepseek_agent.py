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
def _df_head_records(df: Optional[pd.DataFrame], n: int = 25) -> List[Dict]:
    """Convert the first n rows of a DataFrame into JSON-serializable dicts."""
    if df is None or df.empty:
        return []
    safe = df.head(n).copy()
    # Convert any non-JSON-friendly types
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
    build_doc_text: str,
    exclusion_summary: Dict,
    approved_zip_summary: Dict,
    case_studies: Optional[str],
) -> Dict:
    """Assemble the JSON-friendly payload sent to DeepSeek."""
    return {
        "kpi_summary": campaign_summary,
        "campaign_health": _df_head_records(health_df, 50),
        "site_recommendation_candidates": _df_head_records(site_recs, 50),
        "zip_remove_candidates": _df_head_records(zip_recs_remove, 50),
        "zip_add_candidates": _df_head_records(zip_recs_add, 50),
        "pmp_findings": _df_head_records(pmp_review_df, 50),
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
    "budgets, or remove brand-safety/verification settings. Use the "
    "rule-based findings provided as the primary evidence; treat case "
    "studies as weaker, supporting evidence only. Be specific, cite the "
    "relevant numbers, and call out missing data instead of guessing."
)

USER_INSTRUCTIONS = """\
Using the data below, produce a single JSON object with these keys:

- campaign_id (string or null)
- campaign_name (string or null)
- health_status (one of: "Healthy", "Underpacing but efficient",
  "Underpacing and inefficient", "Overspending and inefficient",
  "Overspending but efficient", "Review", "Unknown")
- confidence_score (float between 0 and 1 indicating how much you trust
  the recommendation given data quality and volume)
- executive_summary (2-4 sentence plain-English overview)
- top_issues (array of short strings)
- recommendations (array of objects with: area, action, rationale,
  evidence — each a string. "area" is one of "campaign", "site", "zip",
  "pmp", "creative", "measurement")
- do_not_change (array of short strings — guardrails the human reviewer
  should preserve, e.g. brand safety, viewability rails)
- human_next_steps (array of short strings the analyst should do next)

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
