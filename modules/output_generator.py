"""
Excel output generator.

Builds a single .xlsx file with all the recommendation tabs the analyst
needs to share. Uses XlsxWriter as the engine via Pandas.
"""

from __future__ import annotations

import io
from typing import Dict, Optional

import pandas as pd


SHEET_ORDER = [
    "Prioritized Recommendations",
    "Campaign Summary",
    "Pacing Analysis",
    "KPI Analysis",
    "Site Recommendations",
    "Zip Recommendations",
    "PMP Deal Review",
    "Final Recommendation",
    "Do Not Change",
    "Data QA",
]


def _safe_sheet_name(name: str) -> str:
    # Excel sheet names must be <= 31 chars, no [ ] : * ? / \ characters
    bad = '[]:*?/\\'
    cleaned = "".join("_" if c in bad else c for c in str(name))
    return cleaned[:31] or "Sheet"


def _df_or_placeholder(df: Optional[pd.DataFrame], placeholder_msg: str) -> pd.DataFrame:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame({"info": [placeholder_msg]})
    return df


def build_excel(
    *,
    campaign_summary_df: Optional[pd.DataFrame],
    pacing_df: Optional[pd.DataFrame],
    kpi_df: Optional[pd.DataFrame],
    site_recs_df: Optional[pd.DataFrame],
    zip_recs_df: Optional[pd.DataFrame],
    pmp_df: Optional[pd.DataFrame],
    final_recommendation: Dict,
    do_not_change_df: Optional[pd.DataFrame],
    qa_df: Optional[pd.DataFrame],
    prioritized_df: Optional[pd.DataFrame] = None,
) -> bytes:
    """Return the binary contents of the assembled .xlsx file."""

    # Convert the DeepSeek dict into a printable DataFrame
    final_rows = []
    if final_recommendation:
        final_rows.append({"field": "campaign_id", "value": final_recommendation.get("campaign_id")})
        final_rows.append({"field": "campaign_name", "value": final_recommendation.get("campaign_name")})
        final_rows.append({"field": "health_status", "value": final_recommendation.get("health_status")})
        final_rows.append({"field": "confidence_score", "value": final_recommendation.get("confidence_score")})
        final_rows.append({"field": "executive_summary", "value": final_recommendation.get("executive_summary")})
        for i, item in enumerate(final_recommendation.get("top_issues", []) or [], start=1):
            final_rows.append({"field": f"top_issue_{i}", "value": item})
        for i, rec in enumerate(final_recommendation.get("recommendations", []) or [], start=1):
            if isinstance(rec, dict):
                value = " | ".join(
                    f"{k}: {v}" for k, v in rec.items() if v is not None
                )
            else:
                value = str(rec)
            final_rows.append({"field": f"recommendation_{i}", "value": value})
        for i, item in enumerate(final_recommendation.get("human_next_steps", []) or [], start=1):
            final_rows.append({"field": f"next_step_{i}", "value": item})
    final_df = pd.DataFrame(final_rows) if final_rows else pd.DataFrame(
        {"info": ["No final recommendation generated yet"]}
    )

    sheets = {
        "Prioritized Recommendations": _df_or_placeholder(
            prioritized_df,
            "No prioritized recommendations — campaigns appear healthy or no data was provided",
        ),
        "Campaign Summary": _df_or_placeholder(campaign_summary_df, "No campaign report uploaded"),
        "Pacing Analysis": _df_or_placeholder(pacing_df, "No pacing data calculated"),
        "KPI Analysis": _df_or_placeholder(kpi_df, "No KPI data calculated"),
        "Site Recommendations": _df_or_placeholder(site_recs_df, "No site exclusion candidates"),
        "Zip Recommendations": _df_or_placeholder(zip_recs_df, "No zip recommendations"),
        "PMP Deal Review": _df_or_placeholder(pmp_df, "No PMP report uploaded"),
        "Final Recommendation": final_df,
        "Do Not Change": _df_or_placeholder(do_not_change_df, "No guardrails listed"),
        "Data QA": _df_or_placeholder(qa_df, "No data QA summary"),
    }

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#1f3b57", "font_color": "white",
            "border": 1,
        })
        wrap_fmt = wb.add_format({"text_wrap": True, "valign": "top"})

        for sheet_name in SHEET_ORDER:
            df = sheets.get(sheet_name)
            if df is None:
                df = pd.DataFrame({"info": ["No data"]})
            df.to_excel(writer, index=False, sheet_name=_safe_sheet_name(sheet_name))
            ws = writer.sheets[_safe_sheet_name(sheet_name)]
            # Header row
            for col_idx, col in enumerate(df.columns):
                ws.write(0, col_idx, str(col), header_fmt)
                # Auto width with a sensible cap
                max_len = max(
                    [len(str(col))]
                    + [len(str(v)) for v in df[col].astype(str).head(50).tolist()]
                )
                ws.set_column(col_idx, col_idx, min(max(12, max_len + 2), 60), wrap_fmt)
            ws.freeze_panes(1, 0)

    return buf.getvalue()
