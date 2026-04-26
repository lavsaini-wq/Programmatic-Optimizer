"""
Programmatic Optimization Recommendation Agent — Streamlit app.

This file wires the modules together:
- Upload reports
- Map columns
- Clean + calculate KPIs
- Run rule-based optimization
- Send a *summary* (never raw rows) to DeepSeek
- Show a dashboard and download a final Excel report

Important: this app NEVER changes anything in a DSP. It only produces
recommendations for a human to review.
"""

from __future__ import annotations

import io
import os
from datetime import date, timedelta
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from modules import (
    data_cleaner,
    deepseek_agent,
    guardrails,
    kpi_calculator,
    optimization_rules,
    output_generator,
)


# ---------------------------------------------------------------------------
# Page config + light helpers
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Programmatic Optimization Recommendation Agent",
    page_icon="📊",
    layout="wide",
)


REPORT_TYPES = [
    ("campaign",       "Campaign performance report"),
    ("ad_group",       "Ad group report"),
    ("site",           "Site / app report"),
    ("zip",            "Zip report"),
    ("pmp",            "PMP deal report"),
    ("build_doc",      "Build doc (PDF / DOCX / TXT)"),
    ("exclusion_list", "Exclusion list"),
    ("approved_zips",  "Approved zip list"),
    ("dv",             "DV report (optional)"),
    ("past_log",       "Past optimization log (optional)"),
]

STANDARD_FIELDS: Dict[str, list] = {
    "campaign": [
        "campaign_id", "campaign_name", "ad_group_id", "ad_group_name", "tactic",
        "flight_start", "flight_end", "budget", "spend", "impressions", "clicks",
        "conversions", "cpa", "cpm", "ctr", "cvr", "viewability", "out_of_geo",
        "ivt", "frequency", "base_bid", "max_bid", "pacing",
    ],
    "ad_group": [
        "campaign_id", "campaign_name", "ad_group_id", "ad_group_name", "tactic",
        "spend", "impressions", "clicks", "conversions", "cpa", "cpm", "ctr",
        "cvr", "viewability", "out_of_geo", "ivt",
    ],
    "site": [
        "site", "app", "spend", "impressions", "clicks", "conversions", "cpa",
        "ctr", "viewability", "out_of_geo", "ivt",
    ],
    "zip": [
        "zip", "dma", "spend", "impressions", "clicks", "conversions", "cpa",
        "out_of_geo", "approved_status",
    ],
    "pmp": [
        "deal_id", "deal_name", "publisher", "floor_cpm", "bid", "spend",
        "impressions", "win_rate", "cpa", "viewability",
    ],
    "exclusion_list": ["site", "app", "reason"],
    "approved_zips": ["zip", "dma"],
    "dv": ["site", "app", "viewability", "ivt", "out_of_geo", "brand_safety"],
    "past_log": ["date", "change", "rationale", "outcome"],
}


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "raw_files": {},          # key -> uploaded bytes / text
        "raw_dfs": {},            # key -> source DataFrame (pre-mapping)
        "mappings": {},           # key -> {standard_field: source_column}
        "cleaned": {},            # key -> cleaned DataFrame
        "build_doc_text": "",
        "benchmarks": {
            "cpa_benchmark": 25.0,
            "viewability_goal": 0.70,
            "out_of_geo_threshold": 0.05,
            "ivt_threshold": 0.02,
            "min_spend_threshold": 250.0,
            "pacing_goal": 1.0,
            "flight_start": date.today() - timedelta(days=14),
            "flight_end": date.today() + timedelta(days=14),
        },
        "case_studies": "",
        "results": None,          # last full run result
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------
def _read_tabular(uploaded) -> Optional[pd.DataFrame]:
    """Read a CSV or Excel file from a Streamlit uploaded file object."""
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    raw = uploaded.read()
    bio = io.BytesIO(raw)
    try:
        if name.endswith(".csv") or name.endswith(".txt"):
            return pd.read_csv(bio)
        if name.endswith(".xls") or name.endswith(".xlsx") or name.endswith(".xlsm"):
            return pd.read_excel(bio)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read {uploaded.name}: {exc}")
    return None


def _read_text_doc(uploaded) -> str:
    """Best-effort text extraction for the build doc (txt / md / csv only)."""
    if uploaded is None:
        return ""
    name = uploaded.name.lower()
    raw = uploaded.read()
    if name.endswith((".txt", ".md", ".csv")):
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
    # Binary docs (pdf/docx) — we just note that the file was uploaded.
    return f"[Build doc uploaded: {uploaded.name} — paste key constraints below if needed.]"


# ---------------------------------------------------------------------------
# Sidebar — benchmarks & flight
# ---------------------------------------------------------------------------
def _sidebar_benchmarks():
    st.sidebar.header("Benchmarks & thresholds")
    b = st.session_state.benchmarks

    b["cpa_benchmark"] = st.sidebar.number_input(
        "CPA benchmark ($)", min_value=0.0, value=float(b["cpa_benchmark"]), step=1.0
    )
    b["viewability_goal"] = st.sidebar.slider(
        "Viewability goal", 0.0, 1.0, float(b["viewability_goal"]), 0.01
    )
    b["out_of_geo_threshold"] = st.sidebar.slider(
        "Out-of-geo threshold", 0.0, 1.0, float(b["out_of_geo_threshold"]), 0.01
    )
    b["ivt_threshold"] = st.sidebar.slider(
        "IVT threshold", 0.0, 1.0, float(b["ivt_threshold"]), 0.01
    )
    b["min_spend_threshold"] = st.sidebar.number_input(
        "Min spend for exclusion ($)", min_value=0.0,
        value=float(b["min_spend_threshold"]), step=25.0,
    )
    b["pacing_goal"] = st.sidebar.slider(
        "Pacing goal", 0.5, 1.5, float(b["pacing_goal"]), 0.05
    )
    b["flight_start"] = st.sidebar.date_input("Flight start", value=b["flight_start"])
    b["flight_end"] = st.sidebar.date_input("Flight end", value=b["flight_end"])

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "DeepSeek model: " + (os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    )
    st.sidebar.caption(
        "API base: " + (os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    )
    if not os.environ.get("DEEPSEEK_API_KEY"):
        st.sidebar.warning("DEEPSEEK_API_KEY not set — AI summary will fall back.")


# ---------------------------------------------------------------------------
# Upload page
# ---------------------------------------------------------------------------
def _page_upload():
    st.subheader("1. Upload reports")
    st.caption(
        "Upload CSV or Excel files for each report you have. The build doc "
        "can be a text file; PDF/DOCX are accepted but you may need to paste "
        "key constraints below."
    )
    cols = st.columns(2)
    for idx, (key, label) in enumerate(REPORT_TYPES):
        with cols[idx % 2]:
            uploaded = st.file_uploader(
                label,
                type=(
                    ["csv", "xlsx", "xls", "xlsm"]
                    if key not in {"build_doc"}
                    else ["txt", "md", "csv", "pdf", "docx"]
                ),
                key=f"upload_{key}",
                accept_multiple_files=False,
            )
            if uploaded is not None:
                if key == "build_doc":
                    text = _read_text_doc(uploaded)
                    st.session_state.build_doc_text = text
                    st.success(f"Loaded build doc: {uploaded.name}")
                else:
                    df = _read_tabular(uploaded)
                    if df is not None:
                        st.session_state.raw_dfs[key] = df
                        st.success(f"Loaded {uploaded.name} — {len(df)} rows")

    st.markdown("---")
    st.text_area(
        "Build doc constraints (paste key items: frequency caps, geo, "
        "dayparting, brand safety, audience, KPI goals)",
        key="build_doc_text",
        height=160,
    )
    st.text_area(
        "Optional case study notes (treated as supporting evidence only)",
        key="case_studies",
        height=120,
    )


# ---------------------------------------------------------------------------
# Mapping page
# ---------------------------------------------------------------------------
def _page_mapping():
    st.subheader("2. Data mapping")
    st.caption(
        "Map your uploaded columns to the standard fields the app expects. "
        "Unmapped fields are skipped. You can leave a field blank."
    )

    if not st.session_state.raw_dfs:
        st.info("Upload reports first.")
        return

    for key, df in st.session_state.raw_dfs.items():
        std_fields = STANDARD_FIELDS.get(key, [])
        if not std_fields:
            continue
        with st.expander(f"{key} — {len(df)} rows, {df.shape[1]} columns", expanded=False):
            cleaned_cols = data_cleaner.standardize_column_names(df).columns.tolist()
            options = [""] + cleaned_cols
            current = st.session_state.mappings.get(key, {})
            new_map: Dict[str, str] = {}
            grid_cols = st.columns(3)
            for i, fld in enumerate(std_fields):
                # Auto-suggest if a normalized column matches the field name
                default = current.get(fld) or (fld if fld in cleaned_cols else "")
                with grid_cols[i % 3]:
                    sel = st.selectbox(
                        fld,
                        options,
                        index=options.index(default) if default in options else 0,
                        key=f"map_{key}_{fld}",
                    )
                    if sel:
                        new_map[fld] = sel
            st.session_state.mappings[key] = new_map
            st.caption(f"Mapped {len(new_map)} / {len(std_fields)} fields")


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
def _run_pipeline() -> Dict:
    cleaned: Dict[str, pd.DataFrame] = {}
    for key, df in st.session_state.raw_dfs.items():
        mapping = st.session_state.mappings.get(key, {})
        cleaned[key] = data_cleaner.clean_dataframe(df, mapping)
    st.session_state.cleaned = cleaned

    benchmarks = st.session_state.benchmarks
    campaign_df = cleaned.get("campaign", pd.DataFrame())
    site_df = cleaned.get("site", pd.DataFrame())
    zip_df = cleaned.get("zip", pd.DataFrame())
    pmp_df = cleaned.get("pmp", pd.DataFrame())
    approved_zip_df = cleaned.get("approved_zips", pd.DataFrame())
    exclusion_df = cleaned.get("exclusion_list", pd.DataFrame())

    # KPIs
    campaign_kpi_df = kpi_calculator.calculate_campaign_kpis(
        campaign_df, benchmarks["flight_start"], benchmarks["flight_end"],
        benchmarks["pacing_goal"],
    )
    site_kpi_df = kpi_calculator.calculate_site_kpis(site_df)
    zip_kpi_df = kpi_calculator.calculate_zip_kpis(zip_df)
    pmp_kpi_df = kpi_calculator.calculate_pmp_kpis(pmp_df)
    summary = kpi_calculator.build_kpi_summary(campaign_kpi_df, benchmarks)

    # Rules
    health_df = optimization_rules.classify_campaign_health(campaign_kpi_df, benchmarks)
    site_recs = optimization_rules.site_exclusion_candidates(site_kpi_df, benchmarks)
    zip_remove = optimization_rules.zip_removal_candidates(zip_kpi_df, benchmarks)
    zip_add = optimization_rules.zip_expansion_candidates(
        approved_zip_df, zip_kpi_df, health_df, benchmarks
    )
    zip_recs_combined = pd.concat([
        d for d in [zip_remove, zip_add] if d is not None and not d.empty
    ], ignore_index=True) if (zip_remove is not None or zip_add is not None) else pd.DataFrame()
    pmp_review_df = optimization_rules.pmp_review(pmp_kpi_df, benchmarks)
    pacing_df = optimization_rules.build_pacing_table(campaign_kpi_df)
    qa_df = data_cleaner.build_qa_dataframe(cleaned)

    # DeepSeek
    payload = deepseek_agent.build_summary_payload(
        campaign_summary=summary,
        health_df=health_df,
        site_recs=site_recs,
        zip_recs_remove=zip_remove,
        zip_recs_add=zip_add,
        pmp_review_df=pmp_review_df,
        build_doc_text=st.session_state.build_doc_text or "",
        exclusion_summary={
            "rows": int(len(exclusion_df)) if exclusion_df is not None else 0,
        },
        approved_zip_summary={
            "rows": int(len(approved_zip_df)) if approved_zip_df is not None else 0,
        },
        case_studies=st.session_state.case_studies or "",
    )
    final_recommendation = deepseek_agent.generate_recommendation(payload)

    # Pack results
    return {
        "summary": summary,
        "campaign_kpi_df": campaign_kpi_df,
        "site_kpi_df": site_kpi_df,
        "zip_kpi_df": zip_kpi_df,
        "pmp_kpi_df": pmp_kpi_df,
        "health_df": health_df,
        "site_recs": site_recs,
        "zip_remove": zip_remove,
        "zip_add": zip_add,
        "zip_recs_combined": zip_recs_combined,
        "pmp_review_df": pmp_review_df,
        "pacing_df": pacing_df,
        "qa_df": qa_df,
        "final_recommendation": final_recommendation,
    }


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------
def _fmt_pct(v):
    return "—" if v is None or pd.isna(v) else f"{float(v):.1%}"


def _fmt_money(v):
    return "—" if v is None or pd.isna(v) else f"${float(v):,.2f}"


def _fmt_int(v):
    return "—" if v is None or pd.isna(v) else f"{int(v):,}"


def _render_dashboard(results: Dict):
    summary = results["summary"]
    final = results["final_recommendation"]
    health_df = results["health_df"]

    st.subheader("Campaign health")
    if health_df is not None and not health_df.empty and "health_status" in health_df.columns:
        for _, row in health_df.head(10).iterrows():
            status = row.get("health_status", "Review")
            reason = row.get("health_reason", "")
            name = row.get("campaign_name") or row.get("campaign_id") or "Campaign"
            icon = {
                "Healthy": "✅",
                "Underpacing but efficient": "⚠️",
                "Underpacing and inefficient": "🚨",
                "Overspending and inefficient": "🚨",
                "Overspending but efficient": "⚠️",
                "Review": "ℹ️",
                "Unknown": "❔",
            }.get(status, "ℹ️")
            st.markdown(f"{icon} **{name}** — {status}  \n_{reason}_")
    else:
        st.info("Upload a campaign report and run the analysis to see health status.")

    st.subheader("KPI snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spend", _fmt_money(summary.get("total_spend")))
    c2.metric("Impressions", _fmt_int(summary.get("total_impressions")))
    c3.metric("Conversions", _fmt_int(summary.get("total_conversions")))
    c4.metric("Blended CPA", _fmt_money(summary.get("blended_cpa")))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Blended CPM", _fmt_money(summary.get("blended_cpm")))
    c6.metric("Blended CTR", _fmt_pct(summary.get("blended_ctr")))
    c7.metric("Avg viewability", _fmt_pct(summary.get("avg_viewability")))
    c8.metric("Avg pacing", _fmt_pct(summary.get("avg_pacing")))

    st.subheader("AI executive summary")
    risk = "low"
    score = float(final.get("confidence_score") or 0)
    if score < 0.4:
        risk = "high — review carefully"
    elif score < 0.7:
        risk = "medium — verify before action"
    st.markdown(f"**Status:** {final.get('health_status')}  \n"
                f"**Confidence:** {score:.0%} (risk level: {risk})")
    st.write(final.get("executive_summary") or "—")

    st.subheader("Top issues")
    issues = final.get("top_issues") or []
    if issues:
        for it in issues:
            st.markdown(f"- ⚠️ {it}")
    else:
        st.caption("None reported by the AI summary.")

    st.subheader("Top recommendations")
    recs = final.get("recommendations") or []
    if recs:
        rec_rows = []
        for r in recs:
            if isinstance(r, dict):
                rec_rows.append({
                    "area": r.get("area"),
                    "action": r.get("action"),
                    "rationale": r.get("rationale"),
                    "evidence": r.get("evidence"),
                })
            else:
                rec_rows.append({"area": "—", "action": str(r), "rationale": "", "evidence": ""})
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No AI recommendations available — see rule-based tables below.")

    st.subheader("Human next steps")
    for step in final.get("human_next_steps") or []:
        st.markdown(f"- 👤 {step}")


# ---------------------------------------------------------------------------
# Detail tabs
# ---------------------------------------------------------------------------
def _render_detail_tabs(results: Dict):
    tabs = st.tabs([
        "Pacing", "KPI tables", "Site recs", "Zip recs", "PMP review",
        "Do not change", "Data QA",
    ])

    with tabs[0]:
        df = results.get("pacing_df")
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No pacing data — upload a campaign report with budget + spend "
                    "and set the flight dates.")

    with tabs[1]:
        st.markdown("**Campaign KPI**")
        st.dataframe(results.get("campaign_kpi_df", pd.DataFrame()),
                     use_container_width=True, hide_index=True)
        st.markdown("**Site KPI**")
        st.dataframe(results.get("site_kpi_df", pd.DataFrame()),
                     use_container_width=True, hide_index=True)
        st.markdown("**Zip KPI**")
        st.dataframe(results.get("zip_kpi_df", pd.DataFrame()),
                     use_container_width=True, hide_index=True)

    with tabs[2]:
        df = results.get("site_recs")
        if df is None or df.empty:
            st.success("✅ No site exclusion candidates given current thresholds.")
        else:
            st.warning(f"⚠️ {len(df)} site(s) flagged for exclusion review")
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[3]:
        zr = results.get("zip_remove")
        za = results.get("zip_add")
        if (zr is None or zr.empty) and (za is None or za.empty):
            st.success("✅ No zip changes recommended right now.")
        if zr is not None and not zr.empty:
            st.warning(f"⚠️ {len(zr)} ZIP(s) recommended for removal/reduction")
            st.dataframe(zr, use_container_width=True, hide_index=True)
        if za is not None and not za.empty:
            st.info(f"➕ {len(za)} approved ZIP(s) suggested for expansion")
            st.dataframe(za, use_container_width=True, hide_index=True)

    with tabs[4]:
        df = results.get("pmp_review_df")
        if df is None or df.empty:
            st.info("No PMP report uploaded.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.dataframe(guardrails.get_do_not_change_df(),
                     use_container_width=True, hide_index=True)
        for item in (results["final_recommendation"].get("do_not_change") or []):
            st.markdown(f"- 🛑 {item}")

    with tabs[6]:
        df = results.get("qa_df")
        if df is None or df.empty:
            st.info("No QA summary yet.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("📊 Programmatic Optimization Recommendation Agent")
    st.caption(
        "Recommendations only — this app never makes DSP changes, never "
        "pauses campaigns, and never changes budgets. A human must review "
        "every recommendation before acting."
    )

    _sidebar_benchmarks()

    sections = st.tabs(["Upload reports", "Data mapping", "Analysis & dashboard"])

    with sections[0]:
        _page_upload()

    with sections[1]:
        _page_mapping()

    with sections[2]:
        st.subheader("3. Run analysis")
        if not st.session_state.raw_dfs:
            st.info("Upload at least one report on the first tab.")
        col_run, col_dl = st.columns([1, 3])
        with col_run:
            run = st.button("Run analysis", type="primary", use_container_width=True)
        if run:
            with st.spinner("Cleaning data, calculating KPIs, calling DeepSeek…"):
                st.session_state.results = _run_pipeline()

        results = st.session_state.results
        if not results:
            st.caption("Run the analysis to see the dashboard and download the report.")
            return

        st.markdown("---")
        _render_dashboard(results)

        st.markdown("---")
        st.subheader("Detail")
        _render_detail_tabs(results)

        # Excel download
        excel_bytes = output_generator.build_excel(
            campaign_summary_df=results.get("campaign_kpi_df"),
            pacing_df=results.get("pacing_df"),
            kpi_df=results.get("campaign_kpi_df"),
            site_recs_df=results.get("site_recs"),
            zip_recs_df=results.get("zip_recs_combined"),
            pmp_df=results.get("pmp_review_df"),
            final_recommendation=results.get("final_recommendation", {}),
            do_not_change_df=guardrails.get_do_not_change_df(),
            qa_df=results.get("qa_df"),
        )
        st.download_button(
            label="⬇️ Download recommendation report (Excel)",
            data=excel_bytes,
            file_name="programmatic_recommendation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
