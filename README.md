# Programmatic Optimization Recommendation Agent

A local-first Streamlit app that helps a programmatic advertising
analyst turn raw campaign reports into a clean, human-readable
recommendation report.

> **Important:** This app is a *recommendation engine only*. It does
> **not** connect to any DSP, **never** pauses campaigns, **never**
> changes budgets, and **never** removes brand-safety / verification
> rails. Every suggestion requires a human reviewer before it is acted
> on.

## What it does

1. **Upload reports** — campaign performance, ad group, site/app, ZIP,
   PMP deal, build doc, exclusion list, approved ZIP list, plus
   optional DV report and past optimization log.
2. **Map columns** — tell the app which of your columns map to its
   standard fields (CPA, viewability, etc.).
3. **Clean data** — standardize names, drop blanks/duplicates, fix
   currency/percent/ZIP formats, and produce a Data QA summary.
4. **Calculate KPIs** — pacing, budget remaining, daily required spend,
   blended CPA / CPM / CTR / CVR, viewability/geo/IVT gaps, top/bottom
   ZIPs, PMP win-rate and floor mismatches.
5. **Apply rule-based optimization** — campaign health classification,
   site exclusion candidates, ZIP add/remove suggestions, PMP review
   flags.
6. **Summarize with DeepSeek** — sends a *summary* (never raw rows)
   to the DeepSeek API and asks for a structured JSON recommendation.
7. **Show a dashboard** — health, KPI cards, top issues, top
   recommendations, human next steps.
8. **Download a final Excel report** with one tab per section
   (Campaign Summary, Pacing, KPIs, Site Recs, ZIP Recs, PMP Review,
   Final Recommendation, Do Not Change, Data QA).

## How to run

The app is already wired up as the workspace's main workflow. It is
served at `/` on port 5000 and reloads automatically when you change a
file.

To run it manually:

```bash
streamlit run app.py --server.port 5000
```

## Configuration

Set these environment variables (the workspace already creates them):

| Variable             | Default                        | Purpose                                  |
| -------------------- | ------------------------------ | ---------------------------------------- |
| `DEEPSEEK_API_KEY`   | _required_                     | Your DeepSeek API key                    |
| `DEEPSEEK_BASE_URL`  | `https://api.deepseek.com`     | DeepSeek API base URL                    |
| `DEEPSEEK_MODEL`     | `deepseek-v4-flash`            | Model used for the recommendation call   |

If you ever need to swap the model (DeepSeek's catalog changes from
time to time — e.g. `deepseek-chat`, `deepseek-reasoner`), just update
`DEEPSEEK_MODEL` in your environment / Replit secrets.

### Adding your DeepSeek API key

1. Open the **Secrets** pane in Replit.
2. Add a new secret with key `DEEPSEEK_API_KEY` and your key as the value.
3. Restart the workflow. The sidebar will stop showing the warning.

## Using the app

1. Open the **Upload reports** tab and upload each report you have.
   CSV and Excel are supported. The build doc accepts text files
   directly; for PDF/DOCX, paste the key constraints into the text
   box at the bottom of the upload tab.
2. Open the **Data mapping** tab and map your columns to the standard
   fields. The app will pre-fill any auto-detected matches.
3. In the sidebar, set the benchmarks (CPA goal, viewability goal,
   geo / IVT thresholds, minimum spend for exclusion, pacing goal,
   flight start/end).
4. Open the **Analysis & dashboard** tab and click **Run analysis**.
5. Review the dashboard and detail tabs.
6. Click **⬇️ Download recommendation report (Excel)** to get the
   final report you can hand off.

## Project layout

```text
app.py                          # Streamlit entry point + UI wiring
requirements.txt                # Python dependencies
README.md                       # This file
modules/
├── data_cleaner.py             # Column normalization + type coercion
├── kpi_calculator.py           # Pacing + KPI math
├── optimization_rules.py       # Rule-based recommendation engine
├── deepseek_agent.py           # DeepSeek API client + JSON parsing
├── output_generator.py         # Excel writer
└── guardrails.py               # "Do not change" rules
```

## Guardrails

The **Do Not Change** section in the dashboard and the matching tab in
the Excel export remind reviewers to:

- Not pause campaigns automatically.
- Not change budgets automatically.
- Not remove DV / viewability / brand-safety rails without approval.
- Not exclude sites or ZIPs that lack enough data.
- Not treat web case studies as stronger evidence than campaign data.
- Honor the build doc constraints (frequency, geo, dayparting, etc.).
- Always require a human reviewer for every recommendation.
