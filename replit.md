# Workspace

## Overview

This workspace contains the **Programmatic Optimization Recommendation
Agent** — a local-first Streamlit app for programmatic ad analysts.
It analyzes uploaded campaign reports, applies rule-based optimization
logic, and uses the DeepSeek API to produce a human-readable
recommendation report. The app **does not connect to any DSP** and
**only generates recommendations** — it never pauses campaigns, changes
budgets, or alters brand-safety / verification settings.

The workspace also retains the original pnpm monorepo scaffolding
(api-server + mockup-sandbox) for any future TypeScript work.

## Stack

- **Primary app**: Python 3.11 + Streamlit
- **Data**: Pandas, OpenPyXL, XlsxWriter, NumPy
- **AI**: DeepSeek API via the OpenAI-compatible Python client
- **Monorepo tool**: pnpm workspaces (TypeScript scaffolding only)

## Key files

- `app.py` — Streamlit entry point, UI, and pipeline wiring
- `modules/data_cleaner.py` — column normalization + type coercion
- `modules/kpi_calculator.py` — pacing + KPI math
- `modules/optimization_rules.py` — rule-based recommendation engine
- `modules/deepseek_agent.py` — DeepSeek API client + JSON parsing
- `modules/output_generator.py` — Excel report writer
- `modules/guardrails.py` — "do not change" guardrails
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit server config (port 5000)

## Environment variables

| Variable             | Default                        | Purpose                       |
| -------------------- | ------------------------------ | ----------------------------- |
| `DEEPSEEK_API_KEY`   | _required (Replit secret)_     | DeepSeek API key              |
| `DEEPSEEK_BASE_URL`  | `https://api.deepseek.com`     | DeepSeek API base URL         |
| `DEEPSEEK_MODEL`     | `deepseek-v4-flash`            | Model used for the AI summary |

## Workflow

- `Streamlit App` runs `streamlit run app.py --server.port 5000` and
  serves the app on port 5000.
