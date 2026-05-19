# AI Car Concierge

Premium dealership chatbot: hybrid RAG (SQLite + Gemini embeddings), Gemini intent/replies, Resend emails, 2022+ policy enforcement.

## Assignment checklist

| Requirement | Status |
|-------------|--------|
| SQLite inventory + safe queries | Done |
| Policy RAG (Gemini embeddings + keyword fallback) | Done |
| Hybrid inventory + policy in one question | Done |
| Pre-2022 visible, not sold/reserved (`legacy_year_conflict`) | Done |
| Purchase email (Resend) | Done (`RESEND_API_KEY`) |
| Reserve → `stock_count--` | Done |
| FastAPI + Streamlit | Done |
| Tests + CI | `pytest -q` · GitHub Actions |
| Public URL | Deploy via `Dockerfile` / `render.yaml` |

## Business rules (deterministic)

- `year < 2022` → `pending_delisting`; block reserve/purchase (HTTP 409 on actions).
- Questions mentioning a pre-2022 year (e.g. “2020 Tesla”) → `legacy_year_conflict`, even with policy words (“refund”, “price”).
- Inventory and hybrid replies use **structured DB/RAG text only** (no LLM paraphrase on prices/stock).
- Policy and general chat may use Gemini to polish wording from retrieved policy context only.

## Environment

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Gemini (`AIza...`) |
| `RESEND_API_KEY` | Purchase emails |
| `SHOW_DEBUG_META` | Streamlit debug footer |

Config loads from project-root `.env` via `bootstrap()` (see `backend/config.py`).

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
streamlit run frontend/app.py
pytest -q
```

## HTTP (`POST /api/chat`)

| Status | When |
|--------|------|
| 200 | Normal reply; legacy-year informational (`blocked: true` in body) |
| 409 | Reserve/purchase action blocked; `reply` explains why |

## Deploy

- API: Render/Railway with `Dockerfile` + `render.yaml` (set secrets in dashboard).
- UI: Streamlit Cloud → `frontend/app.py`, `BACKEND_URL=https://your-api...`
- Docker: `.dockerignore` excludes `.env` and local DB files.

## CI

GitHub Actions: `.github/workflows/ci.yml` runs `pytest` without API keys (rules + keyword paths).

## AI transparency

Built with Cursor. Google Gemini (`google-genai`) for intent, embeddings, and policy/general phrasing. No Text-to-SQL.
