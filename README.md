# AI Car Concierge

> **60-second demo:** Open [Streamlit](https://ai-car-concierge.streamlit.app) → try `Do you have a 2020 BMW?` then `reserve vehicle #16`. API: [docs](https://ai-car-concierge-a073.onrender.com/docs).

[![CI](https://github.com/ofirbts/ai-car-concierge/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ofirbts/ai-car-concierge/actions/workflows/ci.yml)

Premium dealership chatbot: hybrid RAG (SQLite inventory + Gemini policy embeddings), structured Gemini intent extraction, Resend purchase automations, and strict 2022+ sales policy enforcement.

## Live links

| Service | URL |
|---------|-----|
| **Streamlit UI** | https://ai-car-concierge.streamlit.app |
| **API (Render)** | https://ai-car-concierge-a073.onrender.com |
| **API docs** | https://ai-car-concierge-a073.onrender.com/docs |
| **GitHub** | https://github.com/ofirbts/ai-car-concierge |

## Assignment checklist

| Requirement | Status |
|-------------|--------|
| SQLite inventory + safe parameterized queries | Done |
| Policy RAG (Gemini embeddings + keyword fallback) | Done |
| Hybrid inventory + policy in one question | Done |
| Pre-2022 visible, not sold/reserved (`legacy_year_conflict`) | Done |
| Purchase email (Resend), including inquiry without vehicle | Done |
| Reserve → `stock_count--` with idempotency | Done |
| FastAPI + Streamlit | Done |
| Tests + CI | pytest suite · GitHub Actions · RAG golden eval |
| Public URL | API + Streamlit UI live |
| Observability + audit | Request ID, action_audit, purchase idempotency |

## Architecture

```
Streamlit UI
    │ POST /api/chat (+ X-API-Key, rate limit)
    ▼
FastAPI orchestrator
    ├─ intent.py          Gemini structured intent + keyword fallback
    ├─ intent_validate.py normalize make/model against DB
    ├─ database.py        parameterized SQLite (inventory, idempotency, audit)
    ├─ rag_service.py     PolicyRAGService (Gemini embeddings | keyword fallback)
    ├─ middleware.py      X-Request-ID, access logs
    └─ automations.py     Resend (purchase + inquiry, HTML-escaped)
```

**LLM usage (explicit):**

| Component | Gemini? | Source of truth |
|-----------|---------|-----------------|
| Intent classification | Yes (or rules if no key) | Pydantic `ExtractedIntent` |
| Policy retrieval | Embeddings (or keyword) | `data/policies/*.md` chunks |
| Inventory prices/stock | **No** | SQLite only |
| User-visible reply text | **No** | Formatted DB + RAG excerpts |
| Reserve / email actions | **No** | Validated code paths only |

### Agent design (no LangGraph)

1. **Intent (structured)** — Gemini extracts `ExtractedIntent` (Pydantic). Keyword rules handle reserve, purchase, legacy-year conflict.
2. **Retrieval (tools)** — SQLite for inventory; `PolicyRAGService` for policy chunks. No Text-to-SQL.
3. **Actions (validated)** — Reserve and email only after `assert_sellable` and email validation.
4. **Responses (deterministic)** — Inventory, hybrid, policy, and general chat use templates and DB/RAG text only.

This keeps actions safe and auditable. Gemini is used only where structured extraction and semantic policy search add value—not for inventing prices or policy text.

See [docs/DECISIONS.md](docs/DECISIONS.md) for tradeoffs. Build timeline: [docs/BUILD_LOG.md](docs/BUILD_LOG.md).

## Business rules (hierarchy of truth)

- `policy.md` (2022+ sales only) overrides DB display semantics.
- `year < 2022` → `pending_delisting`; block reserve/purchase (HTTP **409** on actions).
- Questions mentioning a pre-2022 year (e.g. “2020 BMW”) → `legacy_year_conflict`.
- Inventory and hybrid replies use **structured DB/RAG text only** (no LLM paraphrase on prices/stock).
- Policy replies are deterministic from retrieved chunks (no synthesis).

## Known limitations (MVP honesty)

- **SQLite on Render:** `car_inventory.db` lives on the instance disk. Redeploy or cold start can reset inventory/reservations unless you attach persistent disk or migrate to Postgres.
- **API_KEY required in production:** Reviewers need the key (or use Streamlit UI only).
- **RAG golden tests** use keyword mode in CI; production may use `gemini_embeddings` when `GOOGLE_API_KEY` is set.

For production beyond demo: attach Render persistent disk or use managed Postgres; treat current SQLite as **demo state only**.

## Security & production boundaries

| Control | Behavior |
|---------|----------|
| `API_KEY` | Required in production (Render + Streamlit). Empty locally/CI for tests. |
| Rate limit | `CHAT_RATE_LIMIT` (default `30/minute`) on chat; `10/minute` on REST reserve. |
| Idempotency | Reserve + purchase emails deduplicated via SQLite + stable Streamlit keys. |
| CORS | Production: `https://ai-car-concierge.streamlit.app` (comma-separated list). |
| Observability | `X-Request-ID` header + `request_id` in chat JSON; structured access + `chat_outcome` logs. |
| Audit | `action_audit` table for reserve and purchase outcomes. |
| Errors | No raw tracebacks to clients; structured JSON errors. |

## Environment

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Intent + embeddings (not reply synthesis) |
| `RESEND_API_KEY` | Purchase and inquiry emails |
| `API_KEY` | Optional API authentication |
| `CHAT_RATE_LIMIT` | Chat endpoint rate limit |
| `CORS_ORIGINS` | Allowed browser origins |
| `BACKEND_URL` | Streamlit → API base URL |
| `SHOW_DEBUG_META` | Streamlit intent/rag debug footer |

Config loads from project-root `.env` via `bootstrap()` in `create_app()` / Streamlit startup (`backend/config.py`).

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
streamlit run frontend/app.py
pytest -q
```

## HTTP semantics

### `POST /api/chat`

| Status | When |
|--------|------|
| 200 | Normal reply; legacy-year informational (`blocked: true` in body) |
| 409 | Reserve/purchase action blocked |
| 401 | Missing/invalid `X-API-Key` when configured |
| 429 | Rate limit exceeded |

**ChatResponse fields (body):** `reply`, `intent`, `vehicles`, `blocked`, `reserved_vehicle`, `email_sent`, `policy_context_used`, `rag_mode`, `request_id`.

### `POST /vehicles/{id}/reserve`

Supports `Idempotency-Key` header. Same key + same vehicle replays without double-decrementing stock.

## Local testing

```bash
pytest -q
```

CI runs without `.env`. With a local `.env` that sets `API_KEY`, tests still pass because `conftest.py` clears `API_KEY` for the suite.

Production smoke (optional):

```bash
pytest tests/test_production_smoke.py -q
```

(requires `API_KEY` in environment for authenticated checks)

## Deploy

| Platform | Responsibility |
|----------|----------------|
| **Render Web Service (API)** | FastAPI, SQLite, Gemini, Resend |
| **Streamlit Cloud** | Chat UI → API with `BACKEND_URL` + `API_KEY` |
| **GitHub** | Source + CI |

API health check: `/ready`. For production beyond demo, use persistent disk or Postgres (see Known limitations).

### Streamlit Cloud

1. [share.streamlit.io](https://share.streamlit.io) → repo `ofirbts/ai-car-concierge`, `frontend/app.py`.
2. Secrets: `BACKEND_URL` = `https://ai-car-concierge-a073.onrender.com`, `API_KEY` = same as API.

## Submission demo

Live app: https://ai-car-concierge.streamlit.app  
API: https://ai-car-concierge-a073.onrender.com/docs  
Repo: https://github.com/ofirbts/ai-car-concierge  
Recruiter notes: [SUBMISSION.md](SUBMISSION.md)

Suggested prompts:

1. `Tesla under $70000`
2. `Do you have a 2020 BMW?`
3. `Model 3 price and refund policy`
4. `reserve vehicle #16`
5. `buy vehicle #48 with you@email.com`

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` without live API keys — keyword RAG fallback and rule paths stay green. `pip-audit` fails the job on known vulnerable dependencies.

## AI transparency

Built with Cursor. `.cursorrules` in the repo documents local dev conventions for Cursor; not required to run the app.

Google Gemini (`google-genai`) for **intent extraction and embeddings only** — not for inventory prices or policy text shown to users. I removed LLM reply synthesis after review feedback: all customer-facing facts come from SQLite/RAG text. No Text-to-SQL.
