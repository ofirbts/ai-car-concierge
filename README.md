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

- **SQLite on Render:** database file is `data/car_inventory.db`, mounted on a **1GB persistent disk** via `render.yaml` (`/app/data`). Survives redeploys on the same service; not a substitute for managed Postgres at scale.
- **API_KEY required in production:** Reviewers need the key (or use Streamlit UI only).
- **RAG golden tests** use keyword mode in CI; production may use `gemini_embeddings` when `GOOGLE_API_KEY` is set.

For production beyond demo: Postgres + migrations (see [docs/DECISIONS.md](docs/DECISIONS.md)).

## Security & production boundaries

| Control | Behavior |
|---------|----------|
| `API_KEY` | Required in production (Render + Streamlit). Empty locally/CI for unit tests. Shared secret for demo (not multi-tenant). |
| Rate limit | Per `X-API-Key` when present, else per client IP; `CHAT_RATE_LIMIT` (default `30/minute`) on chat; `10/minute` on REST reserve. |
| Idempotency | Reserve + purchase emails deduplicated via SQLite + stable Streamlit keys. |
| CORS | Production: `https://ai-car-concierge.streamlit.app` (comma-separated list). |
| Observability | `X-Request-ID` header + `request_id` in chat JSON; structured access + `chat_outcome` logs. |
| Audit | `action_audit` table for reserve and purchase outcomes. |
| Errors | No raw tracebacks to clients; structured JSON errors. |

## Environment

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Optional | Intent + embeddings (keyword/rules if unset) |
| `RESEND_API_KEY` | Optional | Purchase and inquiry emails |
| `RESEND_FROM_EMAIL` | If Resend | Sender address |
| `RESEND_TO_EMAIL` | If Resend | Inbox for notifications |
| `API_KEY` | Production | API auth (`X-API-Key`); empty locally/CI unit tests |
| `CHAT_RATE_LIMIT` | Optional | Chat rate limit (default `30/minute`) |
| `CORS_ORIGINS` | Production | Allowed browser origins |
| `BACKEND_URL` | Streamlit | API base URL for UI |
| `GEMINI_CHAT_MODEL` | Optional | Intent model (default `gemini-2.5-flash`) |
| `GEMINI_EMBEDDING_MODEL` | Optional | Embeddings (default `gemini-embedding-001`) |
| `SHOW_DEBUG_META` | Optional | Streamlit intent/rag debug footer |
| `USE_QUALITY_LLM` | Optional | Use quality chat model for intent |

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

API health check: `/ready`. Render blueprint mounts `data/car_inventory.db` on persistent disk (see `render.yaml`).

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

GitHub Actions (`.github/workflows/ci.yml`):

| Job | When | What |
|-----|------|------|
| `test` | Every push/PR | Full suite (no live keys) + production `/ready` + chat 401 |
| `smoke-prod` | Push to `master`/`main` | Full production smoke **with** `PRODUCTION_API_KEY` secret |

**Required repo secret (for `smoke-prod`):** `PRODUCTION_API_KEY` — same value as `API_KEY` on Render. Job **fails** if the secret is missing or wrong.

`pip-audit` fails the job on known vulnerable dependencies. Manual re-run: **Actions → CI → Run workflow**.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `/ready` 503 or timeout | Render cold start | Wait 60s, retry; open service URL once |
| Chat 401 from Streamlit | `API_KEY` mismatch | Same key in Render API + Streamlit secrets |
| Chat 429 | Rate limit | Wait or raise `CHAT_RATE_LIMIT` on Render |
| Gemini errors / keyword RAG only | Missing or invalid `GOOGLE_API_KEY` | Set key on Render, redeploy |
| Stock reset after redeploy | Disk not attached | Sync blueprint `render.yaml` disk on API service |

## AI transparency

Built with Cursor. `.cursorrules` in the repo documents local dev conventions for Cursor; not required to run the app.

Google Gemini (`google-genai`) for **intent extraction and embeddings only** — not for inventory prices or policy text shown to users. I removed LLM reply synthesis after review feedback: all customer-facing facts come from SQLite/RAG text. No Text-to-SQL.
