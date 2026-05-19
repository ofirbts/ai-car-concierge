# AI Car Concierge

Premium dealership chatbot: hybrid RAG (SQLite inventory + Gemini policy embeddings), structured Gemini intent extraction, Resend purchase automations, and strict 2022+ sales policy enforcement.

## Live links

| Service | URL |
|---------|-----|
| **API (Render)** | https://ai-car-concierge-a073.onrender.com |
| **API docs** | https://ai-car-concierge-a073.onrender.com/docs |
| **Streamlit UI** | _Set after Streamlit Cloud deploy_ |
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
| Tests + CI | `pytest -q` · GitHub Actions |
| Public URL | https://ai-car-concierge-a073.onrender.com |

## Architecture

```
Streamlit UI
    │ POST /api/chat (+ optional X-API-Key, rate limit)
    ▼
FastAPI orchestrator
    ├─ intent.py          keyword rules + Gemini structured extraction
    ├─ intent_validate.py normalize make/model against DB
    ├─ database.py        parameterized SQLite search / reserve
    ├─ rag_service.py     PolicyRAGService (Gemini embeddings or keyword)
    ├─ llm_service.py     synthesis for general chat only
    └─ automations.py     Resend purchase + inquiry emails
```

### Agent design (no LangGraph)

This project uses a **deterministic orchestrator** rather than a multi-agent framework:

1. **Intent agent (structured)** — Gemini extracts `ExtractedIntent` (Pydantic). Keyword rules handle high-confidence paths (reserve, purchase, legacy-year conflict).
2. **Retrieval agent (tools)** — SQLite for inventory; `PolicyRAGService` for policy chunks. No Text-to-SQL.
3. **Action agent (validated)** — Reserve and email only run after structured validation and policy checks (`assert_sellable`, email present).
4. **Response agent (selective LLM)** — Inventory, hybrid, and policy replies are **deterministic** from DB/RAG text. General chat may use Gemini to polish from provided context only.

This keeps actions safe and auditable while still using LLMs where they add value (intent + embeddings + general phrasing).

## Business rules (hierarchy of truth)

- `policy.md` (2022+ sales only) overrides DB display semantics.
- `year < 2022` → `pending_delisting`; block reserve/purchase (HTTP **409** on actions).
- Questions mentioning a pre-2022 year (e.g. “2020 BMW”) → `legacy_year_conflict`, even with policy words (“refund”, “price”).
- Inventory and hybrid replies use **structured DB/RAG text only** (no LLM paraphrase on prices/stock).
- Policy replies are also deterministic from retrieved chunks (no synthesis).

## Security & production boundaries

| Control | Behavior |
|---------|----------|
| `API_KEY` | Optional. When set, mutating/list endpoints require `X-API-Key`. |
| Rate limit | `CHAT_RATE_LIMIT` (default `30/minute`) on `POST /api/chat` via slowapi. |
| Idempotency | `Idempotency-Key` header on REST reserve; chat sends a stable per-session key only for reserve messages (same vehicle → replay without double-decrement). Conflicts return 409. |
| CORS | `CORS_ORIGINS` comma-separated list (default `*`). |
| Errors | No raw tracebacks to clients; structured JSON errors. |

## Environment

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Gemini chat, intent, embeddings |
| `RESEND_API_KEY` | Purchase and inquiry emails |
| `API_KEY` | Optional API authentication |
| `CHAT_RATE_LIMIT` | Chat endpoint rate limit |
| `CORS_ORIGINS` | Allowed browser origins |
| `BACKEND_URL` | Streamlit → API base URL |
| `SHOW_DEBUG_META` | Streamlit intent/rag debug footer |

Config loads from project-root `.env` via `bootstrap()` (`backend/config.py`).

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

### `POST /vehicles/{id}/reserve`

Supports `Idempotency-Key` header. Same key + same vehicle replays without double-decrementing stock.

## Deploy

### API (Render)

1. Connect GitHub repo → New Web Service.
2. Use `render.yaml` or Docker: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
3. Set secrets: `GOOGLE_API_KEY`, `RESEND_*`, optional `API_KEY`.
4. Health check: `/ready`.

### UI (Streamlit Cloud)

1. App path: `frontend/app.py`.
2. Secrets: `BACKEND_URL=https://your-api.onrender.com`, `API_KEY` if enabled.

## 24-hour build narrative

**Hours 0–4 — Foundation:** SQLite schema, policy markdown corpus, FastAPI skeleton, parameterized search, 2022+ enforcement tests.

**Hours 4–10 — RAG + intent:** Gemini embeddings with keyword fallback, hybrid routing, structured intent extraction, legacy-year conflict path.

**Hours 10–16 — Product surface:** Streamlit chat, orchestrator wiring, Resend purchase emails, reserve automation, deterministic inventory replies (anti-hallucination).

**Hours 16–20 — Production polish:** Idempotency, optional API key, rate limiting, CORS, `/ready` probe, Docker + Render config, CI.

**Hours 20–24 — Quality:** Expanded test suite, README, deploy checklist, end-to-end manual scenarios (Tesla search, 2020 BMW conflict, reserve #16, purchase email).

**AI tools used:** Cursor for implementation velocity; Gemini for intent/embedding/synthesis; pytest for regression safety. Deliberately avoided agent frameworks to keep behavior deterministic and reviewable.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` without live API keys — keyword RAG fallback and rule paths stay green in CI.

## AI transparency

Built with Cursor. Google Gemini (`google-genai`) for intent, embeddings, and general-chat phrasing. No Text-to-SQL. Inventory numbers always come from SQLite.
