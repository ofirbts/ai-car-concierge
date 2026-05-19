# AI Car Concierge

Premium dealership chatbot: hybrid RAG (SQLite inventory + Gemini policy embeddings), structured Gemini intent extraction, Resend purchase automations, and strict 2022+ sales policy enforcement.

## Live links

| Service | URL |
|---------|-----|
| **API (Render)** | https://ai-car-concierge-a073.onrender.com |
| **API docs** | https://ai-car-concierge-a073.onrender.com/docs |
| **Streamlit UI** | https://ai-car-concierge.streamlit.app |
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
| Tests + CI | 103 tests · GitHub Actions · RAG golden eval |
| Public URL | API + Streamlit UI live |
| Observability + audit | Request ID, action_audit, purchase idempotency |

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
4. **Response layer (deterministic)** — Inventory, hybrid, policy, and general chat replies are **deterministic** (no LLM paraphrase on facts). Gemini is used for intent classification and policy embeddings only.

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
| `API_KEY` | Required in production (Render + Streamlit). Empty locally/CI for tests. |
| Rate limit | `CHAT_RATE_LIMIT` (default `30/minute`) on chat; `10/minute` on REST reserve. |
| Idempotency | Reserve + purchase emails deduplicated via SQLite + stable Streamlit keys. |
| CORS | Production: `https://ai-car-concierge.streamlit.app` (comma-separated list). |
| Observability | `X-Request-ID` on every response; structured access + `chat_outcome` logs. |
| Audit | `action_audit` table for reserve and purchase outcomes. |
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

## Local testing

```bash
pytest -q
```

CI runs without `.env`. With a local `.env` that sets `API_KEY`, tests still pass because `conftest.py` clears `API_KEY` for the suite. For production smoke against Render:

```bash
pytest tests/test_production_smoke.py -q
```

(requires `API_KEY` in `.env` or environment)

## Deploy architecture (who does what)

| Platform | Responsibility | URL |
|----------|----------------|-----|
| **Render Web Service (API)** | FastAPI, SQLite, Gemini, Resend, policy enforcement | https://ai-car-concierge-a073.onrender.com |
| **Render Web Service (UI)** or **Streamlit Cloud** | Chat UI only — calls API with `BACKEND_URL` + `API_KEY` | https://ai-car-concierge.streamlit.app |
| **GitHub** | Source code + CI (`pytest`) | https://github.com/ofirbts/ai-car-concierge |

**Blueprint** on Render = file `render.yaml` that defines services automatically when you sync the repo. You can also create each Web Service manually (same result).

### API (Render) — already live

1. Connect GitHub repo → New Web Service.
2. Use `render.yaml` or Docker: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
3. Set secrets: `GOOGLE_API_KEY`, `RESEND_*`, optional `API_KEY`.
4. Health check: `/ready`.

### UI (Render — recommended)

1. Dashboard → **Blueprints** → sync repo (or **New Web Service**).
2. Service `ai-car-concierge-ui` uses `Dockerfile.streamlit`.
3. Environment (required):
   - `BACKEND_URL` = `https://ai-car-concierge-a073.onrender.com`
   - `API_KEY` = same value as on the API service
4. Health check: `/` (Streamlit root).
5. After deploy, open the UI URL (e.g. `https://ai-car-concierge-ui.onrender.com`).

### UI (Streamlit Cloud — alternative)

1. [share.streamlit.io](https://share.streamlit.io) → New app → repo `ofirbts/ai-car-concierge`.
2. Main file: `frontend/app.py`, branch `main`.
3. Secrets:
   ```toml
   BACKEND_URL = "https://ai-car-concierge-a073.onrender.com"
   API_KEY = "your-api-key"
   ```

## Submission demo (copy into assessment)

Live app: https://ai-car-concierge.streamlit.app  
API: https://ai-car-concierge-a073.onrender.com/docs  
Repo: https://github.com/ofirbts/ai-car-concierge

Suggested prompts (proves policy + inventory + actions):

1. `Tesla under $70000` — inventory + pre-2022 note  
2. `Do you have a 2020 BMW?` — legacy conflict + 2022+ alternatives  
3. `Model 3 price and refund policy` — hybrid RAG (Tesla + policy chunks)  
4. `reserve vehicle #16` — stock decrement + idempotency  
5. `buy vehicle #48 with you@email.com` — Resend purchase email  

Verify API auth (optional, for reviewers with key):

```bash
curl -s https://ai-car-concierge-a073.onrender.com/ready
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://ai-car-concierge-a073.onrender.com/api/chat \
  -H "Content-Type: application/json" -d '{"message":"Tesla"}'
```

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

Built with Cursor. Google Gemini (`google-genai`) for **intent extraction and embeddings only** — not for inventory prices or policy text. No Text-to-SQL. Inventory numbers always come from SQLite.
