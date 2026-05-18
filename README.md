# AI Car Concierge

Premium dealership chatbot: **hybrid RAG** (SQLite + OpenAI embeddings or keyword fallback), **2022+ policy enforcement**, **Resend purchase emails**, **SQLite reservations**.

## Assignment checklist

| Requirement | Status |
|-------------|--------|
| Structured inventory (`inventory.sql` → SQLite) | Done |
| Policy knowledge base (5 markdown files) | Done |
| Hybrid RAG in chat (`hybrid_rag` intent, SQL + policy RAG) | Done |
| Policy RAG uses same config as LLM (`bootstrap()` + `get_settings()`) | Done |
| Conflict: pre-2022 visible, not sold/reserved | Done (`legacy_year_conflict`, HTTP 409 on chat reserve/purchase) |
| Live email on purchase intent | Done (`RESEND_API_KEY` required) |
| Live DB reserve (`stock_count--`) | Done |
| FastAPI + Streamlit | Done |
| Tests | `pytest` (60 tests) |
| **GitHub repo** | **You: `git commit` + push** |
| **Public URL** | **You: deploy via `Dockerfile` / `render.yaml`** |

## Architecture

```
Streamlit → POST /api/chat
              └─ Depends(get_policy_rag_service)   # embeddings when OPENAI_API_KEY set
              └─ orchestrator.handle_chat
                    ├─ intent.py (rules + gpt-4o-mini structured)
                    ├─ SQLite search (parameterized)
                    ├─ PolicyRAGService.search (embeddings | keyword)
                    └─ llm_service.synthesize_reply (optional natural answer)
```

## Configuration (single source)

All modules use `backend.config.get_settings()` after `bootstrap()` loads `.env` via `python-dotenv` on app startup.

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Intent LLM, reply synthesis, policy embeddings |
| `RESEND_API_KEY` | Purchase emails |
| `SHOW_DEBUG_META` | Streamlit intent/rag debug footer |
| `USE_QUALITY_LLM` | Use `gpt-4o` instead of `gpt-4o-mini` for replies |

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

uvicorn backend.main:app --reload
streamlit run frontend/app.py
pytest -q
```

## Deploy

```bash
# Render: connect repo, uses render.yaml + Dockerfile
# Set env vars in dashboard, then:
# Streamlit Cloud: point to frontend/app.py, set BACKEND_URL to deployed API
```

## Example prompts

- `Tesla under $70000`
- `Do you have a 2020 Tesla?`
- `Model 3 price and refund policy`
- `reserve vehicle #16`
- `buy vehicle #48 with you@email.com`

## AI transparency

Built with Cursor-assisted development. OpenAI is used for optional intent parsing, optional embeddings, and optional reply synthesis — never for SQL execution.
