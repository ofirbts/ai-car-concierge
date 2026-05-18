# AI Car Concierge

Premium dealership chatbot: **hybrid RAG** (SQLite + Google Gemini embeddings), **Gemini** intent/replies, **Resend** emails, **2022+ policy** enforcement.

## Stack

- FastAPI + Pydantic v2
- Streamlit UI
- SQLite inventory (parameterized queries, no Text-to-SQL)
- **Google Gemini** — chat, structured intent, policy embeddings
- Resend — purchase emails

## Environment

Single API key via any of these names (first match in `.env`):

- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`
- `OPENAI_API_KEY` (legacy name — use your `AIza...` Google key here)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` | Intent + replies |
| `GEMINI_CHAT_MODEL_QUALITY` | `gemini-2.5-pro` | When `USE_QUALITY_LLM=true` |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Policy RAG vectors |
| `RESEND_*` | — | Purchase email automation |

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Put AIza... key in GOOGLE_API_KEY or OPENAI_API_KEY

uvicorn backend.main:app --reload
streamlit run frontend/app.py
pytest -q
```

- `/ready` → `"rag_mode": "gemini_embeddings"` when key is set
- Chat: `buy vehicle #48 with you@email.com`

## Architecture

```
POST /api/chat → PolicyRAGService (Gemini embeddings | keyword fallback)
              → classify_intent (Gemini JSON | rules fallback)
              → SQLite / Resend
              → synthesize_reply (Gemini | template fallback)
```

## AI transparency

Built with Cursor. **Google Gemini** for LLM + embeddings; **no OpenAI SDK**. Rules-based fallbacks if API fails or quota exceeded.

## Deliverables still on you

- Git push
- Public deploy URL (see `Dockerfile`, `render.yaml`)
