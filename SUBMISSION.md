# Submission notes

- Built with Cursor + manual review; commits on `master`.
- Gemini: intent extraction (`ExtractedIntent`) and policy embeddings only.
- I do **not** use Gemini to generate inventory prices or policy sentences shown to users.
- Tests: `pytest -q` (107+ tests; `conftest.py` clears `API_KEY` for the unit suite).
- CI: `test` job without live keys; `smoke-prod` requires `PRODUCTION_API_KEY` secret (fails if missing).
- Live: API https://ai-car-concierge-a073.onrender.com | UI https://ai-car-concierge.streamlit.app
- SQLite: `data/car_inventory.db` on Render **persistent disk** (`render.yaml`); Postgres + migrations for real production scale.
- Commit history shows iterative fixes after self-review (deterministic replies, idempotency, docs, CI).

## Reviewer access

- **Streamlit:** https://ai-car-concierge.streamlit.app — no API key in the browser if Cloud secrets are configured.
- **API curl:** `X-API-Key` from the assessment portal / email.
- **OpenAPI:** https://ai-car-concierge-a073.onrender.com/docs

## What is sent to Gemini

| Data | Sent? | Purpose |
|------|-------|---------|
| User chat message | Yes | Structured intent (`ExtractedIntent` JSON schema) |
| Policy markdown chunks | Yes (as embed inputs) | Semantic policy retrieval |
| Raw SQLite inventory rows | **No** | Prices/stock formatted in Python only |
| Customer emails in replies | **No** to Gemini | Used only for Resend after validation |
| PII beyond the message | **No** | No full DB export |

Models (from env): `gemini-2.5-flash` (intent), `gemini-embedding-001` (policy embeddings). Optional `gemini-2.5-pro` when `USE_QUALITY_LLM=true`.

## Troubleshooting (ops)

- **`/ready` fails:** Render waking up — retry after ~60s.
- **401 on API:** `API_KEY` on Render must match Streamlit `API_KEY` secret.
- **429:** rate limit — per API key when `X-API-Key` is set.
- **Reservations “reset”:** ensure API service has persistent disk from `render.yaml`; otherwise redeploy on old ephemeral disk wiped state.
