# Submission notes

- Built with Cursor + manual review; commits on `master`.
- Gemini: intent extraction (`ExtractedIntent`) and policy embeddings only.
- I do **not** use Gemini to generate inventory prices or policy sentences shown to users.
- Tests: `pytest -q` (106+ tests; `conftest.py` clears `API_KEY` for the suite).
- CI: GitHub Actions without live API keys; keyword RAG + rule paths; production `/ready` + unauthenticated chat 401 smoke.
- Live: API https://ai-car-concierge-a073.onrender.com | UI https://ai-car-concierge.streamlit.app
- SQLite on Render is demo-only; reservations may reset on redeploy. Production would use Postgres + migrations.
- Commit history shows iterative fixes after self-review (deterministic replies, idempotency, docs).

## Reviewer access

- **Streamlit:** https://ai-car-concierge.streamlit.app — no API key needed in the browser if Cloud secrets are configured.
- **API curl:** use `API_KEY` from the assessment portal / email as header `X-API-Key`.
- **OpenAPI:** https://ai-car-concierge-a073.onrender.com/docs
