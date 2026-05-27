# Decisions

| Decision | Why | Rejected alternative |
|----------|-----|-------------------|
| Grounded sales NLG | Gemini phrases replies; prices/stock only from vehicle facts in context, `reply_prices_grounded`, and template fallback | Free-form paraphrase on inventory |
| Deterministic inventory/policy text | Search listings and policy answers stay template- or chunk-based | LLM inventing stock or policy |
| Keyword + Gemini intent | CI works without API key; predictable tests | LLM-only routing |
| SQLite | 24h MVP, single file deploy | Postgres (next step) |
| No Text-to-SQL | Injection + wrong joins risk | LLM-generated SQL |
| `Model 3` → Tesla | Inventory only contains Tesla Model 3 / Model Y | Full NER (out of scope) |
| No LangGraph / AutoGen | Auditable linear orchestrator | Multi-agent frameworks |
| Policy hierarchy | `year < 2022` enforced in code (`SALES_MIN_YEAR`), aligned with policy.md text | Runtime RAG override of SQL rows |
| SQLite + disk on Render | MVP persistence for demo; `data/car_inventory.db` on mounted volume | Ephemeral disk only |
| Next scale-out | Postgres + stateless API replicas + external vector store (or managed embeddings) | Single-process SQLite + in-memory embeddings |
