# Decisions

| Decision | Why | Rejected alternative |
|----------|-----|-------------------|
| Deterministic replies | Prevent price/stock hallucination | Gemini paraphrase on inventory |
| Keyword + Gemini intent | CI works without API key; predictable tests | LLM-only routing |
| SQLite | 24h MVP, single file deploy | Postgres (next step) |
| No Text-to-SQL | Injection + wrong joins risk | LLM-generated SQL |
| `Model 3` → Tesla | Inventory only contains Tesla Model 3 / Model Y | Full NER (out of scope) |
| No LangGraph / AutoGen | Auditable linear orchestrator | Multi-agent frameworks |
| Policy hierarchy | `year < 2022` enforced in code (`SALES_MIN_YEAR`), aligned with policy.md text | Runtime RAG override of SQL rows |
