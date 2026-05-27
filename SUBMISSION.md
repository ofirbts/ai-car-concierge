# AI Car Concierge — Home Task Submission

**Candidate deliverable package** (matches assessment sections 1–5).

---

## 4. Public URL (live application)

| Surface | URL |
|---------|-----|
| **Chat UI (primary)** | https://ai-car-concierge.streamlit.app |
| **API + OpenAPI** | https://ai-car-concierge-a073.onrender.com/docs |
| **Health** | https://ai-car-concierge-a073.onrender.com/ready |

> Use the **Streamlit** link for the full demo (no API key in the browser).  
> Old URL `https://ai-car-concierge.onrender.com` is a stale deployment — use **`-a073`** only.

---

## 1. GitHub repository (source code)

https://github.com/ofirbts/ai-car-concierge  
Branch: `master`

---

## 2. Run locally

```bash
git clone https://github.com/ofirbts/ai-car-concierge.git
cd ai-car-concierge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: GOOGLE_API_KEY, RESEND_API_KEY, optional API_KEY

uvicorn backend.main:app --reload
# Second terminal:
streamlit run frontend/app.py

pytest -q
```

Database: `data/inventory.sql` → `data/car_inventory.db` on first startup (`init_db`).

---

## 3. README & AI tools (24 hours)

See [README.md](README.md) sections:

- **Assignment checklist** — maps to functional requirements
- **AI tools & 24-hour build** — Cursor + Gemini + pytest/CI
- **Architecture** — hybrid RAG, no Text-to-SQL, deterministic replies
- **Run locally** / **Deploy** / **Submission demo** prompts

---

## Functional requirements (how to verify in 2 minutes)

| Requirement | Demo prompt (Streamlit) | Expected |
|-------------|----------------------|----------|
| Hybrid RAG | `Model 3 price and refund policy` | Tesla inventory + policy excerpts |
| 2020/2021 conflict | `Do you have a 2020 BMW?` | In stock context + cannot sell (2022+ policy) |
| Real email | `buy vehicle #48 with you@email.com` | Purchase email via Resend (if configured) |
| DB reserve | `reserve vehicle #16` | `stock_count` decreases; idempotent replay |

---

## Reviewer access

- **UI:** https://ai-car-concierge.streamlit.app (secrets preconfigured on Streamlit Cloud).
- **API:** `X-API-Key` header — value provided via assessment portal / email (same as Render `API_KEY`).
- **CI:** green on `master`; `smoke-prod` job hits live API with GitHub secret `PRODUCTION_API_KEY`.

---

## Tech stack (assessment §4)

- **Backend:** FastAPI (Python 3.11)
- **Frontend:** Streamlit
- **AI:** Google Gemini (intent + embeddings)
- **DB:** SQLite (`data/car_inventory.db`, persistent disk on Render)
- **Email:** Resend
- **Deploy:** Render (API) + Streamlit Cloud (UI)

---

## Design choices (brief)

- **No Text-to-SQL** — parameterized Python queries only.
- **Gemini for conversational phrasing only** — inventory search, reserves, and stock counts come from SQLite; vehicle facts are injected into the NLG context. If Gemini returns a dollar amount not in that context, the reply falls back to deterministic templates and `output_validation` rejects ungrounded prices.
- **Policy hierarchy** — `SALES_MIN_YEAR = 2022` in code, aligned with `data/policies/policy.md`.
- **Session** — `session_id` on `POST /api/chat` powers discovery slots, shortlist, and conversational sales (`features.conversational_sales` on `/`).

Details: [docs/DECISIONS.md](docs/DECISIONS.md)

### What is sent to Gemini

- Policy RAG: top policy chunks (not raw SQL).
- Sales NLG: slot values + explicit vehicle fact lines (`#id`, make/model, price, stock, fuel).
- Intent classification (when enabled): user message only.

Prices and stock are never invented by the model in the happy path; they are copied from facts or template fallbacks.

### Troubleshooting (live demo)

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `401` on `/api/chat` | Missing/wrong `X-API-Key` | Set Streamlit secret `API_KEY` = Render env `API_KEY` |
| `503` on `/ready` | Render cold start or DB init | Wait 30–60s; retry `/ready` |
| UI old behavior | Streamlit cache | Streamlit Cloud → **Reboot** + hard refresh |
| API old version | Deploy still rolling | Check `GET /` → `version` matches `backend/version.py` |

### Security (demo scope)

- Shared secret between Streamlit and Render (`API_KEY`) — demo only, not production IAM.
- Governor/runtime journals are local dev artifacts (gitignored); not required for the take-home demo.

### Out of scope for reviewer demo

`job_broker`, `codebase_packager`, and `ResilientIterationController` are engineering extras. Start with **Streamlit UI → `POST /api/chat` → hybrid RAG + policy + conversational sales**.
