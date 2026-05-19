import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx
import streamlit as st

from backend.chat_idempotency import (
    stable_purchase_idempotency_key,
    stable_reserve_idempotency_key,
)


def _apply_streamlit_secrets() -> None:
    try:
        for key in ("BACKEND_URL", "API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


_apply_streamlit_secrets()
from backend.config import bootstrap, get_settings

bootstrap()
settings = get_settings()
BACKEND_URL = settings.backend_url.rstrip("/")
API_HEADERS: dict[str, str] = {}
if settings.api_key.strip():
    API_HEADERS["X-API-Key"] = settings.api_key.strip()


def _parse_chat_response(response: httpx.Response) -> tuple[str, dict, bool]:
    data: dict = {}
    if response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
    reply = data.get("reply") or data.get("error", "No response.")
    is_blocked = response.status_code == 409 or bool(data.get("blocked"))
    return reply, data, is_blocked


st.set_page_config(page_title="AI Car Concierge", page_icon="🚗", layout="centered")
st.title("AI Car Concierge")
st.caption("Hybrid RAG · SQLite inventory · Gemini · Resend automations")

with st.sidebar:
    st.subheader("Your details")
    user_email = st.text_input("Email (for purchase follow-up)", placeholder="you@example.com")
    st.markdown("**Try:**")
    st.code("Tesla under $70000")
    st.code("Do you have a 2020 BMW?")
    st.code("Model 3 price and refund policy")
    st.code("reserve vehicle #16")
    st.code("buy vehicle #48 with you@email.com")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "reserve_idempotency_keys" not in st.session_state:
    st.session_state.reserve_idempotency_keys = {}
if "purchase_idempotency_keys" not in st.session_state:
    st.session_state.purchase_idempotency_keys = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("blocked"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask about cars, policies, reservations…"):
    st.session_state.messages.append({"role": "user", "content": prompt, "blocked": False})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload: dict = {"message": prompt}
    reserve_key = stable_reserve_idempotency_key(
        prompt,
        st.session_state.reserve_idempotency_keys,
    )
    if reserve_key:
        payload["idempotency_key"] = reserve_key
    else:
        purchase_key = stable_purchase_idempotency_key(
            prompt,
            user_email.strip() if user_email else None,
            st.session_state.purchase_idempotency_keys,
        )
        if purchase_key:
            payload["idempotency_key"] = purchase_key
    if user_email and user_email.strip():
        payload["user_email"] = user_email.strip()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/api/chat",
                    json=payload,
                    headers=API_HEADERS,
                    timeout=90.0,
                )
                if response.status_code in (200, 409):
                    reply, data, is_blocked = _parse_chat_response(response)
                    if settings.show_debug_meta:
                        meta = []
                        if data.get("intent"):
                            meta.append(f"intent: `{data['intent']}`")
                        if data.get("rag_mode"):
                            meta.append(f"rag: `{data['rag_mode']}`")
                        if data.get("email_sent"):
                            meta.append("✉️ sales email sent")
                        if data.get("reserved_vehicle"):
                            v = data["reserved_vehicle"]
                            meta.append(f"reserved #{v['id']}")
                        if is_blocked:
                            meta.append("⛔ blocked")
                        if meta:
                            reply += "\n\n---\n" + " · ".join(meta)
                    if is_blocked:
                        st.warning(reply)
                    else:
                        st.markdown(reply)
                elif response.status_code == 429:
                    reply = "Too many requests. Please wait a moment and try again."
                    is_blocked = True
                    st.warning(reply)
                elif response.is_success:
                    reply, _, is_blocked = _parse_chat_response(response)
                    st.markdown(reply)
                else:
                    reply, _, is_blocked = _parse_chat_response(response)
                    st.error(reply)
            except httpx.ConnectError:
                reply = (
                    f"Cannot reach backend at {BACKEND_URL}.\n\n"
                    "Start API: `uvicorn backend.main:app --reload`"
                )
                is_blocked = False
                st.error(reply)
            except Exception:
                reply = "Unexpected error talking to the backend."
                is_blocked = False
                st.error(reply)
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "blocked": is_blocked}
    )
