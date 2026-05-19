import httpx
import streamlit as st

from backend.config import bootstrap, get_settings

bootstrap()
settings = get_settings()
BACKEND_URL = settings.backend_url.rstrip("/")


def _parse_chat_response(response: httpx.Response) -> tuple[str, dict]:
    data: dict = {}
    if response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
    reply = data.get("reply") or data.get("error", "No response.")
    return reply, data


st.set_page_config(page_title="AI Car Concierge", page_icon="🚗", layout="centered")
st.title("AI Car Concierge")
st.caption("Hybrid RAG: SQLite inventory + policy documents")

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about cars, policies, reservations…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload: dict = {"message": prompt}
    if user_email and user_email.strip():
        payload["user_email"] = user_email.strip()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/api/chat",
                    json=payload,
                    timeout=90.0,
                )
                if response.status_code in (200, 409):
                    reply, data = _parse_chat_response(response)
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
                        if data.get("blocked"):
                            meta.append("⛔ policy block")
                        if meta:
                            reply += "\n\n---\n" + " · ".join(meta)
                elif response.is_success:
                    reply, _ = _parse_chat_response(response)
                else:
                    reply, _ = _parse_chat_response(response)
            except httpx.ConnectError:
                reply = (
                    f"Cannot reach backend at {BACKEND_URL}.\n\n"
                    "Start API: `uvicorn backend.main:app --reload`"
                )
            except Exception:
                reply = "Unexpected error talking to the backend."
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
