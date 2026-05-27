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

SLOT_LABELS = {
    "budget": "Budget",
    "passengers": "Passengers",
    "family_size": "Family size",
    "use_case": "Use case",
    "body_type": "Body type",
    "fuel_preference": "Fuel",
    "space_priority": "Space vs fuel",
    "timeline": "Timeline",
    "contact_email": "Email",
}

DEMO_FLOWS = [
    {
        "title": "Family purchase (EN)",
        "turns": [
            "I'm looking for a family car",
            "four people, budget 75000",
            "need space for family trips",
            "what's the best value?",
            "reserve vehicle #55",
        ],
    },
    {
        "title": "משפחה + תקציב (HE)",
        "turns": [
            "אני מחפש רכב למשפחה",
            "ארבעה אנשים, תקציב 75000",
            "חשוב מרווח למשפחה",
            "מה הכי משתלם?",
        ],
    },
]


def _parse_chat_response(response: httpx.Response) -> tuple[str, dict, bool]:
    data: dict = {}
    if response.headers.get("content-type", "").startswith("application/json"):
        data = response.json()
    reply = data.get("reply") or data.get("error", "No response.")
    is_blocked = response.status_code == 409 or bool(data.get("blocked"))
    return reply, data, is_blocked


def _render_vehicle_cards(vehicles: list[dict]) -> None:
    if not vehicles:
        return
    st.markdown("**Recommended vehicles**")
    for vehicle in vehicles[:4]:
        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(
                    f"**#{vehicle['id']}** · {vehicle['year']} {vehicle['make']} {vehicle['model']}"
                )
                st.caption(
                    f"{vehicle.get('color', '')} · {vehicle.get('fuel_type', '')} · "
                    f"stock {vehicle.get('stock_count', 0)}"
                )
            with cols[1]:
                st.markdown(f"**${vehicle['price']:,.0f}**")
            if vehicle.get("pending_delisting"):
                st.warning("Pending De-listing (pre-2022)")


def _render_progress(progress: dict) -> None:
    tracked = [
        "passengers",
        "budget",
        "use_case",
        "body_type",
        "space_priority",
    ]
    filled = 0
    for key in tracked:
        val = progress.get(key)
        if key == "passengers" and val is None:
            val = progress.get("family_size")
        if val is not None and val != "":
            filled += 1
    st.progress(filled / len(tracked), text=f"Discovery {filled}/{len(tracked)}")
    for key in tracked:
        val = progress.get(key)
        if key == "passengers" and val is None:
            val = progress.get("family_size")
        icon = "✅" if val is not None and val != "" else "○"
        label = SLOT_LABELS.get(key, key)
        display = val if val is not None else "—"
        st.caption(f"{icon} {label}: {display}")


def _send_chat(payload: dict) -> tuple[str, dict, bool]:
    response = httpx.post(
        f"{BACKEND_URL}/api/chat",
        json=payload,
        headers=API_HEADERS,
        timeout=90.0,
    )
    if response.status_code in (200, 409):
        return _parse_chat_response(response)
    if response.status_code == 429:
        return "Too many requests. Please wait a moment and try again.", {}, True
    if response.is_success:
        return _parse_chat_response(response)
    return _parse_chat_response(response)


st.set_page_config(page_title="AI Car Concierge", page_icon="🚗", layout="wide")
st.title("AI Car Concierge")
st.caption("Conversational sales agent · Hybrid RAG · SQLite · Gemini · Resend")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "reserve_idempotency_keys" not in st.session_state:
    st.session_state.reserve_idempotency_keys = {}
if "purchase_idempotency_keys" not in st.session_state:
    st.session_state.purchase_idempotency_keys = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "conversation_progress" not in st.session_state:
    st.session_state.conversation_progress = {}
if "shortlist_vehicles" not in st.session_state:
    st.session_state.shortlist_vehicles = []
if "dialogue_phase" not in st.session_state:
    st.session_state.dialogue_phase = None

with st.sidebar:
    st.subheader("Your details")
    user_email = st.text_input("Email (for purchase follow-up)", placeholder="you@example.com")

    st.subheader("Conversation progress")
    if st.session_state.conversation_progress:
        _render_progress(st.session_state.conversation_progress)
    else:
        st.caption("Start chatting to fill your profile.")

    if st.session_state.shortlist_vehicles:
        st.subheader("Shortlist")
        for vehicle in st.session_state.shortlist_vehicles[:6]:
            st.caption(
                f"#{vehicle['id']} {vehicle['year']} {vehicle['make']} {vehicle['model']} "
                f"· ${vehicle['price']:,.0f}"
            )

    if st.session_state.dialogue_phase:
        st.caption(f"Phase: `{st.session_state.dialogue_phase}`")

    st.subheader("Demo flows")
    for flow in DEMO_FLOWS:
        with st.expander(flow["title"]):
            for idx, turn in enumerate(flow["turns"], start=1):
                st.caption(f"{idx}. {turn}")

    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.conversation_progress = {}
        st.session_state.shortlist_vehicles = []
        st.session_state.dialogue_phase = None
        st.rerun()

    st.markdown("**Quick commands**")
    st.code("I'm looking for a family car")
    st.code("compare #55 vs #92")
    st.code("reserve vehicle #55")

chat_col, _ = st.columns([1, 0.01])

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("blocked"):
                st.warning(message["content"])
            else:
                st.markdown(message["content"])
            if message.get("vehicles"):
                _render_vehicle_cards(message["vehicles"])

    if prompt := st.chat_input("Tell me what you need — budget, family size, preferences…"):
        st.session_state.messages.append({"role": "user", "content": prompt, "blocked": False})
        with st.chat_message("user"):
            st.markdown(prompt)

        payload: dict = {"message": prompt}
        if st.session_state.session_id:
            payload["session_id"] = st.session_state.session_id
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
                    reply, data, is_blocked = _send_chat(payload)
                    vehicles = data.get("vehicles") or []
                    if data.get("session_id"):
                        st.session_state.session_id = data["session_id"]
                    if data.get("conversation_progress"):
                        st.session_state.conversation_progress = data["conversation_progress"]
                    if data.get("dialogue_phase"):
                        st.session_state.dialogue_phase = data["dialogue_phase"]
                    if vehicles:
                        st.session_state.shortlist_vehicles = vehicles
                    if settings.show_debug_meta:
                        meta = []
                        if data.get("dialogue_phase"):
                            meta.append(f"phase: `{data['dialogue_phase']}`")
                        if data.get("intent"):
                            meta.append(f"intent: `{data['intent']}`")
                        if data.get("rag_mode"):
                            meta.append(f"rag: `{data['rag_mode']}`")
                        if data.get("email_sent"):
                            meta.append("email sent")
                        if data.get("reserved_vehicle"):
                            v = data["reserved_vehicle"]
                            meta.append(f"reserved #{v['id']}")
                        if meta:
                            reply += "\n\n---\n" + " · ".join(meta)
                    if is_blocked:
                        st.warning(reply)
                    else:
                        st.markdown(reply)
                    if vehicles and not is_blocked:
                        _render_vehicle_cards(vehicles)
                except httpx.ConnectError:
                    reply = (
                        f"Cannot reach backend at {BACKEND_URL}.\n\n"
                        "Start API: `uvicorn backend.main:app --reload`"
                    )
                    is_blocked = False
                    vehicles = []
                    st.error(reply)
                except Exception:
                    reply = "Unexpected error talking to the backend."
                    is_blocked = False
                    vehicles = []
                    st.error(reply)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "blocked": is_blocked,
                "vehicles": vehicles if not is_blocked else [],
            }
        )
        st.rerun()
