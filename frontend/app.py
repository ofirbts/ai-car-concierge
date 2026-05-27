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
        "title": "Family road-trip buyer",
        "turns": [
            "I'm looking for a family car",
            "four people, budget 75000",
            "we do long family trips and cargo space matters",
            "need space for family trips",
            "what's the best value here?",
            "reserve vehicle #55",
        ],
    },
    {
        "title": "City electric buyer",
        "turns": [
            "I mostly drive in the city and want something quiet",
            "it's just me and my partner, budget 75000",
            "I prefer electric or hybrid",
            "what would you personally shortlist first?",
            "hold your top pick for me",
        ],
    },
    {
        "title": "Budget-conscious practical buyer",
        "turns": [
            "I need a practical family car but I'm price-sensitive",
            "family of 4, mostly city and weekend drives",
            "budget is 60000 max",
            "these feel expensive, can we go cheaper?",
            "which one is the smartest value-for-money choice?",
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


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg0: #070b13;
            --bg1: #0e1625;
            --glass: rgba(17, 25, 40, 0.55);
            --stroke: rgba(126, 189, 255, 0.22);
            --text: #f2f7ff;
            --muted: #9db3cc;
            --cyan: #55c8ff;
            --blue: #579dff;
        }
        .stApp {
            background: radial-gradient(1200px 600px at 10% -10%, rgba(68,140,255,.25), transparent 40%),
                        radial-gradient(1000px 500px at 95% 0%, rgba(85,200,255,.18), transparent 45%),
                        linear-gradient(180deg, var(--bg1), var(--bg0));
            color: var(--text);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #070b13 100%);
            border-right: 1px solid var(--stroke);
        }
        section[data-testid="stSidebar"] * {
            color: var(--text);
        }
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] textarea {
            background: rgba(12, 20, 34, 0.9);
            border: 1px solid var(--stroke);
            color: var(--text);
            border-radius: 10px;
        }
        section[data-testid="stSidebar"] .stProgress > div > div {
            background: linear-gradient(90deg, #3f9dff, #55c8ff);
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid var(--stroke);
            border-radius: 12px;
            background: rgba(14, 22, 37, 0.72);
        }
        section[data-testid="stSidebar"] button {
            border-radius: 12px;
            border: 1px solid var(--stroke);
            background: rgba(18, 28, 46, 0.85);
            color: var(--text);
        }
        section[data-testid="stSidebar"] button:hover {
            border-color: rgba(126, 189, 255, 0.45);
            color: #bfe7ff;
        }
        section[data-testid="stSidebar"] pre,
        section[data-testid="stSidebar"] code {
            background: rgba(10, 16, 28, 0.9);
            border: 1px solid var(--stroke);
            color: #c9def5;
            border-radius: 8px;
        }
        .main > div {
            padding-top: 1.2rem;
        }
        .aura-hero {
            border: 1px solid var(--stroke);
            background: linear-gradient(140deg, rgba(16,24,38,.72), rgba(10,16,28,.62));
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 14px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 16px 40px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06);
        }
        .aura-title {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: .2px;
            margin-bottom: 4px;
        }
        .aura-sub {
            color: var(--muted);
            font-size: .95rem;
        }
        .aura-orb {
            position: absolute;
            right: 22px;
            top: 18px;
            width: 92px;
            height: 92px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 30%, #d6efff 0%, #82d7ff 18%, #2e7cff 58%, #0f1f3e 100%);
            box-shadow: 0 0 0 10px rgba(90,168,255,.08), 0 0 48px rgba(63,157,255,.45);
            animation: pulseOrb 3.2s ease-in-out infinite;
        }
        .chip-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 12px;
        }
        .aura-chip {
            border: 1px solid var(--stroke);
            background: rgba(14, 22, 35, 0.68);
            color: #dce9f8;
            border-radius: 999px;
            padding: 6px 11px;
            font-size: .78rem;
            line-height: 1;
        }
        .chat-wrap {
            display: flex;
            margin: 10px 0;
        }
        .chat-wrap.user {
            justify-content: flex-end;
        }
        .chat-wrap.assistant {
            justify-content: flex-start;
        }
        .chat-bubble {
            max-width: 84%;
            border-radius: 14px;
            padding: 10px 12px;
            border: 1px solid var(--stroke);
            backdrop-filter: blur(8px);
            animation: fadeUp .3s cubic-bezier(.22,1,.36,1);
        }
        .chat-wrap.user .chat-bubble {
            background: linear-gradient(140deg, rgba(64,117,228,.45), rgba(52,87,165,.35));
        }
        .chat-wrap.assistant .chat-bubble {
            background: linear-gradient(140deg, rgba(18,26,42,.8), rgba(12,20,34,.75));
        }
        .chat-wrap.blocked .chat-bubble {
            background: linear-gradient(140deg, rgba(122,40,40,.58), rgba(82,22,22,.52));
            border-color: rgba(255,124,124,.35);
        }
        .aura-card {
            border: 1px solid var(--stroke);
            background: linear-gradient(155deg, rgba(19,29,45,.8), rgba(11,18,31,.75));
            border-radius: 14px;
            padding: 12px;
            margin: 8px 0;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
            transition: transform .2s ease, box-shadow .2s ease;
        }
        .aura-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.06);
        }
        .aura-card-title {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 4px;
            color: #eff6ff;
        }
        .aura-card-meta {
            color: var(--muted);
            font-size: .86rem;
            margin-bottom: 6px;
        }
        .aura-price {
            font-size: 1.08rem;
            font-weight: 700;
            color: #bfe7ff;
        }
        .aura-section-title {
            margin: 14px 0 6px;
            color: #d8e7f8;
            font-size: .93rem;
            text-transform: uppercase;
            letter-spacing: .8px;
        }
        @keyframes pulseOrb {
            0% { transform: scale(1); box-shadow: 0 0 0 10px rgba(90,168,255,.08), 0 0 40px rgba(63,157,255,.30); }
            50% { transform: scale(1.04); box-shadow: 0 0 0 12px rgba(90,168,255,.14), 0 0 56px rgba(63,157,255,.52); }
            100% { transform: scale(1); box-shadow: 0 0 0 10px rgba(90,168,255,.08), 0 0 40px rgba(63,157,255,.30); }
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <div class="aura-hero">
            <div class="aura-title">AURA AI Advisor</div>
            <div class="aura-sub">Premium conversational automotive guidance</div>
            <div class="chip-row">
                <span class="aura-chip">Memory-aware</span>
                <span class="aura-chip">Reasoning-first</span>
                <span class="aura-chip">Shortlist intelligence</span>
                <span class="aura-chip">Reservation-ready</span>
            </div>
            <div class="aura-orb"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _should_show_vehicle_cards(messages: list[dict], index: int, message: dict) -> bool:
    if message.get("role") != "assistant" or not message.get("vehicles"):
        return False
    if message.get("show_vehicle_cards") is False:
        return False
    if message.get("show_vehicle_cards") is True:
        return True
    last_idx = -1
    for idx, item in enumerate(messages):
        if item.get("role") == "assistant" and item.get("vehicles"):
            last_idx = idx
    return index == last_idx


def _render_vehicle_cards(vehicles: list[dict], *, title: str = "Recommended vehicles") -> None:
    if not vehicles:
        return
    st.markdown(f"<div class='aura-section-title'>{title}</div>", unsafe_allow_html=True)
    for vehicle in vehicles[:3]:
        pending = " · Pending De-listing" if vehicle.get("pending_delisting") else ""
        st.markdown(
            (
                "<div class='aura-card'>"
                f"<div class='aura-card-title'>#{vehicle['id']} · {vehicle['year']} {vehicle['make']} {vehicle['model']}</div>"
                f"<div class='aura-card-meta'>{vehicle.get('color','')} · {vehicle.get('fuel_type','')} · stock {vehicle.get('stock_count',0)}{pending}</div>"
                f"<div class='aura-price'>${vehicle['price']:,.0f}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


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
_inject_theme()
_render_hero()

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
    info_chips = []
    if st.session_state.dialogue_phase:
        info_chips.append(f"Phase: {st.session_state.dialogue_phase}")
    if st.session_state.session_id:
        info_chips.append("Session active")
    if st.session_state.shortlist_vehicles:
        info_chips.append(f"Shortlist {len(st.session_state.shortlist_vehicles)}")
    if info_chips:
        st.markdown(
            "<div class='chip-row'>" + "".join([f"<span class='aura-chip'>{chip}</span>" for chip in info_chips]) + "</div>",
            unsafe_allow_html=True,
        )

    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        blocked = message.get("blocked", False)
        role_cls = "user" if role == "user" else "assistant"
        if blocked:
            role_cls = f"{role_cls} blocked"
        body = message["content"].replace("\n", "<br>")
        st.markdown(
            (
                f"<div class='chat-wrap {role_cls}'>"
                f"<div class='chat-bubble'>{body}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if _should_show_vehicle_cards(st.session_state.messages, idx, message):
            reserved = message.get("reserved") or (
                len(message["vehicles"]) == 1
                and message.get("content", "").lower().startswith("done")
            )
            title = "Reserved" if reserved else "Top picks"
            _render_vehicle_cards(message["vehicles"], title=title)

    if prompt := st.chat_input("Tell me what you need — budget, family size, preferences…"):
        st.session_state.messages.append({"role": "user", "content": prompt, "blocked": False})
        st.markdown(
            f"<div class='chat-wrap user'><div class='chat-bubble'>{prompt}</div></div>",
            unsafe_allow_html=True,
        )

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

        show_cards = True
        data: dict = {}
        reserved_flag = False
        with st.spinner("Aura is reasoning..."):
            try:
                reply, data, is_blocked = _send_chat(payload)
                vehicles = data.get("vehicles") or []
                if data.get("session_id"):
                    st.session_state.session_id = data["session_id"]
                if data.get("conversation_progress"):
                    st.session_state.conversation_progress = data["conversation_progress"]
                if data.get("dialogue_phase"):
                    st.session_state.dialogue_phase = data["dialogue_phase"]
                if data.get("reserved_vehicle"):
                    rv = data["reserved_vehicle"]
                    vehicles = [rv]
                if vehicles:
                    st.session_state.shortlist_vehicles = vehicles
                show_cards = data.get("show_vehicle_cards", True)
                reserved_flag = bool(data.get("reserved_vehicle"))
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
            except httpx.ConnectError:
                reply = (
                    f"Cannot reach backend at {BACKEND_URL}.\n\n"
                    "Start API: `uvicorn backend.main:app --reload`"
                )
                is_blocked = False
                vehicles = []
                show_cards = False
            except Exception:
                reply = "Unexpected error talking to the backend."
                is_blocked = False
                vehicles = []
                show_cards = False

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "blocked": is_blocked,
                "vehicles": vehicles if not is_blocked else [],
                "show_vehicle_cards": show_cards if not is_blocked else False,
                "reserved": reserved_flag if not is_blocked else False,
            }
        )
        st.rerun()
