from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from backend.conversation_state import ConversationState, DialoguePhase, save_conversation_state
from backend.conversational_nlg import (
    generate_clarifying_question,
    generate_comparison,
    generate_purchase_prompt,
    generate_recommendations,
    generate_reserve_prompt,
    generate_welcome,
)
from backend.database import Vehicle, get_vehicle_by_id
from backend.dialogue_analysis import analyze_dialogue_turn
from backend.dialogue_policy import choose_dialogue_policy
from backend.intent import (
    ExtractedIntent,
    IntentKind,
    extract_email,
    extract_price_max,
    extract_vehicle_id,
)
from backend.inventory_retrieval import (
    detect_semantic_profiles,
    family_fit_score,
    hybrid_search_inventory,
    infer_body_type,
)
from backend.rag_service import PolicyRAGService

SALES_SIGNALS = (
    r"\bfamily\b",
    r"\bkids?\b",
    r"\bchildren\b",
    r"\blooking for\b",
    r"\bhelp me\b",
    r"\brecommend\b",
    r"\bsuggest\b",
    r"\bnot sure\b",
    r"\bdon't know\b",
    r"\bfirst car\b",
    r"\bcompare\b",
    r"\bbest value\b",
    r"\bmost affordable\b",
    r"\bwhat fits\b",
    r"\bמחפש\b",
    r"\bמשפחה\b",
    r"\bתקציב\b",
    r"\bהמלץ\b",
    r"\bלא יודע\b",
)

RESERVE_SIGNALS = (r"\breserve\b", r"\bhold\b", r"\bbook\b", r"\bשמור\b")
PURCHASE_SIGNALS = (r"\bbuy\b", r"\bpurchase\b", r"\border\b", r"\bלקנות\b")
COMPARE_SIGNALS = (
    r"\bcompare\b",
    r"\bvs\b",
    r"\bversus\b",
    r"\bbetter\b",
    r"\bwhich one\b",
    r"\bbest value\b",
    r"\bmost affordable\b",
    r"\bwhat fits best\b",
    r"\bהכי\b",
)
SPACE_SIGNALS = (r"\bspace\b", r"\broomy\b", r"\bspacious\b", r"\bמרווח\b")
FUEL_SIGNALS = (r"\bfuel\b", r"\bgas\b", r"\belectric\b", r"\bhybrid\b", r"\bחיסכון\b", r"\bדלק\b")

DIALOGUE_CONTRACT = {
    "min_discovery_turns_before_full_reco": 2,
    "force_clarify_on_objection": True,
    "force_language_lock": True,
    "max_similar_replies_in_row": 1,
    "force_progress_after_stall_turns": 2,
}


@dataclass
class SalesTurnResult:
    reply: str
    state: ConversationState
    vehicles: list[Vehicle]
    intent: IntentKind
    phase: DialoguePhase
    rag_mode: str | None = None
    delegate: str | None = None
    vehicle_id: int | None = None
    show_vehicle_cards: bool = True


def has_sales_signal(message: str) -> bool:
    lower = message.lower()
    return any(re.search(pattern, lower) for pattern in SALES_SIGNALS)


def should_use_sales_dialogue(
    extracted: ExtractedIntent,
    message: str,
    state: ConversationState | None,
    session_id: str | None,
) -> bool:
    ongoing = bool(session_id or (state and state.turn_count > 0))
    if extracted.intent == IntentKind.RESERVE_INTENT and extract_vehicle_id(message):
        return ongoing
    if extracted.intent == IntentKind.PURCHASE_INTENT:
        return ongoing
    if session_id or (state and state.turn_count > 0):
        return True
    if extracted.intent == IntentKind.GENERAL_CHAT:
        return True
    if has_sales_signal(message):
        return True
    if detect_semantic_profiles(message):
        return True
    if extracted.intent == IntentKind.INVENTORY_SEARCH:
        if extracted.make and not has_sales_signal(message):
            return False
        return True
    return False


def _matches_any(message: str, patterns: tuple[str, ...]) -> bool:
    lower = message.lower()
    return any(re.search(p, lower) for p in patterns)


def _parse_passengers(message: str) -> int | None:
    lower = message.lower()
    stripped = lower.strip()
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    for word, value in word_numbers.items():
        if re.search(rf"\b{word}\s+(people|passengers|riders)\b", lower):
            return value
    for word, value in word_numbers.items():
        if stripped == word:
            return value
    if re.fullmatch(r"\d+", stripped):
        value = int(stripped)
        if 1 <= value <= 9:
            return value
    if re.search(r"\bfamily of (\d+)\b", lower):
        return int(re.search(r"\bfamily of (\d+)\b", lower).group(1))
    if re.search(r"\b(\d+)\s*(people|passengers|riders)\b", lower):
        return int(re.search(r"\b(\d+)\s*(people|passengers|riders)\b", lower).group(1))
    if re.search(r"\bone kid\b|\bone child\b|\bילד אחד\b", lower):
        return 3
    if re.search(r"\btwo kids\b|\btwo children\b|\bשני ילדים\b", lower):
        return 4
    if re.search(r"\bcouple\b|\btwo people\b|\bזוג\b", lower):
        return 2
    if re.search(r"\bsolo\b|\balone\b|\bjust me\b|\bonly me\b|\bלבד\b|\bרק אני\b", lower):
        return 1
    if re.search(r"\bme and my partner\b|\bmy partner and i\b", lower):
        return 2
    if re.search(r"\bpair\b", lower):
        return 2
    if re.search(r"\bbaby\b|\binfant\b|\bתינוק\b", lower):
        return 3
    return None


def _parse_budget(message: str, extracted: ExtractedIntent) -> float | None:
    price = extract_price_max(message)
    if price is not None:
        return price
    if extracted.price_max is not None:
        return extracted.price_max
    lower = message.lower()
    match = re.search(
        r"(?:budget|around|about|roughly|up to|max)\s*(?:is|of|around|about)?\s*\$?\s*([\d,]+)",
        lower,
    )
    if match:
        return float(match.group(1).replace(",", ""))
    match = re.search(r"\$?\s*([\d,]+)\s*(?:budget|max|total)", lower)
    if match:
        return float(match.group(1).replace(",", ""))
    match = re.search(r"\b(\d{2,3})\s*k\b", lower)
    if match:
        return float(match.group(1)) * 1000
    return None


def _parse_body_type(message: str) -> str | None:
    lower = message.lower()
    if re.search(r"\bsuv\b", lower):
        return "suv"
    if re.search(r"\bsedan\b", lower):
        return "sedan"
    if re.search(r"\bsports?\b|\bcoupe\b", lower):
        return "sports"
    return None


def _parse_fuel(message: str) -> str | None:
    lower = message.lower()
    if "electric" in lower or "ev" in lower:
        return "Electric"
    if "hybrid" in lower:
        return "Hybrid"
    if "plug-in" in lower or "plugin" in lower:
        return "Plug-in Hybrid"
    if "gas" in lower or "gasoline" in lower:
        return "Gasoline"
    return None


def _parse_use_case(message: str) -> str | None:
    lower = message.lower()
    if re.search(r"(without|no|not)\s+family", lower) or re.search(r"בלי\s+טיולים\s+משפחתיים", lower):
        return "city driving"
    if re.search(r"\bמשהו אחר\b", lower):
        return "city driving"
    if re.search(r"\bcity\b", lower) and re.search(r"\bweekend\b", lower):
        return "city and weekend drives"
    if re.search(r"\bcity\b|\burban\b|\bcommute\b|\bעיר\b", lower):
        return "city driving"
    if re.search(r"\bfamily\b|\bkids\b|\bchildren\b|\bמשפחה\b", lower):
        return "family trips"
    if re.search(r"\bhighway\b|\blong drive\b", lower):
        return "highway travel"
    if re.search(r"\bwork\b|\bdaily\b", lower):
        return "daily commute"
    return None


def _parse_timeline(message: str) -> str | None:
    lower = message.lower()
    if re.search(r"\basap\b|\bsoon\b|\bthis week\b|\bמיד\b", lower):
        return "soon"
    if re.search(r"\bmonth\b|\bnext month\b", lower):
        return "within a month"
    return None


def update_state_from_message(
    state: ConversationState,
    message: str,
    extracted: ExtractedIntent,
    user_email: str | None,
) -> ConversationState:
    if _is_topic_reset_request(message):
        state.use_case = None
        state.body_type = None
        state.space_priority = None
        state.last_refinement_key = None
        state.last_recommended_ids = []

    passengers = _parse_passengers(message)
    if passengers is not None:
        state.passengers = passengers
        state.family_size = passengers

    budget = _parse_budget(message, extracted)
    if budget is not None:
        state.budget = budget

    body = _parse_body_type(message)
    if body:
        state.body_type = body
    elif passengers and passengers >= 4 and not state.body_type:
        state.body_type = "suv"
    elif passengers == 2 and not state.body_type:
        state.body_type = "sedan"

    fuel = _parse_fuel(message)
    if fuel:
        state.fuel_preference = fuel

    use_case = _parse_use_case(message)
    if use_case:
        state.use_case = use_case
    else:
        for profile in detect_semantic_profiles(message):
            if profile != "budget":
                state.use_case = profile.replace("_", " ")
                break

    timeline = _parse_timeline(message)
    if timeline:
        state.timeline = timeline

    if _matches_any(message, SPACE_SIGNALS):
        state.space_priority = "space"
    elif _matches_any(message, FUEL_SIGNALS):
        state.space_priority = "fuel"

    email = extract_email(message, user_email)
    if email:
        state.contact_email = email

    if extracted.make and not state.body_type:
        pass

    vehicle_id = extract_vehicle_id(message) or extracted.vehicle_id
    if vehicle_id is not None and vehicle_id not in state.shortlist_ids:
        state.shortlist_ids.append(vehicle_id)

    return state


def _next_discovery_question(state: ConversationState) -> str | None:
    if state.passengers is None and state.family_size is None:
        return "How many people usually ride along with you?"
    if not state.use_case and not state.body_type:
        return "What will you mainly use the car for — family trips, city driving, or something else?"
    if state.budget is None:
        return "What's your target budget (roughly)?"
    if state.space_priority is None and (state.passengers or 0) >= 3:
        return "Do you care more about interior space or fuel economy?"
    return None


def _contextual_discovery_question(state: ConversationState, question: str) -> str:
    if question == "What's your target budget (roughly)?":
        parts: list[str] = []
        if state.passengers:
            parts.append(f"{state.passengers} riders")
        if state.use_case:
            parts.append(state.use_case)
        if parts:
            return f"Got it — {', '.join(parts)}. What budget should I stay within?"
    if question.startswith("How many people") and state.use_case:
        return f"For {state.use_case}, how many people usually ride along?"
    return question


def _resolve_compare_ids(state: ConversationState, message: str) -> list[int]:
    ids = [int(x) for x in re.findall(r"#(\d+)", message)]
    if len(ids) >= 2:
        return ids[:4]
    if state.last_recommended_ids:
        return state.last_recommended_ids[: min(3, len(state.last_recommended_ids))]
    if state.shortlist_ids:
        return state.shortlist_ids[: min(3, len(state.shortlist_ids))]
    return []


def _wants_new_search(message: str) -> bool:
    lower = message.lower()
    return bool(
        re.search(
            r"\b(show|find|search|other|different|another|more options|something else|samthing else|somthing else)\b|משהו אחר",
            lower,
        )
    )


def _is_topic_reset_request(message: str) -> bool:
    lower = message.lower()
    return bool(
        re.search(
            r"\b(something else|samthing else|somthing else|another direction|different direction|not family|without family)\b|משהו אחר|בלי טיולים משפחתיים",
            lower,
        )
    )


def _is_smalltalk(message: str) -> bool:
    lower = message.lower().strip()
    return bool(
        re.search(
            r"\b(how old are you|how old are u|who are you|what are you|tell me about yourself|joke)\b|בן כמה|מי אתה|מה אתה",
            lower,
        )
    )


def _is_preference_refinement(message: str, state: ConversationState) -> bool:
    if not state.last_recommended_ids:
        return False
    if _wants_new_search(message):
        return False
    if _matches_any(
        message,
        COMPARE_SIGNALS + RESERVE_SIGNALS + PURCHASE_SIGNALS,
    ):
        return False
    return bool(
        _matches_any(message, SPACE_SIGNALS + FUEL_SIGNALS)
        or _parse_use_case(message)
        or _parse_fuel(message)
        or _parse_body_type(message)
    )


def _is_unclear_followup(message: str, state: ConversationState) -> bool:
    if not state.last_recommended_ids:
        return False
    lower = message.lower().strip()
    if len(lower) > 28:
        return False
    if _matches_any(message, COMPARE_SIGNALS + RESERVE_SIGNALS + PURCHASE_SIGNALS):
        return False
    if _is_budget_objection(message):
        return False
    if _parse_use_case(message) or _parse_fuel(message) or _parse_body_type(message):
        return False
    return bool(re.search(r"\b(no|nah|not this|other|else)\b|לא|אחר", lower))


def _lang(state: ConversationState, en: str, he: str) -> str:
    return he if state.language_preference == "he" else en


def _explicit_fast_options_request(message: str) -> bool:
    lower = message.lower()
    return bool(
        re.search(r"\b(show me options now|just show options|skip questions|just recommend)\b|תראה אופציות עכשיו", lower)
    )


def _too_similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.82


def _guard_repetition_and_progress(state: ConversationState, reply: str) -> str:
    previous = state.last_assistant_reply or ""
    if previous and _too_similar(reply, previous):
        state.repetition_count += 1
    else:
        state.repetition_count = 0
    state.last_assistant_reply = reply
    if state.repetition_count >= DIALOGUE_CONTRACT["max_similar_replies_in_row"]:
        state.stall_turns += 1
    else:
        state.stall_turns = 0
    if state.stall_turns >= DIALOGUE_CONTRACT["force_progress_after_stall_turns"]:
        state.stall_turns = 0
        if state.language_preference == "he":
            return "נעצור רגע ונדייק. מה הכי חשוב עכשיו: מחיר, מרווח, נוחות או חסכון בדלק?"
        return "Let's recalibrate quickly. What matters most now: price, space, comfort, or efficiency?"
    return reply


def _is_budget_objection(message: str) -> bool:
    lower = message.lower()
    return bool(
        re.search(
            r"\b(too expensive|expensive|cheaper|lower budget|over budget|budget issue|can't afford)\b",
            lower,
        )
    )


def _handle_budget_objection(
    state: ConversationState,
    message: str,
) -> SalesTurnResult | None:
    if not state.last_recommended_ids:
        return None
    current = _vehicles_from_ids(state.last_recommended_ids[:6])
    if not current:
        return None
    floor_price = min(v.price for v in current)
    cap = floor_price - 500
    if state.budget:
        cap = min(cap, state.budget * 0.85)
    cap = max(cap, 12000)
    extracted = ExtractedIntent(
        intent=IntentKind.INVENTORY_SEARCH,
        price_max=cap,
    )
    query = message.strip() or "affordable practical value under budget"
    result = hybrid_search_inventory(query, state=state, extracted=extracted, limit=6)
    fresh = [v for v in result.vehicles if v.price < floor_price]
    if not fresh:
        fresh = sorted(current, key=lambda v: v.price)[:3]
        best = fresh[0]
        alternatives = ", ".join(f"#{v.id}" for v in fresh[:3])
        reply = (
            f"Fair point — within your criteria, #{best.id} at ${best.price:,.0f} is already "
            f"the strongest value in this shortlist. I can compare {alternatives}, "
            f"or hold #{best.id} if you want to move now."
        )
        show_cards = False
    else:
        fresh = sorted(fresh, key=lambda v: v.price)[:3]
        state.last_recommended_ids = [v.id for v in fresh]
        state.shortlist_ids = list(dict.fromkeys(state.shortlist_ids + state.last_recommended_ids))
        state.last_refinement_key = None
        best = fresh[0]
        alts = ", ".join(f"#{v.id}" for v in fresh[1:3])
        reply = (
            f"Fair point — I looked for stronger value under ${cap:,.0f}. "
            f"I'd start with the {best.year} {best.make} {best.model} (#{best.id}) at ${best.price:,.0f}."
        )
        if alts:
            reply += f" Also worth a look: {alts}."
        reply += f" Want a compare, or should I hold #{best.id}?"
        show_cards = True
    reply = _guard_repetition_and_progress(state, reply)
    state.phase = DialoguePhase.RECOMMENDING
    save_conversation_state(state)
    return SalesTurnResult(
        reply=reply,
        state=state,
        vehicles=fresh,
        intent=IntentKind.INVENTORY_SEARCH,
        phase=state.phase,
        rag_mode="sales_dialogue+budget_objection",
        show_vehicle_cards=show_cards,
    )


def _vehicles_from_ids(ids: list[int]) -> list[Vehicle]:
    out: list[Vehicle] = []
    for vid in ids:
        vehicle = get_vehicle_by_id(vid)
        if vehicle is not None:
            out.append(vehicle)
    return out


def _refinement_key(message: str, state: ConversationState) -> str:
    if _matches_any(message, SPACE_SIGNALS) or state.space_priority == "space":
        return "space"
    if _matches_any(message, FUEL_SIGNALS) or state.fuel_preference:
        return f"fuel:{state.fuel_preference or ''}"
    body = _parse_body_type(message) or state.body_type
    if body:
        return f"body:{body}"
    use_case = _parse_use_case(message) or state.use_case
    if use_case:
        return f"use:{use_case}"
    return message.strip().lower()[:48]


def _handle_preference_refinement(
    state: ConversationState,
    message: str,
) -> SalesTurnResult | None:
    if not _is_preference_refinement(message, state):
        return None
    vehicles = _vehicles_from_ids(state.last_recommended_ids[:4])
    if not vehicles:
        return None
    ranked = sorted(
        vehicles,
        key=lambda v: (family_fit_score(v, state), -v.price),
        reverse=True,
    )
    best = ranked[0]
    ids_hint = " vs ".join(f"#{v.id}" for v in ranked[:3])
    refine_key = _refinement_key(message, state)
    if state.last_refinement_key == refine_key:
        reply = (
            f"We are aligned on that priority, so I'd still lead with the "
            f"{best.year} {best.make} {best.model} (#{best.id}). "
            f"Do you want a quick compare ({ids_hint}), or should I hold #{best.id}?"
        )
    elif state.space_priority == "space":
        reply = (
            f"Since space matters most for you, I'd prioritize comfort and cabin room. "
            f"From your shortlist, the {best.year} {best.make} {best.model} (#{best.id}) "
            f"is still the strongest overall fit. "
            f"Want a quick compare ({ids_hint}), or should I hold #{best.id} now?"
        )
    else:
        reply = (
            f"Got it. Based on your preference update, your top shortlist pick is still the "
            f"{best.year} {best.make} {best.model} (#{best.id}). "
            f"We can compare ({ids_hint}) quickly, or I can reserve #{best.id} now."
        )
    reply = _guard_repetition_and_progress(state, reply)
    state.last_refinement_key = refine_key
    state.phase = DialoguePhase.RECOMMENDING
    save_conversation_state(state)
    return SalesTurnResult(
        reply=reply,
        state=state,
        vehicles=[],
        intent=IntentKind.INVENTORY_SEARCH,
        phase=state.phase,
        rag_mode="sales_dialogue+preference_refine",
        show_vehicle_cards=False,
    )


def _pick_vehicle_for_action(state: ConversationState, message: str) -> Vehicle | None:
    vid = extract_vehicle_id(message)
    if vid is not None:
        return get_vehicle_by_id(vid)
    if state.shortlist_ids:
        return get_vehicle_by_id(state.shortlist_ids[-1])
    if state.last_recommended_ids:
        return get_vehicle_by_id(state.last_recommended_ids[0])
    return None


def handle_sales_turn(
    message: str,
    extracted: ExtractedIntent,
    state: ConversationState,
    user_email: str | None,
    rag: PolicyRAGService,
) -> SalesTurnResult:
    state.bump_turn()
    update_state_from_message(state, message, extracted, user_email)
    analysis = analyze_dialogue_turn(message, state)
    policy = choose_dialogue_policy(state, analysis)

    if policy.action == "switch_language_he":
        state.language_preference = "he"
        state.phase = DialoguePhase.DISCOVERY if not state.has_discovery_basics() else DialoguePhase.RECOMMENDING
        save_conversation_state(state)
        return SalesTurnResult(
            reply="מעולה, ממשיך בעברית. כדי לדייק אותך מהר, מה התקציב ומה הכי חשוב לך כרגע?",
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+language_switch",
            show_vehicle_cards=False,
        )
    if policy.action == "switch_language_en":
        state.language_preference = "en"
        state.phase = DialoguePhase.DISCOVERY if not state.has_discovery_basics() else DialoguePhase.RECOMMENDING
        save_conversation_state(state)
        return SalesTurnResult(
            reply="Great, I'll continue in English. What should I optimize first: budget, comfort, efficiency, or space?",
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+language_switch",
            show_vehicle_cards=False,
        )
    if policy.action == "clarify_budget_low":
        state.phase = DialoguePhase.CLARIFICATION
        reply = _lang(
            state,
            "Just to confirm, did you mean around $200k or $20k?",
            "רק לוודא — התכוונת לסביבות 200 אלף דולר או 20 אלף?",
        )
        reply = _guard_repetition_and_progress(state, reply)
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+clarify_budget",
            show_vehicle_cards=False,
        )
    if policy.action == "clarify_budget_high":
        state.phase = DialoguePhase.CLARIFICATION
        reply = _lang(
            state,
            "That is a huge number. Want me to keep this in a practical family-car range, like $40k-$90k?",
            "זה מספר עצום. שנכוון לטווח ריאלי לרכב משפחתי, למשל 40–90 אלף דולר?",
        )
        reply = _guard_repetition_and_progress(state, reply)
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+clarify_budget",
            show_vehicle_cards=False,
        )
    if policy.action == "explain_product":
        state.phase = DialoguePhase.EXPLORATORY_GUIDANCE
        reply = _lang(
            state,
            "Great question. I help you pick a car through a short conversation, then I compare top fits and can reserve one when you are ready.",
            "שאלה מצוינת. אני עוזר לבחור רכב דרך שיחה קצרה, ואז משווה התאמות טובות ויכול גם לשמור רכב כשתרצה.",
        )
        reply = _guard_repetition_and_progress(state, reply)
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+explain_product",
            show_vehicle_cards=False,
        )
    if policy.action == "playful_response":
        state.phase = DialoguePhase.EXPLORATORY_GUIDANCE
        reply = _lang(
            state,
            "Love the vibe 😄. Want me to keep this practical, fun, or a mix of both?",
            "אהבתי את הווייב 😄 רוצה שנלך על פרקטי, כיף, או שילוב של שניהם?",
        )
        reply = _guard_repetition_and_progress(state, reply)
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+playful",
            show_vehicle_cards=False,
        )
    if policy.action == "smalltalk_repair":
        state.phase = DialoguePhase.RECOMMENDING if state.last_recommended_ids else DialoguePhase.DISCOVERY
        reply = (
            "I'm your AI car advisor. If you want, I can recalibrate now based on what matters most to you."
        )
        reply = _guard_repetition_and_progress(state, reply)
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+smalltalk",
            show_vehicle_cards=False,
        )
    if policy.action in {"repair_turn", "topic_shift_recalibrate"}:
        state.phase = DialoguePhase.DISCOVERY
        state.last_refinement_key = None
        state.last_recommended_ids = []
        save_conversation_state(state)
        follow_up = policy.question or "What matters most in your decision right now?"
        if state.language_preference == "he":
            follow_up = "הבנתי. מה היה פחות מדויק — מחיר, גודל, נוחות או צריכת דלק?"
        follow_up = _guard_repetition_and_progress(state, follow_up)
        return SalesTurnResult(
            reply=follow_up,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+repair_turn",
            show_vehicle_cards=False,
        )

    if _is_unclear_followup(message, state):
        state.phase = DialoguePhase.DISCOVERY
        reply = _lang(
            state,
            "Understood. What should I change first — price, size, fuel efficiency, or body style?",
            "הבנתי. מה לשנות קודם — מחיר, גודל, חסכון בדלק או סוג רכב?",
        )
        reply = _guard_repetition_and_progress(state, reply)
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+clarify_constraints",
            show_vehicle_cards=False,
        )

    if _is_budget_objection(message):
        budget_turn = _handle_budget_objection(state, message)
        if budget_turn is not None:
            return budget_turn

    if state.turn_count == 1 and not state.filled_slots():
        reply = generate_welcome(state)
        state.phase = DialoguePhase.DISCOVERY
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue",
        )

    if _matches_any(message, RESERVE_SIGNALS):
        vehicle = _pick_vehicle_for_action(state, message)
        if vehicle is None:
            reply = generate_clarifying_question(
                state,
                "Which vehicle would you like to reserve? You can say e.g. reserve vehicle #16.",
            )
            state.phase = DialoguePhase.RESERVE
            save_conversation_state(state)
            return SalesTurnResult(
                reply=reply,
                state=state,
                vehicles=[],
                intent=IntentKind.RESERVE_INTENT,
                phase=state.phase,
                rag_mode="sales_dialogue",
            )
        if vehicle.pending_delisting:
            state.phase = DialoguePhase.RESERVE
            save_conversation_state(state)
            return SalesTurnResult(
                reply=(
                    f"Vehicle #{vehicle.id} is Pending De-listing (pre-2022) and cannot be reserved. "
                    "Want alternatives from our 2022+ inventory?"
                ),
                state=state,
                vehicles=[vehicle],
                intent=IntentKind.RESERVE_INTENT,
                phase=state.phase,
                rag_mode="sales_dialogue",
            )
        state.phase = DialoguePhase.RESERVE
        save_conversation_state(state)
        return SalesTurnResult(
            reply="",
            state=state,
            vehicles=[vehicle],
            intent=IntentKind.RESERVE_INTENT,
            phase=state.phase,
            rag_mode="sales_dialogue",
            delegate="reserve",
            vehicle_id=vehicle.id,
        )

    if _matches_any(message, PURCHASE_SIGNALS) or (
        state.phase == DialoguePhase.PURCHASE and state.contact_email
    ):
        vehicle = _pick_vehicle_for_action(state, message)
        if not state.contact_email and not extract_email(message, user_email):
            reply = generate_purchase_prompt(state, vehicle)
            state.phase = DialoguePhase.PURCHASE
            save_conversation_state(state)
            return SalesTurnResult(
                reply=reply,
                state=state,
                vehicles=[vehicle] if vehicle else [],
                intent=IntentKind.PURCHASE_INTENT,
                phase=state.phase,
                rag_mode="sales_dialogue",
            )
        state.phase = DialoguePhase.PURCHASE
        save_conversation_state(state)
        return SalesTurnResult(
            reply="",
            state=state,
            vehicles=[vehicle] if vehicle else [],
            intent=IntentKind.PURCHASE_INTENT,
            phase=state.phase,
            rag_mode="sales_dialogue",
            delegate="purchase",
            vehicle_id=vehicle.id if vehicle else None,
        )

    if _matches_any(message, COMPARE_SIGNALS) or state.phase == DialoguePhase.COMPARING:
        compare_ids = _resolve_compare_ids(state, message)
        vehicles = [v for vid in compare_ids if (v := get_vehicle_by_id(vid)) is not None]
        if len(vehicles) < 2:
            reply = generate_clarifying_question(
                state,
                "Which two or three vehicles should I compare? Mention ids like #16 and #22.",
            )
            state.phase = DialoguePhase.COMPARING
            save_conversation_state(state)
            return SalesTurnResult(
                reply=reply,
                state=state,
                vehicles=vehicles,
                intent=IntentKind.INVENTORY_SEARCH,
                phase=state.phase,
                rag_mode="sales_dialogue",
            )
        reply = generate_comparison(state, vehicles)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.COMPARING
        state.compare_vehicle_ids = [v.id for v in vehicles]
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=vehicles,
            intent=IntentKind.INVENTORY_SEARCH,
            phase=state.phase,
            rag_mode="sales_dialogue",
            show_vehicle_cards=False,
        )

    refinement = _handle_preference_refinement(state, message)
    if refinement is not None:
        return refinement

    question = _next_discovery_question(state)
    if (
        question is None
        and state.turn_count < DIALOGUE_CONTRACT["min_discovery_turns_before_full_reco"]
        and not _explicit_fast_options_request(message)
    ):
        question = "Before I shortlist, do you drive mostly in the city or on longer highway trips?"
    if question and not state.has_discovery_basics():
        reply = generate_clarifying_question(state, _contextual_discovery_question(state, question))
        state.phase = DialoguePhase.DISCOVERY
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue",
        )

    retrieval = hybrid_search_inventory(message, state=state, extracted=extracted, limit=3)
    vehicles = retrieval.vehicles
    state.last_recommended_ids = [v.id for v in vehicles]
    state.last_refinement_key = None
    state.shortlist_ids = list(dict.fromkeys(state.shortlist_ids + state.last_recommended_ids))
    state.phase = DialoguePhase.RECOMMENDING
    reply = generate_recommendations(state, vehicles)
    reply = _guard_repetition_and_progress(state, reply)
    save_conversation_state(state)
    return SalesTurnResult(
        reply=reply,
        state=state,
        vehicles=vehicles,
        intent=IntentKind.INVENTORY_SEARCH,
        phase=state.phase,
        rag_mode=f"sales_dialogue+{retrieval.retrieval_mode}",
    )
