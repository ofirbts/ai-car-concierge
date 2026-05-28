from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from backend.conversation_state import ConversationState, DialoguePhase, save_conversation_state
from backend.conversation_understanding import (
    ConvIntent,
    ConversationUnderstanding,
    understand_conversation,
)
from backend.conversational_nlg import (
    generate_ask_budget,
    generate_ask_city_vs_highway,
    generate_ask_passengers,
    generate_ask_use_case,
    generate_clarifying_question,
    generate_comparison,
    generate_criteria_explanation,
    generate_exploratory_response,
    generate_greeting_response,
    generate_purchase_prompt,
    generate_recommendations,
    generate_repair_turn_response,
    generate_reserve_prompt,
    generate_smalltalk_response,
    generate_topic_shift_response,
    generate_welcome,
)
from backend.database import Vehicle, get_vehicle_by_id
from backend.dialogue_policy import PolicyAction, PolicyDecision, decide_policy
from backend.intent import (
    ExtractedIntent,
    IntentKind,
    extract_email,
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
    search_explanation: dict | None = None


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


def _too_similar(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.82


def _guard_repetition_and_progress(state: ConversationState, reply: str) -> str:
    previous = state.last_assistant_reply or ""
    if previous and _too_similar(reply, previous):
        state.repetition_count += 1
    else:
        state.repetition_count = 0
    state.last_assistant_reply = reply
    if state.repetition_count >= 1:
        state.stall_turns += 1
    else:
        state.stall_turns = 0
    if state.stall_turns >= DIALOGUE_CONTRACT["force_progress_after_stall_turns"]:
        state.stall_turns = 0
        if state.language_preference == "he":
            return "נעצור רגע ונדייק. מה הכי חשוב עכשיו: מחיר, מרווח, נוחות או חסכון בדלק?"
        return "Let's recalibrate quickly. What matters most now: price, space, comfort, or efficiency?"
    return reply


def _lang(state: ConversationState, en: str, he: str) -> str:
    return he if state.language_preference == "he" else en


def _explicit_fast_options_request(message: str) -> bool:
    lower = message.lower()
    return bool(
        re.search(
            r"\b(show me options now|just show options|skip questions|just recommend)\b|תראה אופציות עכשיו",
            lower,
        )
    )


def _vehicles_from_ids(ids: list[int]) -> list[Vehicle]:
    out: list[Vehicle] = []
    for vid in ids:
        vehicle = get_vehicle_by_id(vid)
        if vehicle is not None:
            out.append(vehicle)
    return out


def _resolve_compare_ids(state: ConversationState, message: str) -> list[int]:
    ids = [int(x) for x in re.findall(r"#(\d+)", message)]
    if len(ids) >= 2:
        return ids[:4]
    if state.last_recommended_ids:
        return state.last_recommended_ids[: min(3, len(state.last_recommended_ids))]
    if state.shortlist_ids:
        return state.shortlist_ids[: min(3, len(state.shortlist_ids))]
    return []


def _pick_vehicle_for_action(state: ConversationState, message: str) -> Vehicle | None:
    vid = extract_vehicle_id(message)
    if vid is not None:
        return get_vehicle_by_id(vid)
    if state.shortlist_ids:
        return get_vehicle_by_id(state.shortlist_ids[-1])
    if state.last_recommended_ids:
        return get_vehicle_by_id(state.last_recommended_ids[0])
    return None


def _refinement_key(message: str, state: ConversationState) -> str:
    if _matches_any(message, SPACE_SIGNALS) or state.space_priority == "space":
        return "space"
    if _matches_any(message, FUEL_SIGNALS) or state.fuel_preference:
        return f"fuel:{state.fuel_preference or ''}"
    if state.body_type:
        return f"body:{state.body_type}"
    if state.use_case:
        return f"use:{state.use_case}"
    return message.strip().lower()[:48]


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


def update_state_from_message(
    state: ConversationState,
    message: str,
    extracted: ExtractedIntent,
    user_email: str | None,
) -> ConversationState:
    understanding = understand_conversation(message, state)
    return _update_state_from_understanding(state, understanding, message, extracted, user_email)


def _update_state_from_understanding(
    state: ConversationState,
    understanding: ConversationUnderstanding,
    message: str,
    extracted: ExtractedIntent,
    user_email: str | None,
) -> ConversationState:
    if understanding.slot_confidence >= 0.65:
        slots = understanding.slots
        if slots.passengers is not None:
            state.passengers = slots.passengers
            state.family_size = slots.passengers
        if slots.budget is not None:
            state.budget = slots.budget
        if slots.budget_unconstrained:
            state.budget_unconstrained = True
        if slots.use_case is not None:
            state.use_case = slots.use_case
        if slots.city_vs_highway is not None:
            state.city_vs_highway = slots.city_vs_highway
        if slots.comfort_vs_efficiency is not None:
            state.comfort_vs_efficiency = slots.comfort_vs_efficiency
        if slots.body_type is not None:
            state.body_type = slots.body_type
        if slots.fuel_preference is not None:
            state.fuel_preference = slots.fuel_preference
        if slots.comfort_vs_efficiency == "efficiency":
            state.comfort_vs_efficiency = "efficiency"
            state.space_priority = "fuel"

    if state.use_case and "family" in state.use_case.lower() and state.passengers == 1:
        state.passengers = 4
        state.family_size = 4

    if state.passengers is not None and state.body_type is None:
        if state.passengers >= 4:
            state.body_type = "suv"

    if extracted.price_max and state.budget is None and not state.budget_unconstrained:
        state.budget = extracted.price_max

    email = extract_email(message, user_email)
    if email:
        state.contact_email = email

    vehicle_id = extract_vehicle_id(message) or extracted.vehicle_id
    if vehicle_id is not None and vehicle_id not in state.shortlist_ids:
        state.shortlist_ids.append(vehicle_id)

    return state


def _handle_budget_objection(
    state: ConversationState,
    message: str,
    extracted: ExtractedIntent,
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
    search_extracted = ExtractedIntent(
        intent=IntentKind.INVENTORY_SEARCH,
        price_max=cap,
    )
    query = message.strip() or "affordable practical value under budget"
    result = hybrid_search_inventory(query, state=state, extracted=search_extracted, limit=6)
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


def _handle_preference_refinement(
    state: ConversationState,
    message: str,
) -> SalesTurnResult | None:
    if not state.last_recommended_ids and not _matches_any(message, FUEL_SIGNALS):
        return None
    if not state.last_recommended_ids and not state.has_discovery_basics():
        return None
    if _matches_any(message, FUEL_SIGNALS):
        state.space_priority = "fuel"
        state.comfort_vs_efficiency = "efficiency"
        retrieval = hybrid_search_inventory(
            message.strip() or "efficient hybrid electric low fuel",
            state=state,
            limit=3,
        )
        vehicles = retrieval.vehicles
        if not vehicles:
            return None
        state.last_recommended_ids = [v.id for v in vehicles]
        state.shortlist_ids = list(dict.fromkeys(state.shortlist_ids + state.last_recommended_ids))
        state.last_refinement_key = None
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
            rag_mode=f"sales_dialogue+efficiency_refine+{retrieval.retrieval_mode}",
            show_vehicle_cards=True,
        )
    if not state.last_recommended_ids:
        return None
    if not (
        _matches_any(message, SPACE_SIGNALS)
        or state.space_priority == "space"
        or state.fuel_preference
    ):
        return None
    vehicles = _vehicles_from_ids(state.last_recommended_ids[:4])
    if not vehicles:
        return None
    from backend.inventory_retrieval import _efficiency_fit_score

    ranked = sorted(
        vehicles,
        key=lambda v: (
            family_fit_score(v, state),
            _efficiency_fit_score(v, state),
            -v.price,
        ),
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


def handle_sales_turn(
    message: str,
    extracted: ExtractedIntent,
    state: ConversationState,
    user_email: str | None,
    rag: PolicyRAGService,
) -> SalesTurnResult:
    state.bump_turn()
    state.add_history_turn("user", message)

    understanding = understand_conversation(message, state)
    logger.info(
        "conv_metrics session=%s turn=%d intent=%s lang=%s phase=%s slots_filled=%s",
        state.session_id,
        state.turn_count,
        understanding.conv_intent.value,
        understanding.language,
        state.phase.value,
        list(state.filled_slots().keys()),
    )

    _update_state_from_understanding(state, understanding, message, extracted, user_email)

    import re as _re
    _explicit_lang_switch = bool(
        _re.search(r"hebrew|בעברית|עברית|דבר עברית|תדבר עברית|answer in hebrew|speak hebrew", message.lower())
        or _re.search(r"\b(english|speak english|answer in english)\b|דבר אנגלית|תדבר אנגלית|באנגלית", message.lower())
    )
    language_just_switched = _explicit_lang_switch and (
        (understanding.language == "he" and state.language_preference != "he")
        or (understanding.language == "en" and state.language_preference == "he")
    )

    policy = decide_policy(state, understanding, message)

    lang = policy.language

    if language_just_switched:
        if state.language_preference == "he":
            reply = "מעולה, ממשיך בעברית. כדי לדייק אותך מהר, מה התקציב ומה הכי חשוב לך כרגע?"
        else:
            reply = "Great, I'll continue in English. What should I optimize first: budget, comfort, efficiency, or space?"
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY if not state.has_discovery_basics() else DialoguePhase.RECOMMENDING
        state.add_history_turn("assistant", reply, "language_switch")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+language_switch",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.HANDLE_GREETING:
        if state.turn_count == 1:
            reply = generate_welcome(state)
        else:
            reply = generate_greeting_response(state)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY
        state.add_history_turn("assistant", reply, "greeting")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+greeting",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.RESPOND_TO_SMALLTALK:
        reply = generate_smalltalk_response(state, message)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY if not state.last_recommended_ids else state.phase
        state.add_history_turn("assistant", reply, "smalltalk")
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

    if policy.action == PolicyAction.EXPLAIN_PRODUCT:
        reply = _lang(
            state,
            "Great question. I help you find a car through a short conversation, "
            "then I compare top fits and can reserve one when you're ready.",
            "שאלה מצוינת. אני עוזר לבחור רכב דרך שיחה קצרה, "
            "ואז משווה את ההתאמות הטובות ויכול לשמור רכב כשתרצה.",
        )
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.EXPLORATORY_GUIDANCE
        state.add_history_turn("assistant", reply, "explain_product")
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

    if policy.action == PolicyAction.EXPLAIN_CRITERIA:
        reply = generate_criteria_explanation(state)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.EXPLORATORY_GUIDANCE
        state.add_history_turn("assistant", reply, "explain_criteria")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+explain_criteria",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.REPAIR_TURN:
        if policy.question_hint == "misunderstood_slot":
            if state.last_asked_field == "passengers":
                state.budget = None
            if state.language_preference == "he":
                if state.last_asked_field == "passengers":
                    reply = "צודק — פירשתי את זה בטעות כתקציב. כמה אנשים בדרך כלל נוסעים איתך?"
                else:
                    reply = "צודק — לא הבנתי נכון. אפשר לנסח שוב בקצרה?"
            elif state.last_asked_field == "passengers":
                reply = "You're right — I misread that as a budget. How many people usually ride along?"
            else:
                reply = "You're right — I misread that. Could you say that again in a few words?"
        else:
            reply = generate_repair_turn_response(state)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.REPAIR_TURN
        state.last_refinement_key = None
        state.last_recommended_ids = []
        state.add_history_turn("assistant", reply, "repair_turn")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+repair_turn",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.CLARIFY_BUDGET:
        state.phase = DialoguePhase.CLARIFICATION
        if policy.question_hint == "too_low":
            reply = _lang(
                state,
                "Just to confirm — did you mean around $200k or perhaps $20k?",
                "רק לוודא — התכוונת לסביבות 200 אלף דולר או 20 אלף?",
            )
        else:
            reply = _lang(
                state,
                "That's quite a range! Want me to keep this within a practical car budget, like $40k–$90k?",
                "זה טווח עצום. שנכוון לטווח ריאלי לרכב, למשל 40–90 אלף דולר?",
            )
        reply = _guard_repetition_and_progress(state, reply)
        state.add_history_turn("assistant", reply, "clarify_budget")
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

    if policy.action == PolicyAction.HANDLE_EXPLORATORY:
        state.use_case = None
        state.body_type = None
        state.last_refinement_key = None
        state.last_recommended_ids = []
        reply = generate_exploratory_response(state, message)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY
        state.add_history_turn("assistant", reply, "exploratory")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+exploratory",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.HANDLE_TOPIC_SHIFT:
        state.last_refinement_key = None
        state.last_recommended_ids = []
        reply = generate_topic_shift_response(state, message)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY
        state.add_history_turn("assistant", reply, "topic_shift")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+topic_shift",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.HANDLE_PRICE_OBJECTION:
        budget_turn = _handle_budget_objection(state, message, extracted)
        if budget_turn is not None:
            state.add_history_turn("assistant", budget_turn.reply, "price_objection")
            return budget_turn

    if policy.action == PolicyAction.RESERVE_VEHICLE or _matches_any(message, RESERVE_SIGNALS):
        vehicle = _pick_vehicle_for_action(state, message)
        if vehicle is None:
            reply = generate_clarifying_question(
                state,
                "Which vehicle would you like to reserve? You can say e.g. reserve vehicle #16.",
            )
            state.phase = DialoguePhase.RESERVE
            state.add_history_turn("assistant", reply, "reserve_clarify")
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
            reply = (
                f"Vehicle #{vehicle.id} is Pending De-listing (pre-2022) and cannot be reserved. "
                "Want alternatives from our 2022+ inventory?"
            )
            state.add_history_turn("assistant", reply, "reserve_blocked")
            save_conversation_state(state)
            return SalesTurnResult(
                reply=reply,
                state=state,
                vehicles=[vehicle],
                intent=IntentKind.RESERVE_INTENT,
                phase=state.phase,
                rag_mode="sales_dialogue",
            )
        state.phase = DialoguePhase.RESERVE
        state.add_history_turn("assistant", f"Reserving #{vehicle.id}", "reserve")
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

    if policy.action == PolicyAction.PURCHASE_VEHICLE or _matches_any(message, PURCHASE_SIGNALS) or (
        state.phase == DialoguePhase.PURCHASE and state.contact_email
    ):
        vehicle = _pick_vehicle_for_action(state, message)
        if not state.contact_email and not extract_email(message, user_email):
            reply = generate_purchase_prompt(state, vehicle)
            state.phase = DialoguePhase.PURCHASE
            state.add_history_turn("assistant", reply, "purchase_prompt")
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
        state.add_history_turn("assistant", "Processing purchase", "purchase")
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

    if policy.action == PolicyAction.COMPARE_VEHICLES or _matches_any(message, COMPARE_SIGNALS) or state.phase == DialoguePhase.COMPARING:
        compare_ids = _resolve_compare_ids(state, message)
        vehicles = [v for vid in compare_ids if (v := get_vehicle_by_id(vid)) is not None]
        if len(vehicles) < 2:
            reply = generate_clarifying_question(
                state,
                "Which two or three vehicles should I compare? Mention ids like #16 and #22.",
            )
            state.phase = DialoguePhase.COMPARING
            state.add_history_turn("assistant", reply, "compare_clarify")
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
        state.add_history_turn("assistant", reply, "comparison")
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
        state.add_history_turn("assistant", refinement.reply, "preference_refine")
        return refinement

    if policy.action == PolicyAction.ASK_PASSENGERS:
        if state.turn_count == 1 and not state.filled_slots():
            reply = generate_welcome(state)
        else:
            reply = generate_ask_passengers(state)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY
        state.last_asked_field = "passengers"
        state.add_history_turn("assistant", reply, "ask_passengers")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+discovery",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.ASK_USE_CASE:
        reply = generate_ask_use_case(state)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY
        state.last_asked_field = "use_case"
        state.add_history_turn("assistant", reply, "ask_use_case")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+discovery",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.ASK_CITY_VS_HIGHWAY:
        reply = generate_ask_city_vs_highway(state)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY
        state.last_asked_field = "city_vs_highway"
        state.add_history_turn("assistant", reply, "ask_city_vs_highway")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+discovery",
            show_vehicle_cards=False,
        )

    if policy.action == PolicyAction.ASK_BUDGET:
        reply = generate_ask_budget(state)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY
        state.last_asked_field = "budget"
        state.add_history_turn("assistant", reply, "ask_budget")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+discovery",
            show_vehicle_cards=False,
        )

    if (
        state.turn_count < DIALOGUE_CONTRACT["min_discovery_turns_before_full_reco"]
        and not state.has_discovery_basics()
        and not _explicit_fast_options_request(message)
    ):
        if state.passengers is None:
            reply = generate_ask_passengers(state)
        elif not state.use_case and not state.body_type:
            reply = generate_ask_use_case(state)
        else:
            reply = generate_ask_budget(state)
        reply = _guard_repetition_and_progress(state, reply)
        state.phase = DialoguePhase.DISCOVERY
        state.add_history_turn("assistant", reply, "early_discovery")
        save_conversation_state(state)
        return SalesTurnResult(
            reply=reply,
            state=state,
            vehicles=[],
            intent=IntentKind.GENERAL_CHAT,
            phase=state.phase,
            rag_mode="sales_dialogue+discovery",
            show_vehicle_cards=False,
        )

    retrieval = hybrid_search_inventory(message, state=state, extracted=extracted, limit=3)
    vehicles = retrieval.vehicles
    state.last_recommended_ids = [v.id for v in vehicles]
    state.last_refinement_key = None
    state.shortlist_ids = list(dict.fromkeys(state.shortlist_ids + state.last_recommended_ids))
    state.phase = DialoguePhase.RECOMMENDING
    reply = generate_recommendations(state, vehicles)
    reply = _guard_repetition_and_progress(state, reply)
    state.add_history_turn("assistant", reply, "recommendation")
    save_conversation_state(state)
    search_explanation = {
        "applied_filters": retrieval.applied_filters,
        "excluded": [e.model_dump() for e in retrieval.excluded_vehicles],
    }
    logger.info(
        "conv_metrics session=%s phase=%s lang=%s turn=%d vehicles=%s filters=%s",
        state.session_id,
        state.phase.value,
        state.language_preference,
        state.turn_count,
        [v.id for v in vehicles],
        retrieval.applied_filters,
    )
    return SalesTurnResult(
        reply=reply,
        state=state,
        vehicles=vehicles,
        intent=IntentKind.INVENTORY_SEARCH,
        phase=state.phase,
        rag_mode=f"sales_dialogue+{retrieval.retrieval_mode}",
        search_explanation=search_explanation,
    )
