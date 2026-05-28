from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from backend.conversation_state import ConversationState
from backend.conversation_understanding import ConvIntent, ConversationUnderstanding


class PolicyAction(str, Enum):
    HANDLE_GREETING = "handle_greeting"
    RESPOND_TO_SMALLTALK = "respond_to_smalltalk"
    EXPLAIN_PRODUCT = "explain_product"
    EXPLAIN_CRITERIA = "explain_criteria"
    CLARIFY_BUDGET = "clarify_budget"
    ASK_PASSENGERS = "ask_passengers"
    ASK_USE_CASE = "ask_use_case"
    ASK_CITY_VS_HIGHWAY = "ask_city_vs_highway"
    ASK_BUDGET = "ask_budget"
    RECOMMEND_VEHICLES = "recommend_vehicles"
    COMPARE_VEHICLES = "compare_vehicles"
    HANDLE_PRICE_OBJECTION = "handle_price_objection"
    HANDLE_EXPLORATORY = "handle_exploratory"
    HANDLE_TOPIC_SHIFT = "handle_topic_shift"
    REPAIR_TURN = "repair_turn"
    RESERVE_VEHICLE = "reserve_vehicle"
    PURCHASE_VEHICLE = "purchase_vehicle"
    SWITCH_LANGUAGE = "switch_language"
    PROGRESS_CONVERSATION = "progress_conversation"


class PolicyDecision(BaseModel):
    action: PolicyAction
    language: str = "en"
    question_hint: str | None = None
    reasoning: str = ""


def _has_enough_context_to_recommend(state: ConversationState) -> bool:
    has_passengers = state.passengers is not None or state.family_size is not None
    has_budget = state.budget is not None
    has_preference = bool(
        state.use_case
        or state.body_type
        or state.city_vs_highway
        or state.space_priority
    )
    return has_passengers and has_budget and has_preference


def decide_policy(
    state: ConversationState,
    understanding: ConversationUnderstanding,
) -> PolicyDecision:
    lang = understanding.language if understanding.language else state.language_preference

    if understanding.language == "he" and state.language_preference != "he":
        state.language_preference = "he"
        lang = "he"
    elif understanding.language == "en" and state.language_preference == "he":
        state.language_preference = "en"
        lang = "en"

    intent = understanding.conv_intent

    if intent == ConvIntent.GREETING:
        return PolicyDecision(
            action=PolicyAction.HANDLE_GREETING,
            language=lang,
            reasoning="User greeted, respond warmly and open discovery",
        )

    if intent == ConvIntent.SOCIAL_SMALLTALK:
        return PolicyDecision(
            action=PolicyAction.RESPOND_TO_SMALLTALK,
            language=lang,
            reasoning="Off-topic social message, respond naturally then gently redirect",
        )

    if intent in (ConvIntent.PRODUCT_EXPLANATION, ConvIntent.CONFUSION):
        return PolicyDecision(
            action=PolicyAction.EXPLAIN_PRODUCT,
            language=lang,
            reasoning="User is confused or asking what this service does",
        )

    if intent == ConvIntent.CRITERIA_INQUIRY:
        return PolicyDecision(
            action=PolicyAction.EXPLAIN_CRITERIA,
            language=lang,
            reasoning="User is asking about buying criteria — explain dimensions, do NOT recommend yet",
        )

    if intent == ConvIntent.FRUSTRATION:
        state.frustration_level = min(state.frustration_level + 1, 3)
        return PolicyDecision(
            action=PolicyAction.REPAIR_TURN,
            language=lang,
            reasoning="User expressed frustration, acknowledge and recalibrate",
        )

    if understanding.slots.budget is not None:
        b = understanding.slots.budget
        if b < 5000:
            return PolicyDecision(
                action=PolicyAction.CLARIFY_BUDGET,
                language=lang,
                question_hint="too_low",
                reasoning="Budget too low to be a car price",
            )
        if b > 1_500_000:
            return PolicyDecision(
                action=PolicyAction.CLARIFY_BUDGET,
                language=lang,
                question_hint="too_high",
                reasoning="Budget unrealistically high",
            )

    if intent == ConvIntent.RESERVATION_INTENT:
        return PolicyDecision(
            action=PolicyAction.RESERVE_VEHICLE,
            language=lang,
            reasoning="User wants to reserve a vehicle",
        )

    if intent == ConvIntent.PURCHASE_INTENT:
        return PolicyDecision(
            action=PolicyAction.PURCHASE_VEHICLE,
            language=lang,
            reasoning="User wants to purchase",
        )

    if intent == ConvIntent.COMPARISON_REQUEST:
        return PolicyDecision(
            action=PolicyAction.COMPARE_VEHICLES,
            language=lang,
            reasoning="User wants to compare vehicles",
        )

    if intent == ConvIntent.EXPLORATORY_FOLLOWUP:
        return PolicyDecision(
            action=PolicyAction.HANDLE_EXPLORATORY,
            language=lang,
            reasoning="User wants to explore other options, not an objection",
        )

    if intent == ConvIntent.TOPIC_SHIFT:
        return PolicyDecision(
            action=PolicyAction.HANDLE_TOPIC_SHIFT,
            language=lang,
            reasoning="User shifted topic, recalibrate without treating as objection",
        )

    if intent == ConvIntent.OBJECTION_PRICE:
        return PolicyDecision(
            action=PolicyAction.HANDLE_PRICE_OBJECTION,
            language=lang,
            reasoning="User says options are too expensive",
        )

    if intent == ConvIntent.DECISION_GUIDANCE:
        if _has_enough_context_to_recommend(state):
            return PolicyDecision(
                action=PolicyAction.RECOMMEND_VEHICLES,
                language=lang,
                reasoning="User asking for guidance, we have enough context to recommend",
            )
        return PolicyDecision(
            action=PolicyAction.ASK_USE_CASE,
            language=lang,
            reasoning="User wants guidance but we need more context first",
        )

    if state.passengers is None and state.family_size is None:
        return PolicyDecision(
            action=PolicyAction.ASK_PASSENGERS,
            language=lang,
            reasoning="Passenger count is unknown",
        )

    if not state.use_case and not state.body_type and not state.city_vs_highway:
        return PolicyDecision(
            action=PolicyAction.ASK_USE_CASE,
            language=lang,
            reasoning="No use case or body type preference known",
        )

    if state.city_vs_highway is None and not _city_known_from_use_case(state):
        return PolicyDecision(
            action=PolicyAction.ASK_CITY_VS_HIGHWAY,
            language=lang,
            reasoning="City vs highway preference unknown",
        )

    if state.budget is None:
        return PolicyDecision(
            action=PolicyAction.ASK_BUDGET,
            language=lang,
            reasoning="Budget is unknown",
        )

    if _has_enough_context_to_recommend(state):
        return PolicyDecision(
            action=PolicyAction.RECOMMEND_VEHICLES,
            language=lang,
            reasoning="Sufficient discovery complete, recommend",
        )

    return PolicyDecision(
        action=PolicyAction.PROGRESS_CONVERSATION,
        language=lang,
        reasoning="Continuing conversation naturally",
    )


def _city_known_from_use_case(state: ConversationState) -> bool:
    if not state.use_case:
        return False
    use = state.use_case.lower()
    return "city" in use or "commute" in use or "highway" in use or "trip" in use


class DialoguePolicyDecision(BaseModel):
    action: str
    question: str | None = None


def choose_dialogue_policy(
    state: ConversationState,
    analysis: object,
) -> DialoguePolicyDecision:
    from backend.dialogue_analysis import DialogueAnalysis

    if not isinstance(analysis, DialogueAnalysis):
        return DialoguePolicyDecision(action="normal_progress")

    if analysis.language_switch_hebrew:
        return DialoguePolicyDecision(action="switch_language_he")
    if analysis.language_switch_english:
        return DialoguePolicyDecision(action="switch_language_en")
    if analysis.invalid_budget == "too_low":
        return DialoguePolicyDecision(action="clarify_budget_low")
    if analysis.invalid_budget == "too_high":
        return DialoguePolicyDecision(action="clarify_budget_high")
    if analysis.confused_user:
        return DialoguePolicyDecision(action="explain_product")
    if analysis.playful_input:
        return DialoguePolicyDecision(action="playful_response")
    if analysis.smalltalk:
        return DialoguePolicyDecision(action="smalltalk_repair")
    if analysis.frustration:
        return DialoguePolicyDecision(
            action="repair_turn",
            question="What felt off — price, size, efficiency, or something else?",
        )
    if analysis.topic_shift:
        return DialoguePolicyDecision(
            action="topic_shift_recalibrate",
            question="Got it. What should I optimize now — price, comfort, efficiency, or size?",
        )
    if analysis.objection == "price":
        return DialoguePolicyDecision(action="handle_price_objection")
    if not state.has_discovery_basics():
        return DialoguePolicyDecision(action="continue_discovery")
    return DialoguePolicyDecision(action="normal_progress")
