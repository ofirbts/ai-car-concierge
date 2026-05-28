from __future__ import annotations

from pydantic import BaseModel

from backend.conversation_state import ConversationState
from backend.dialogue_analysis import DialogueAnalysis


class DialoguePolicyDecision(BaseModel):
    action: str
    question: str | None = None


def choose_dialogue_policy(state: ConversationState, analysis: DialogueAnalysis) -> DialoguePolicyDecision:
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
