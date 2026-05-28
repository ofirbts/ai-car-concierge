from __future__ import annotations

import re
from pydantic import BaseModel

from backend.conversation_state import ConversationState


class DialogueAnalysis(BaseModel):
    intent_label: str
    objection: str | None = None
    frustration: bool = False
    smalltalk: bool = False
    topic_shift: bool = False
    clarification_request: bool = False
    language_switch_hebrew: bool = False
    language_switch_english: bool = False
    exploratory_user: bool = False
    buying_readiness: str = "medium"


def analyze_dialogue_turn(message: str, state: ConversationState) -> DialogueAnalysis:
    lower = message.lower().strip()
    frustration = bool(
        re.search(r"\b(bad idea|not good|terrible|awful|this is bad|you're not listening)\b|זוועה|לא טוב", lower)
    )
    smalltalk = bool(
        re.search(r"\b(how old are you|how old are u|who are you|tell me about yourself)\b|בן כמה|מי אתה", lower)
    )
    objection = None
    if re.search(r"\b(expensive|too expensive|cheaper|price sensitive|budget)\b|יקר|זול", lower):
        objection = "price"
    elif re.search(r"\b(too big|too small|size|space)\b|גדול|קטן|מרווח", lower):
        objection = "size"
    clarification_request = bool(
        re.search(r"\b(what do you mean|explain|why|clarify)\b|תסביר|למה", lower)
    )
    language_switch_hebrew = bool(
        re.search(r"\b(hebrew|עברית)\b", lower)
    )
    language_switch_english = bool(
        re.search(r"\b(english)\b", lower)
    )
    topic_shift = bool(
        re.search(r"\b(something else|another option|different direction|not family)\b|משהו אחר|בלי טיולים משפחתיים", lower)
    )
    exploratory_user = bool(
        re.search(r"\b(not sure|just exploring|still thinking)\b|מתלבט|רק בודק", lower)
    )
    buying_readiness = "high" if re.search(r"\b(reserve|hold|buy|purchase)\b|שמור|לקנות", lower) else "medium"
    if exploratory_user or smalltalk:
        buying_readiness = "low"

    if smalltalk:
        intent_label = "smalltalk"
    elif objection == "price":
        intent_label = "objection_price"
    elif topic_shift:
        intent_label = "topic_shift"
    elif clarification_request:
        intent_label = "clarification_request"
    elif frustration:
        intent_label = "frustration"
    elif buying_readiness == "high":
        intent_label = "reservation_intent"
    else:
        intent_label = "exploratory_user"

    return DialogueAnalysis(
        intent_label=intent_label,
        objection=objection,
        frustration=frustration,
        smalltalk=smalltalk,
        topic_shift=topic_shift,
        clarification_request=clarification_request,
        language_switch_hebrew=language_switch_hebrew,
        language_switch_english=language_switch_english,
        exploratory_user=exploratory_user,
        buying_readiness=buying_readiness,
    )
