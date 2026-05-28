from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from backend.gemini_service import generate_structured

if TYPE_CHECKING:
    from backend.conversation_state import ConversationState

UNDERSTANDING_SYSTEM = """You are a conversation intent classifier for a premium car dealership AI advisor.

Classify the user's message into exactly one conversational intent.

INTENT DEFINITIONS — read every rule before choosing:

greeting        : pure greeting with no other content ("hi", "hello", "hey", "שלום", "היי")
social_smalltalk: off-topic social chat ("how are you?", "what's your name?", "are you okay?", "מה שלומך?")
product_explanation: asking what this bot/service does ("what do you do?", "how does this work?", "מה אתה?")
criteria_inquiry: asking what car-buying criteria or factors exist ("what criteria matter?", "what should I consider?", "what factors are important?", "איזה קריטריונים יש?", "מה חשוב?", "איזה פרמטרים?")
slot_answer     : directly answering a question about passengers, budget, use case, body type, fuel, city/highway
exploratory_followup: wanting different/other options without objecting ("something else", "other options", "משהו אחר", "אחרת") — NOT an objection
decision_guidance: asking for help choosing ("what would you recommend?", "help me decide", "I don't know")
objection_price : saying current options cost too much ("too expensive", "cheaper please", "יקר מדי", "זה יקר")
objection_size  : saying size is wrong ("too big", "too small")
topic_shift     : explicitly pivoting to a completely different angle ("actually I want electric", "forget the family car, I'm buying alone")
frustration     : explicit complaint about the service ("this is useless", "terrible", "you're not helpful")
confusion       : confused about the process ("I don't understand", "I'm lost", "מבולבל")
clarification_request: asking to explain something just said ("why that one?", "what do you mean?", "explain #16")
reservation_intent: wants to reserve/hold a vehicle ("reserve", "hold", "book", "שמור")
purchase_intent : wants to buy ("buy", "purchase", "לקנות")
comparison_request: wants to compare vehicles ("compare", "vs", "which is better", "השווה")
general_search  : searching for specific make/model/price

CRITICAL RULES — these override everything:
1. "something else" / "משהו אחר" = exploratory_followup, NOT objection_price, NOT topic_shift
2. "how are you?" / "מה שלומך?" = social_smalltalk, never slot_answer
3. "what criteria?" / "איזה קריטריונים?" = criteria_inquiry, never general_search
4. Short single-topic answers = slot_answer (e.g. "alone", "לבד", "city", "עיר", "$50k")
5. Pure greetings with no question = greeting
6. frustration requires explicit complaint words; uncertainty is not frustration

For language: return "he" if message contains Hebrew characters, else "en".

For slots — ONLY extract what is EXPLICITLY stated in THIS message. Never infer or guess:
- passengers: "alone"/"solo"/"לבד" → 1; "couple"/"זוג" → 2; "family of N" → N; explicit number + people
- budget: explicit numeric amount with $ or "budget" or "תקציב"
- use_case: ONLY if user says city/urban/עיר (→"city_driving"), family/kids/משפחה (→"family_trips"), highway/long drives/כביש מהיר (→"highway_travel"), daily/work/יומיומי (→"daily_commute")
- city_vs_highway: ONLY if user explicitly says "city", "highway", "עיר", "כביש מהיר", "בעיר", "בין עירוני"
- body_type: ONLY if user says suv/sedan/coupe/hatchback explicitly
- fuel_preference: ONLY if user says electric/hybrid/gas/חשמלי/היברידי
"""


class ConvIntent(str, Enum):
    GREETING = "greeting"
    SOCIAL_SMALLTALK = "social_smalltalk"
    PRODUCT_EXPLANATION = "product_explanation"
    CRITERIA_INQUIRY = "criteria_inquiry"
    SLOT_ANSWER = "slot_answer"
    EXPLORATORY_FOLLOWUP = "exploratory_followup"
    DECISION_GUIDANCE = "decision_guidance"
    OBJECTION_PRICE = "objection_price"
    OBJECTION_SIZE = "objection_size"
    TOPIC_SHIFT = "topic_shift"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"
    CLARIFICATION_REQUEST = "clarification_request"
    RESERVATION_INTENT = "reservation_intent"
    PURCHASE_INTENT = "purchase_intent"
    COMPARISON_REQUEST = "comparison_request"
    GENERAL_SEARCH = "general_search"


class UnderstandingSlots(BaseModel):
    passengers: int | None = None
    budget: float | None = None
    use_case: str | None = None
    city_vs_highway: str | None = None
    comfort_vs_efficiency: str | None = None
    body_type: str | None = None
    fuel_preference: str | None = None


class ConversationUnderstanding(BaseModel):
    conv_intent: ConvIntent
    emotional_tone: str = "neutral"
    language: str = "en"
    slots: UnderstandingSlots = UnderstandingSlots()
    slot_confidence: float = 0.8


def _detect_language(message: str) -> str:
    return "he" if re.search(r"[\u0590-\u05FF]", message) else "en"


def _extract_passengers(message: str) -> int | None:
    lower = message.lower().strip()
    if re.search(r"\bsolo\b|\balone\b|\bjust me\b|\bonly me\b|\blbad\b|\bלבד\b|\bרק אני\b", lower):
        return 1
    if re.search(r"\btwo kids\b|\btwo children\b|\bשני ילדים\b", lower):
        return 4
    if re.search(r"\bone kid\b|\bone child\b|\bילד אחד\b", lower):
        return 3
    if re.search(r"\bcouple\b|\btwo people\b|\bזוג\b|\bשנינו\b", lower):
        return 2
    if re.search(r"\bme and my partner\b|\bmy partner and i\b|\bme and my wife\b|\bme and my husband\b", lower):
        return 2
    if re.search(r"\bbaby\b|\binfant\b|\bתינוק\b", lower):
        return 3
    m = re.search(r"\bfamily of (\d+)\b", lower)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\s*(people|passengers|riders|persons)\b", lower)
    if m:
        v = int(m.group(1))
        return v if 1 <= v <= 9 else None
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
    for word, val in word_map.items():
        if re.search(rf"\b{word}\s+(people|passengers|riders)\b", lower):
            return val
        if lower.strip() == word:
            return val
    if re.fullmatch(r"\d+", lower.strip()):
        v = int(lower.strip())
        return v if 1 <= v <= 9 else None
    return None


def _extract_budget(message: str) -> float | None:
    lower = message.lower()
    m = re.search(r"\$?\s*([\d,]+)\s*k\b", lower)
    if m:
        return float(m.group(1).replace(",", "")) * 1000
    m = re.search(
        r"(?:budget|around|about|roughly|up to|max|תקציב|סביב|עד)\s*(?:is|of|around|about)?\s*\$?\s*([\d,]+)",
        lower,
    )
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"\$?\s*([\d,]+)\s*(?:budget|max|total|דולר)", lower)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(r"(?:under|below|less than|מתחת ל)\s*\$?\s*([\d,]+)", lower)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _extract_use_case(message: str) -> tuple[str | None, str | None]:
    lower = message.lower()
    city_vs_hw: str | None = None
    use_case: str | None = None
    if re.search(r"\bcity\b|\burban\b|\bcommute\b|\bעיר\b|\bבעיר\b|\bעירוני\b|\bיומיומי\b", lower):
        use_case = "city_driving"
        city_vs_hw = "city"
    elif re.search(r"\bhighway\b|\blong drive\b|\blong trip\b|\bכביש מהיר\b|\bבין עירוני\b|\bנסיעות ארוכות\b", lower):
        use_case = "highway_travel"
        city_vs_hw = "highway"
    elif re.search(r"\bfamily\b|\bkids\b|\bchildren\b|\bמשפחה\b|\bילדים\b", lower):
        use_case = "family_trips"
    elif re.search(r"\bwork\b|\bdaily\b|\bעבודה\b", lower):
        use_case = "daily_commute"
    return use_case, city_vs_hw


def _extract_body_type(message: str) -> str | None:
    lower = message.lower()
    if re.search(r"\bsuv\b|\bsuv\b", lower):
        return "suv"
    if re.search(r"\bsedan\b|\bסדאן\b", lower):
        return "sedan"
    if re.search(r"\bsports?\b|\bcoupe\b|\bספורט\b", lower):
        return "sports"
    if re.search(r"\bhatchback\b|\bהאצ.בק\b", lower):
        return "hatchback"
    return None


def _extract_fuel(message: str) -> str | None:
    lower = message.lower()
    if re.search(r"\belectric\b|\bev\b|\bחשמלי\b", lower):
        return "Electric"
    if re.search(r"\bplug.in\b|\bphev\b", lower):
        return "Plug-in Hybrid"
    if re.search(r"\bhybrid\b|\bהיברידי\b", lower):
        return "Hybrid"
    if re.search(r"\bgas\b|\bgasoline\b|\bבנזין\b", lower):
        return "Gasoline"
    return None


def _regex_understand(message: str) -> ConversationUnderstanding:
    lower = message.lower().strip()
    lang = _detect_language(message)
    empty_slots = UnderstandingSlots()

    if re.fullmatch(r"(hi+|hello+|hey+|שלום|היי+|הי+|yo|sup)\s*[!.]*", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.GREETING, language=lang, slots=empty_slots)

    if re.search(
        r"\b(how are you|how r u|how r you|what.s your name|are you okay|you alright|"
        r"tell me about yourself|who are you|how old are you|how old are u|"
        r"מה שלומך|מי אתה|בן כמה|אתה בסדר|מה קורה איתך)\b",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.SOCIAL_SMALLTALK, language=lang, slots=empty_slots)

    if re.search(
        r"\b(what do you do|what is this|how does this work|what can you do|what are you|"
        r"מה אתה|מה עושים כאן|איך זה עובד|מה המטרה שלך)\b",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.PRODUCT_EXPLANATION, language=lang, slots=empty_slots)

    if re.search(
        r"(what criteria|which criteria|what factors|what should i consider|what matters when|"
        r"what are the criteria|איזה קריטריונים|מה הקריטריונים|מה חשוב בבחירה|מה כדאי לשקול|"
        r"איזה פרמטרים|מה שוקלים|מה הגורמים)",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.CRITERIA_INQUIRY, language=lang, slots=empty_slots)

    if re.search(r"\b(reserve|hold|book|שמור|להזמין)\b", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.RESERVATION_INTENT, language=lang, slots=empty_slots)

    if re.search(r"\b(buy|purchase|order|לקנות|לרכוש|אקנה)\b", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.PURCHASE_INTENT, language=lang, slots=empty_slots)

    if re.search(r"\b(compare|vs\.?|versus|which is better|השווה|מה עדיף|השוואה)\b", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.COMPARISON_REQUEST, language=lang, slots=empty_slots)

    if re.search(
        r"\b(too expensive|too pricey|cheaper|lower price|over budget|can.t afford|"
        r"יקר מדי|זה יקר|יקר לי|זול יותר|חורג מהתקציב|מחוץ לתקציב)\b",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.OBJECTION_PRICE, language=lang, slots=empty_slots)

    if re.search(r"\b(too big|too large|too small|גדול מדי|קטן מדי)\b", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.OBJECTION_SIZE, language=lang, slots=empty_slots)

    if re.search(
        r"\b(terrible|awful|useless|not helpful|this is bad|you.re not listening|"
        r"זוועה|גרוע|לא עוזר|מתסכל|לא שימושי)\b",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.FRUSTRATION, language=lang, slots=empty_slots)

    if re.search(
        r"\b(i don.t understand|confused|i.m lost|what.s going on|"
        r"לא מבין|מבולבל|לא הבנתי|איבדתי)\b",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.CONFUSION, language=lang, slots=empty_slots)

    if re.search(
        r"\b(something else|other options|different options|show me others|show something else|"
        r"samthing else|somthing else|"
        r"משהו אחר|אחרות|תראה לי אחרים|אופציות אחרות)\b",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.EXPLORATORY_FOLLOWUP, language=lang, slots=empty_slots)

    if re.search(r"\b(hebrew|עברית|דבר עברית|תדבר עברית)\b", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.SLOT_ANSWER, language="he", slots=empty_slots)

    if re.search(r"\b(english|speak english|answer in english|דבר אנגלית)\b", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.SLOT_ANSWER, language="en", slots=empty_slots)

    if re.fullmatch(r"\$?\s*(\d{2,3})\s*\$?", lower.strip()):
        v = float(re.sub(r"[^\d]", "", lower.strip()))
        if 10 <= v < 1000:
            slots = UnderstandingSlots(budget=v)
            return ConversationUnderstanding(
                conv_intent=ConvIntent.SLOT_ANSWER, language=lang, slots=slots, slot_confidence=0.9
            )

    if re.search(r"^\s*(no|nah|nope|not this|לא|לא זה)\s*$", lower.strip()):
        return ConversationUnderstanding(conv_intent=ConvIntent.EXPLORATORY_FOLLOWUP, language=lang, slots=empty_slots)

    if re.search(
        r"\b(i don.t know|not sure|undecided|help me choose|help me decide|what.s best|"
        r"לא יודע|לא בטוח|עזור לי לבחור|מה הכי מתאים)\b",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.DECISION_GUIDANCE, language=lang, slots=empty_slots)

    slots = UnderstandingSlots()
    slots.passengers = _extract_passengers(message)
    slots.budget = _extract_budget(message)
    use_case, city_vs_hw = _extract_use_case(message)
    slots.use_case = use_case
    slots.city_vs_highway = city_vs_hw
    slots.body_type = _extract_body_type(message)
    slots.fuel_preference = _extract_fuel(message)

    has_any_slot = any([
        slots.passengers is not None,
        slots.budget is not None,
        slots.use_case is not None,
        slots.city_vs_highway is not None,
        slots.body_type is not None,
        slots.fuel_preference is not None,
    ])
    if has_any_slot:
        return ConversationUnderstanding(
            conv_intent=ConvIntent.SLOT_ANSWER,
            language=lang,
            slots=slots,
            slot_confidence=0.85,
        )

    return ConversationUnderstanding(
        conv_intent=ConvIntent.SLOT_ANSWER,
        language=lang,
        slots=slots,
        slot_confidence=0.3,
    )


def _build_context_for_gemini(message: str, state: ConversationState) -> str:
    history_lines: list[str] = []
    for turn in (state.conversation_history or [])[-4:]:
        role = turn.get("role", "?")
        text = turn.get("message", "")
        history_lines.append(f"{role}: {text}")
    history_str = "\n".join(history_lines) if history_lines else "(start of conversation)"
    filled = {
        k: v for k, v in {
            "passengers": state.passengers,
            "budget": state.budget,
            "use_case": state.use_case,
            "city_vs_highway": state.city_vs_highway,
            "body_type": state.body_type,
        }.items() if v is not None
    }
    return (
        f"Recent conversation:\n{history_str}\n\n"
        f"Already known about customer: {filled or 'nothing yet'}\n\n"
        f"Classify this message: {message}"
    )


def understand_conversation(
    message: str,
    state: ConversationState,
) -> ConversationUnderstanding:
    regex_result = _regex_understand(message)

    high_confidence_intents = {
        ConvIntent.GREETING,
        ConvIntent.SOCIAL_SMALLTALK,
        ConvIntent.PRODUCT_EXPLANATION,
        ConvIntent.CRITERIA_INQUIRY,
        ConvIntent.RESERVATION_INTENT,
        ConvIntent.PURCHASE_INTENT,
        ConvIntent.COMPARISON_REQUEST,
        ConvIntent.OBJECTION_PRICE,
        ConvIntent.OBJECTION_SIZE,
        ConvIntent.FRUSTRATION,
        ConvIntent.CONFUSION,
        ConvIntent.EXPLORATORY_FOLLOWUP,
    }

    if regex_result.conv_intent in high_confidence_intents:
        return regex_result

    context = _build_context_for_gemini(message, state)
    gemini_result = generate_structured(UNDERSTANDING_SYSTEM, context, ConversationUnderstanding)
    if gemini_result is not None:
        return gemini_result

    return regex_result
