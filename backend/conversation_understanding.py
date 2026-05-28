from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from backend.gemini_service import generate_structured

if TYPE_CHECKING:
    from backend.conversation_state import ConversationState

UNDERSTANDING_SYSTEM = """You are the Response Arbitration Layer for a premium car dealership AI advisor.

Your job: classify what the user is ACTUALLY trying to do RIGHT NOW.
Do not default to "slot_answer" unless the message clearly answers a discovery question.

INTENT TAXONOMY:

greeting          : pure greeting ("hi", "hello", "שלום", "היי")
social_smalltalk  : off-topic social ("how are you?", "who are you?", "מה שלומך?", "מי אתה?")
product_explanation: asking what this service does ("what do you do?", "מה אתה עושה?", "מה זה?")
criteria_inquiry  : asking about car-buying dimensions, factors, or what the advisor checks
                    ("what criteria?", "what else do you check?", "what factors matter?",
                     "איזה קריטריונים יש?", "מה חשוב?", "איזה עוד דברים אתה בודק?",
                     "מה שוקלים?", "מה עוד אתה בודק?", "מה בודקים?")
slot_answer       : directly answering a discovery question (passengers, budget amount,
                    use case preference, body type, fuel type, city/highway)
exploratory_followup: wanting different/other options ("something else", "משהו אחר", "אחרות")
decision_guidance : asking for a recommendation ("help me decide", "מה עדיף?", "מה מתאים לי?")
objection_price   : current options too expensive ("too expensive", "יקר מדי", "זה יקר לי")
objection_size    : size is wrong ("too big", "too small", "גדול מדי")
topic_shift       : explicitly pivoting ("forget the family angle", "I changed my mind completely")
frustration       : explicit complaint ("this is useless", "terrible", "לא עוזר", "גרוע")
confusion         : confused about the process ("I don't understand", "מבולבל", "לא הבנתי")
clarification_request: asking to explain something said ("why that one?", "explain #16")
reservation_intent: wants to reserve/hold ("reserve", "hold", "שמור")
purchase_intent   : wants to buy ("buy", "purchase", "לקנות")
comparison_request: wants to compare ("compare", "vs", "השווה")
general_search    : searching for specific make/model/price

ABSOLUTE RULES:
1. "something else" / "משהו אחר" = exploratory_followup — never objection_price
2. "what else do you check/consider?" = criteria_inquiry — never general_search or slot_answer
3. Social/greeting messages = social_smalltalk or greeting — never slot_answer
4. "no limit" / "no budget" / "אין מגבלה" / "לא משנה" = slot_answer with budget_unconstrained=true
5. A standalone number when budget was last asked = slot_answer budget
6. A standalone number when passengers was last asked = slot_answer passengers
7. Ambiguous short number: use last_asked_field context to decide

LANGUAGE: "he" if message has Hebrew chars (unicode \\u0590-\\u05FF), "en" otherwise.

SLOTS — extract ONLY what is clearly in this message:
- passengers: "alone"/"לבד"→1, "couple"/"זוג"→2, "family of N"→N, explicit N people/riders
- budget: any numeric amount (use last_asked_field=budget as signal)
- budget_unconstrained: true for "no limit", "no budget", "אין מגבלה", "לא משנה המחיר"
- use_case: city→"city_driving", family→"family_trips", highway→"highway_travel", daily→"daily_commute"
- city_vs_highway: "city"/"עיר"→"city", "highway"/"כביש מהיר"→"highway"
- body_type: explicit suv/sedan/coupe/hatchback only
- fuel_preference: explicit electric/hybrid/gas/חשמלי/היברידי
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
    USER_CORRECTION = "user_correction"
    NEGATIVE_FEEDBACK = "negative_feedback"


class UnderstandingSlots(BaseModel):
    passengers: int | None = None
    budget: float | None = None
    budget_unconstrained: bool = False
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
    if re.search(r"\bsolo\b|\balone\b|\bjust me\b|\bonly me\b|\bלבד\b|\bרק אני\b", lower):
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
    m = re.search(r"\$\s*([\d,]+)", lower)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def _extract_standalone_budget(message: str) -> float | None:
    lower = message.lower().strip()
    if re.fullmatch(r"\$?\s*([\d,]+)\s*\$?", lower):
        raw = re.sub(r"[^\d]", "", lower)
        if raw:
            v = float(raw)
            if v >= 1000:
                return v
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
    if re.search(r"\bsuv\b", lower):
        return "suv"
    if re.search(r"\bsedan\b|\bסדאן\b", lower):
        return "sedan"
    if re.search(r"\bsports?\b|\bcoupe\b|\bספורט\b", lower):
        return "sports"
    if re.search(r"\bhatchback\b", lower):
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


def _is_budget_context(message: str, state: ConversationState | None) -> bool:
    if state and state.last_asked_field == "budget":
        return True
    return _extract_budget(message) is not None


def _contextual_numeric_answer(
    message: str,
    state: ConversationState | None,
) -> ConversationUnderstanding | None:
    if state is None or not state.last_asked_field:
        return None
    lower = message.lower().strip()
    if not re.fullmatch(r"[\d,.\$k\s]+", lower, flags=re.IGNORECASE):
        return None
    lang = _detect_language(message)
    last = state.last_asked_field
    if last == "passengers":
        p = _extract_passengers(message)
        if p is not None:
            return ConversationUnderstanding(
                conv_intent=ConvIntent.SLOT_ANSWER,
                language=lang,
                slots=UnderstandingSlots(passengers=p),
                slot_confidence=0.98,
            )
        raw = re.sub(r"[^\d]", "", lower)
        if raw and int(raw) > 9:
            return ConversationUnderstanding(
                conv_intent=ConvIntent.USER_CORRECTION,
                language=lang,
                slots=UnderstandingSlots(),
                slot_confidence=0.9,
            )
    if last == "budget":
        b = _extract_budget(message) or _extract_standalone_budget(message)
        if b is None and re.search(r"\d", lower):
            raw = re.sub(r"[^\d.]", "", lower)
            if raw:
                b = float(raw)
        if b is not None:
            return ConversationUnderstanding(
                conv_intent=ConvIntent.SLOT_ANSWER,
                language=lang,
                slots=UnderstandingSlots(budget=b),
                slot_confidence=0.98,
            )
    return None


def _extract_efficiency_priority(message: str) -> str | None:
    lower = message.lower()
    if re.search(
        r"צריכת דלק|חיסכון בדלק|יעילות|דלק|fuel efficien|efficiency|mpg|חסכוני",
        lower,
    ):
        return "fuel"
    return None


def _is_no_budget_constraint(lower: str) -> bool:
    return bool(
        re.search(
            r"no limit|no budget|unlimited|doesn.t matter|any budget|not limited|"
            r"don.t care about price|price doesn.t matter|"
            r"אין מגבלה|לא משנה המחיר|כל תקציב|ללא הגבלה|לא מוגבל|מחיר לא חשוב|לא אכפת מהמחיר",
            lower,
        )
    )


def _regex_understand(
    message: str,
    state: ConversationState | None = None,
) -> ConversationUnderstanding:
    lower = message.lower().strip()
    lang = _detect_language(message)
    empty_slots = UnderstandingSlots()

    contextual = _contextual_numeric_answer(message, state)
    if contextual is not None:
        return contextual

    if re.search(
        r"הקלדתי|פירשת|בטעות|טעות|לא התכוונתי|how did you get|i typed|i meant|"
        r"wrong number|not the budget|not my budget|זה לא התקציב|למה שאלת",
        lower,
    ):
        return ConversationUnderstanding(
            conv_intent=ConvIntent.USER_CORRECTION,
            language=lang,
            slots=empty_slots,
        )

    if re.search(
        r"לא טוב|לא מתאים|not good|doesn.t work|didn.t like|לא אהבתי|"
        r"לא מה שרציתי|this isn.t right",
        lower,
    ):
        return ConversationUnderstanding(
            conv_intent=ConvIntent.NEGATIVE_FEEDBACK,
            language=lang,
            slots=empty_slots,
        )

    if re.fullmatch(r"(hi+|hello+|hey+|שלום|היי+|הי+|yo|sup)\s*[!.]*", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.GREETING, language=lang, slots=empty_slots)

    if re.search(
        r"how are you|how r u|how r you|what.s your name|are you okay|you alright|"
        r"tell me about yourself|who are you|how old are you|how old are u|"
        r"מה שלומך|מי אתה|בן כמה|אתה בסדר|מה קורה איתך",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.SOCIAL_SMALLTALK, language=lang, slots=empty_slots)

    if re.search(
        r"what do you do|what is this|how does this work|what can you do|what are you|"
        r"מה אתה עושה|מה אתה|מה עושים כאן|איך זה עובד|מה המטרה שלך",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.PRODUCT_EXPLANATION, language=lang, slots=empty_slots)

    if re.search(
        r"what criteria|which criteria|what factors|what should i consider|"
        r"what else do you consider|what else do you check|what do you evaluate|"
        r"what considerations|what do you look at|what do you look for|"
        r"איזה קריטריונים|מה הקריטריונים|מה חשוב בבחירה|מה כדאי לשקול|"
        r"איזה פרמטרים|מה שוקלים|מה הגורמים|איזה עוד דברים|מה עוד בודק|"
        r"מה עוד חשוב|אילו שיקולים|מה עוד אתה בודק|מה בודקים|אילו פרמטרים|"
        r"מה שוקלים|מה נבדק|מה בוחנים",
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
        r"too expensive|too pricey|cheaper|lower price|over budget|can.t afford|"
        r"price.sensitive|price sensitive|budget.conscious|budget conscious|"
        r"watching my budget|tight budget|on a budget|within budget|keep.?it.?cheap|"
        r"יקר מדי|זה יקר|יקר לי|זול יותר|חורג מהתקציב|מחוץ לתקציב|"
        r"מחיר נמוך|במחיר נוח|חסכוני",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.OBJECTION_PRICE, language=lang, slots=empty_slots)

    if re.search(r"too big|too large|too small|גדול מדי|קטן מדי", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.OBJECTION_SIZE, language=lang, slots=empty_slots)

    if re.search(
        r"terrible|awful|useless|not helpful|this is bad|you.re not listening|"
        r"זוועה|גרוע|לא עוזר|מתסכל|לא שימושי",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.FRUSTRATION, language=lang, slots=empty_slots)

    if re.search(
        r"i don.t understand|confused|i.m lost|what.s going on|"
        r"לא מבין|מבולבל|לא הבנתי|איבדתי",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.CONFUSION, language=lang, slots=empty_slots)

    if re.search(
        r"something else|other options|different options|show me others|show something else|"
        r"samthing else|somthing else|samshing else|somthing|"
        r"bad idea|not this|not these|nope these|don.t like these|not what i wanted|"
        r"משהו אחר|אחרות|תראה לי אחרים|אופציות אחרות|רעיון רע|לא מתאים",
        lower,
    ):
        return ConversationUnderstanding(conv_intent=ConvIntent.EXPLORATORY_FOLLOWUP, language=lang, slots=empty_slots)

    if re.search(r"hebrew|עברית|דבר עברית|תדבר עברית", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.SLOT_ANSWER, language="he", slots=empty_slots)

    if re.search(r"english|speak english|answer in english|דבר אנגלית", lower):
        return ConversationUnderstanding(conv_intent=ConvIntent.SLOT_ANSWER, language="en", slots=empty_slots)

    if _is_no_budget_constraint(lower):
        slots = UnderstandingSlots(budget_unconstrained=True)
        return ConversationUnderstanding(
            conv_intent=ConvIntent.SLOT_ANSWER, language=lang, slots=slots, slot_confidence=0.95
        )

    if state is None or state.last_asked_field != "passengers":
        if re.fullmatch(r"\$?\s*(\d{2,3})\s*\$?", lower.strip()):
            v = float(re.sub(r"[^\d]", "", lower.strip()))
            if 10 <= v < 1000 and _is_budget_context(message, state):
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
    if _is_budget_context(message, state):
        slots.budget = _extract_budget(message)
        if slots.budget is None:
            slots.budget = _extract_standalone_budget(message)
    use_case, city_vs_hw = _extract_use_case(message)
    slots.use_case = use_case
    slots.city_vs_highway = city_vs_hw
    slots.body_type = _extract_body_type(message)
    slots.fuel_preference = _extract_fuel(message)
    eff = _extract_efficiency_priority(message)
    if eff:
        slots.comfort_vs_efficiency = "efficiency"

    has_any_slot = any([
        slots.passengers is not None,
        slots.budget is not None,
        slots.budget_unconstrained,
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
    last_asked = getattr(state, "last_asked_field", None)
    return (
        f"Recent conversation:\n{history_str}\n\n"
        f"Already known about customer: {filled or 'nothing yet'}\n"
        f"Last field asked by advisor: {last_asked or 'none'}\n\n"
        f"Classify this message: {message}"
    )


def understand_conversation(
    message: str,
    state: ConversationState,
) -> ConversationUnderstanding:
    regex_result = _regex_understand(message, state)

    immediate_return_intents = {
        ConvIntent.GREETING,
        ConvIntent.SOCIAL_SMALLTALK,
        ConvIntent.PRODUCT_EXPLANATION,
        ConvIntent.CRITERIA_INQUIRY,
        ConvIntent.USER_CORRECTION,
        ConvIntent.NEGATIVE_FEEDBACK,
        ConvIntent.RESERVATION_INTENT,
        ConvIntent.PURCHASE_INTENT,
        ConvIntent.COMPARISON_REQUEST,
        ConvIntent.OBJECTION_PRICE,
        ConvIntent.OBJECTION_SIZE,
        ConvIntent.FRUSTRATION,
        ConvIntent.CONFUSION,
        ConvIntent.EXPLORATORY_FOLLOWUP,
    }

    if regex_result.conv_intent in immediate_return_intents:
        return regex_result

    if regex_result.conv_intent == ConvIntent.SLOT_ANSWER and regex_result.slot_confidence >= 0.8:
        concrete_slots = any([
            regex_result.slots.passengers is not None,
            regex_result.slots.body_type is not None,
            regex_result.slots.fuel_preference is not None,
            regex_result.slots.use_case is not None,
            regex_result.slots.city_vs_highway is not None,
            regex_result.slots.budget_unconstrained,
        ])
        if concrete_slots:
            return regex_result

    last_asked = getattr(state, "last_asked_field", None)
    if last_asked == "budget" and regex_result.slots.budget is not None:
        return regex_result

    if last_asked == "passengers" and regex_result.slots.passengers is not None:
        return regex_result

    context = _build_context_for_gemini(message, state)
    gemini_result = generate_structured(UNDERSTANDING_SYSTEM, context, ConversationUnderstanding)
    if gemini_result is not None:
        return gemini_result

    return regex_result
