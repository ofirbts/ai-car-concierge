from __future__ import annotations

import pytest

from backend.conversation_state import ConversationState, new_session_id
from backend.conversation_understanding import (
    ConvIntent,
    ConversationUnderstanding,
    _regex_understand,
    understand_conversation,
)


def _state(lang: str = "en") -> ConversationState:
    s = ConversationState(session_id=new_session_id())
    s.language_preference = lang
    return s


class TestRegexUnderstand:
    def test_social_smalltalk_how_are_you(self) -> None:
        result = _regex_understand("how are you?")
        assert result.conv_intent == ConvIntent.SOCIAL_SMALLTALK

    def test_social_smalltalk_are_you_okay(self) -> None:
        result = _regex_understand("are you okay?")
        assert result.conv_intent == ConvIntent.SOCIAL_SMALLTALK

    def test_greeting_hi(self) -> None:
        result = _regex_understand("hi")
        assert result.conv_intent == ConvIntent.GREETING

    def test_greeting_hello(self) -> None:
        result = _regex_understand("hello!")
        assert result.conv_intent == ConvIntent.GREETING

    def test_greeting_hebrew(self) -> None:
        result = _regex_understand("היי")
        assert result.conv_intent == ConvIntent.GREETING
        assert result.language == "he"

    def test_criteria_inquiry_english(self) -> None:
        result = _regex_understand("what criteria matter when choosing a car?")
        assert result.conv_intent == ConvIntent.CRITERIA_INQUIRY

    def test_criteria_inquiry_hebrew(self) -> None:
        result = _regex_understand("איזה קריטריונים יש?")
        assert result.conv_intent == ConvIntent.CRITERIA_INQUIRY
        assert result.language == "he"

    def test_exploratory_something_else(self) -> None:
        result = _regex_understand("something else")
        assert result.conv_intent == ConvIntent.EXPLORATORY_FOLLOWUP

    def test_exploratory_hebrew(self) -> None:
        result = _regex_understand("משהו אחר")
        assert result.conv_intent == ConvIntent.EXPLORATORY_FOLLOWUP

    def test_objection_price_english(self) -> None:
        result = _regex_understand("that's too expensive for me")
        assert result.conv_intent == ConvIntent.OBJECTION_PRICE

    def test_objection_price_hebrew(self) -> None:
        result = _regex_understand("יקר מדי")
        assert result.conv_intent == ConvIntent.OBJECTION_PRICE

    def test_slot_answer_alone(self) -> None:
        result = _regex_understand("alone")
        assert result.conv_intent == ConvIntent.SLOT_ANSWER
        assert result.slots.passengers == 1

    def test_slot_answer_levad(self) -> None:
        result = _regex_understand("לבד")
        assert result.conv_intent == ConvIntent.SLOT_ANSWER
        assert result.slots.passengers == 1
        assert result.language == "he"

    def test_slot_answer_family_of_4(self) -> None:
        result = _regex_understand("family of 4")
        assert result.conv_intent == ConvIntent.SLOT_ANSWER
        assert result.slots.passengers == 4

    def test_slot_answer_city(self) -> None:
        result = _regex_understand("city driving")
        assert result.conv_intent == ConvIntent.SLOT_ANSWER
        assert result.slots.use_case == "city_driving"
        assert result.slots.city_vs_highway == "city"

    def test_slot_answer_budget(self) -> None:
        result = _regex_understand("my budget is $50k")
        assert result.conv_intent == ConvIntent.SLOT_ANSWER
        assert result.slots.budget == 50_000

    def test_slot_answer_suv(self) -> None:
        result = _regex_understand("I want an SUV")
        assert result.conv_intent == ConvIntent.SLOT_ANSWER
        assert result.slots.body_type == "suv"

    def test_product_explanation(self) -> None:
        result = _regex_understand("what do you do here?")
        assert result.conv_intent == ConvIntent.PRODUCT_EXPLANATION

    def test_product_explanation_hebrew(self) -> None:
        result = _regex_understand("מה אתה?")
        assert result.conv_intent == ConvIntent.PRODUCT_EXPLANATION

    def test_reservation_intent(self) -> None:
        result = _regex_understand("I want to reserve vehicle #16")
        assert result.conv_intent == ConvIntent.RESERVATION_INTENT

    def test_comparison_request(self) -> None:
        result = _regex_understand("compare #16 vs #22")
        assert result.conv_intent == ConvIntent.COMPARISON_REQUEST

    def test_frustration(self) -> None:
        result = _regex_understand("this is terrible, you're not helpful")
        assert result.conv_intent == ConvIntent.FRUSTRATION

    def test_confusion(self) -> None:
        result = _regex_understand("I don't understand what's going on")
        assert result.conv_intent == ConvIntent.CONFUSION

    def test_language_detection_hebrew(self) -> None:
        result = _regex_understand("אני רוצה רכב")
        assert result.language == "he"

    def test_language_detection_english(self) -> None:
        result = _regex_understand("I want a car")
        assert result.language == "en"

    def test_meshahu_acher_is_exploratory_not_objection(self) -> None:
        result = _regex_understand("משהו אחר")
        assert result.conv_intent == ConvIntent.EXPLORATORY_FOLLOWUP
        assert result.conv_intent != ConvIntent.OBJECTION_PRICE

    def test_criteria_inquiry_not_general_search(self) -> None:
        result = _regex_understand("מה הקריטריונים לבחירת רכב?")
        assert result.conv_intent == ConvIntent.CRITERIA_INQUIRY
        assert result.conv_intent != ConvIntent.GENERAL_SEARCH

    def test_slot_fuel_electric(self) -> None:
        result = _regex_understand("I prefer electric")
        assert result.slots.fuel_preference == "Electric"

    def test_slot_fuel_hybrid_hebrew(self) -> None:
        result = _regex_understand("אני רוצה היברידי")
        assert result.slots.fuel_preference == "Hybrid"

    def test_no_hallucinated_highway_from_alone(self) -> None:
        result = _regex_understand("לבד")
        assert result.slots.passengers == 1
        assert result.slots.use_case is None
        assert result.slots.city_vs_highway is None

    def test_decision_guidance(self) -> None:
        result = _regex_understand("I don't know what to choose, help me decide")
        assert result.conv_intent == ConvIntent.DECISION_GUIDANCE


class TestUnderstandConversation:
    def test_social_smalltalk_bypasses_gemini(self) -> None:
        state = _state()
        result = understand_conversation("how are you?", state)
        assert result.conv_intent == ConvIntent.SOCIAL_SMALLTALK

    def test_criteria_inquiry_bypasses_gemini(self) -> None:
        state = _state()
        result = understand_conversation("what criteria matter?", state)
        assert result.conv_intent == ConvIntent.CRITERIA_INQUIRY

    def test_exploratory_bypasses_gemini(self) -> None:
        state = _state()
        result = understand_conversation("something else", state)
        assert result.conv_intent == ConvIntent.EXPLORATORY_FOLLOWUP

    def test_greeting_bypasses_gemini(self) -> None:
        state = _state()
        result = understand_conversation("hi", state)
        assert result.conv_intent == ConvIntent.GREETING

    def test_levad_gives_passengers_1(self) -> None:
        state = _state()
        state.language_preference = "he"
        result = understand_conversation("לבד", state)
        assert result.conv_intent == ConvIntent.SLOT_ANSWER
        assert result.slots.passengers == 1
        assert result.slots.city_vs_highway is None

    def test_slot_answer_returns_language(self) -> None:
        state = _state()
        result = understand_conversation("עיר", state)
        assert result.language == "he"
