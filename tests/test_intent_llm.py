from unittest.mock import MagicMock, patch

from backend.intent import ExtractedIntent, IntentKind, classify_intent


@patch("backend.intent.get_settings")
@patch("backend.intent.generate_structured")
def test_classify_intent_uses_gemini_when_configured(mock_generate, mock_settings):
    mock_settings.return_value.has_google_api.return_value = True
    mock_generate.return_value = ExtractedIntent(intent=IntentKind.POLICY_QUESTION)

    result = classify_intent("what is your refund policy?")
    assert result.intent == IntentKind.POLICY_QUESTION
    mock_generate.assert_called_once()


@patch("backend.intent.get_settings")
@patch("backend.intent.generate_structured")
@patch("backend.intent.logger")
def test_classify_intent_falls_back_when_gemini_returns_none(
    mock_logger, mock_generate, mock_settings
):
    mock_settings.return_value.has_google_api.return_value = True
    mock_generate.return_value = None

    result = classify_intent("refund policy")
    assert result.intent == IntentKind.POLICY_QUESTION
    mock_logger.warning.assert_called()
