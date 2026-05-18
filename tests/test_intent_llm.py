from unittest.mock import MagicMock, patch

from backend.intent import ExtractedIntent, IntentKind, classify_intent


@patch("backend.intent.get_settings")
@patch("backend.intent.OpenAI")
def test_classify_intent_uses_llm_when_configured(mock_openai_cls, mock_settings):
    mock_settings.return_value.openai_api_key = "sk-test"
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    parsed = ExtractedIntent(intent=IntentKind.POLICY_QUESTION)
    mock_client.beta.chat.completions.parse.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=parsed))]
    )

    result = classify_intent("what is your refund policy?")
    assert result.intent == IntentKind.POLICY_QUESTION
    mock_client.beta.chat.completions.parse.assert_called_once()


@patch("backend.intent.get_settings")
@patch("backend.intent.logger")
def test_classify_intent_falls_back_on_llm_error(mock_logger, mock_settings):
    mock_settings.return_value.openai_api_key = "sk-test"
    with patch("backend.intent.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.beta.chat.completions.parse.side_effect = RuntimeError("api down")

        result = classify_intent("refund policy")
        assert result.intent == IntentKind.POLICY_QUESTION
    mock_logger.warning.assert_called()
