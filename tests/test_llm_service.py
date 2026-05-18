from unittest.mock import patch

from backend.llm_service import synthesize_reply


@patch("backend.llm_service.generate_text")
def test_synthesize_reply_returns_model_text(mock_generate):
    mock_generate.return_value = "Hello from the concierge."
    result = synthesize_reply("Hi", "context block")
    assert result == "Hello from the concierge."


@patch("backend.llm_service.generate_text")
def test_synthesize_reply_none_when_gemini_unavailable(mock_generate):
    mock_generate.return_value = None
    assert synthesize_reply("Hi", "ctx") is None
