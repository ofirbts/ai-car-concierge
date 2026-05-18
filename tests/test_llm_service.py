from unittest.mock import MagicMock, patch

from backend.llm_service import synthesize_reply


@patch("backend.llm_service.get_openai_client")
def test_synthesize_reply_returns_model_text(mock_client_fn):
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello from the concierge."))]
    )
    with patch("backend.llm_service.get_settings") as mock_settings:
        mock_settings.return_value.use_quality_llm = False
        result = synthesize_reply("Hi", "context block")
    assert result == "Hello from the concierge."


@patch("backend.llm_service.get_openai_client")
def test_synthesize_reply_none_without_client(mock_client_fn):
    mock_client_fn.return_value = None
    assert synthesize_reply("Hi", "ctx") is None
