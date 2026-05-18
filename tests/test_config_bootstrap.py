from pathlib import Path

from backend.config import bootstrap, get_settings, reset_settings_cache
from backend.rag_service import get_policy_rag_service, reset_policy_rag_service


def test_bootstrap_loads_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=AIza-test-key-from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    reset_settings_cache()
    reset_policy_rag_service()
    bootstrap()

    assert get_settings().google_api_key == "AIza-test-key-from-dotenv"
    service = get_policy_rag_service()
    assert service.retrieval_mode == "gemini_embeddings"
