from backend.config import ENV_FILE, bootstrap, get_settings, reset_settings_cache
from backend.gemini_service import reset_gemini_client
from backend.rag_service import get_policy_rag_service, reset_policy_rag_service

KEY_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")


def test_bootstrap_loads_dotenv(monkeypatch, tmp_path):
    for key in KEY_VARS:
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=AIza-test-key-from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr("backend.config.ENV_FILE", env_file)

    reset_settings_cache()
    reset_policy_rag_service()
    reset_gemini_client()
    bootstrap()

    assert get_settings().google_api_key == "AIza-test-key-from-dotenv"
    service = get_policy_rag_service()
    assert service.retrieval_mode == "gemini_embeddings"


def test_bootstrap_skips_missing_env_file(monkeypatch):
    for key in KEY_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-explicit-env")
    monkeypatch.setattr("backend.config.ENV_FILE", ENV_FILE.parent / "nonexistent.env")

    reset_settings_cache()
    bootstrap()

    assert get_settings().google_api_key == "AIza-explicit-env"
