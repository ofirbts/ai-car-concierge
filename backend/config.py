from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_chat_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_CHAT_MODEL")
    gemini_chat_model_quality: str = Field(
        default="gemini-2.5-pro", validation_alias="GEMINI_CHAT_MODEL_QUALITY"
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001", validation_alias="GEMINI_EMBEDDING_MODEL"
    )
    resend_api_key: str = ""
    resend_from_email: str = "concierge@yourdomain.com"
    resend_to_email: str = "sales@yourdomain.com"
    backend_url: str = "http://127.0.0.1:8000"
    show_debug_meta: bool = False
    use_quality_llm: bool = False
    api_key: str = Field(default="", validation_alias="API_KEY")
    chat_rate_limit: str = Field(default="30/minute", validation_alias="CHAT_RATE_LIMIT")
    cors_origins: str = Field(default="*", validation_alias="CORS_ORIGINS")
    validation_profile: str = Field(default="normal", validation_alias="VALIDATION_PROFILE")

    def has_google_api(self) -> bool:
        return bool(self.google_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


def bootstrap() -> Settings:
    from dotenv import load_dotenv

    if ENV_FILE.is_file():
        load_dotenv(ENV_FILE, override=False)
    reset_settings_cache()
    from backend.gemini_service import reset_gemini_client
    from backend.rag_service import reset_policy_rag_service

    reset_gemini_client()
    reset_policy_rag_service()
    return get_settings()
