from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
        ),
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

    def has_google_api(self) -> bool:
        return bool(self.google_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


def bootstrap() -> Settings:
    from dotenv import load_dotenv

    load_dotenv()
    reset_settings_cache()
    from backend.gemini_service import reset_gemini_client
    from backend.rag_service import reset_policy_rag_service

    reset_gemini_client()
    reset_policy_rag_service()
    return get_settings()
