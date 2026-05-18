from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    resend_api_key: str = ""
    resend_from_email: str = "concierge@yourdomain.com"
    resend_to_email: str = "sales@yourdomain.com"
    backend_url: str = "http://127.0.0.1:8000"
    show_debug_meta: bool = False
    use_quality_llm: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


def bootstrap() -> Settings:
    from dotenv import load_dotenv

    load_dotenv()
    reset_settings_cache()
    from backend.rag_service import reset_policy_rag_service

    reset_policy_rag_service()
    return get_settings()
