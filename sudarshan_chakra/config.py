from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_ARTIFACTS_DIR = _PACKAGE_DIR / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Sudarshan Chakra AI Firewall"
    host: str = "127.0.0.1"
    port: int = 8000

    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    FPE_KEY: str = ""
    FPE_TWEAK: str = ""

    sentinel_model_path: str = str(_ARTIFACTS_DIR / "sentinel_model.joblib")
    sentinel_b_model_path: str = str(_ARTIFACTS_DIR / "sentinel_b_model.joblib")
    sentinel_vectorizer_path: str = str(_ARTIFACTS_DIR / "vectorizer.joblib")

    redactor_model_path: str = str(_ARTIFACTS_DIR / "aegis_redactor")
    ner_model_path: str = str(_ARTIFACTS_DIR / "redactor_ner_model.joblib")

    database_path: str = "banking.db"
    weil_key_path: str = "private_key.wc"

    @property
    def llm_provider(self) -> str:
        return self.LLM_PROVIDER

    @property
    def llm_model(self) -> str:
        return self.LLM_MODEL

    @property
    def openai_api_key(self) -> str:
        return self.OPENAI_API_KEY

    @property
    def anthropic_api_key(self) -> str:
        return self.ANTHROPIC_API_KEY

    @property
    def fpe_key(self) -> str:
        return self.FPE_KEY

    @property
    def fpe_tweak(self) -> str:
        return self.FPE_TWEAK


settings = Settings()
