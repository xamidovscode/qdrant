from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Async MVP API"
    ENV: str = Field(default="dev")
    DEBUG: bool = True

    API_V1_PREFIX: str = "/api/v1"

    LOG_LEVEL: str = "INFO"
    HTTP_TIMEOUT_SECONDS: float = 10.0


settings = Settings()
