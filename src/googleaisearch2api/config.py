from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_API_TOKEN = "change-me-google-search"


def _mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}***{value[-4:]}"


class ServiceConfig(BaseModel):
    default_model: str = "google-search"
    api_token: str = DEFAULT_API_TOKEN
    browser_headless: bool = True
    browser_user_agent: str | None = None
    browser_locale: str = "en-US"
    browser_base_url: str = "https://www.google.com/search?udm=50&aep=11&hl=en"
    browser_timeout_ms: int = 90_000
    answer_timeout_ms: int = 45_000
    browser_proxy_server: str | None = None
    browser_proxy_username: str | None = None
    browser_proxy_password: str | None = None
    browser_proxy_bypass: str | None = None

    @property
    def browser_label(self) -> str:
        return "chrome"

    @property
    def proxy_enabled(self) -> bool:
        return bool(self.browser_proxy_server)

    @property
    def masked_api_token(self) -> str:
        return _mask_secret(self.api_token)

    @property
    def masked_proxy_password(self) -> str:
        return _mask_secret(self.browser_proxy_password)

    def pool_wait_timeout_ms(self, *, buffer_ms: int = 5_000) -> int:
        first_wait_ms = min(self.answer_timeout_ms, 15_000)
        total_ms = (self.browser_timeout_ms * 3) + first_wait_ms + self.answer_timeout_ms
        return max(total_ms + buffer_ms, 1_000)

    @classmethod
    def from_settings(cls, settings: AppSettings) -> ServiceConfig:
        return cls(
            default_model=settings.default_model,
            api_token=settings.api_token,
            browser_headless=settings.browser_headless,
            browser_user_agent=settings.browser_user_agent or None,
            browser_locale=settings.browser_locale,
            browser_base_url=settings.browser_base_url,
            browser_timeout_ms=settings.browser_timeout_ms,
            answer_timeout_ms=settings.answer_timeout_ms,
            browser_proxy_server=settings.browser_proxy_server or None,
            browser_proxy_username=settings.browser_proxy_username or None,
            browser_proxy_password=settings.browser_proxy_password or None,
            browser_proxy_bypass=settings.browser_proxy_bypass or None,
        )


class ServiceConfigUpdate(BaseModel):
    default_model: str
    api_token: str
    browser_headless: bool
    browser_user_agent: str | None = None
    browser_locale: str
    browser_base_url: str
    browser_timeout_ms: int
    answer_timeout_ms: int
    browser_proxy_server: str | None = None
    browser_proxy_username: str | None = None
    browser_proxy_password: str | None = None
    browser_proxy_bypass: str | None = None


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Google AI Search2API", validation_alias="APP_NAME")
    app_host: str = Field(default="127.0.0.1", validation_alias="APP_HOST")
    app_port: int = Field(default=8000, validation_alias="APP_PORT")
    app_log_level: str = Field(default="INFO", validation_alias="APP_LOG_LEVEL")
    app_data_dir: Path = Field(default=Path("data"), validation_alias="APP_DATA_DIR")

    default_model: str = Field(default="google-search", validation_alias="DEFAULT_MODEL")
    api_token: str = Field(default=DEFAULT_API_TOKEN, validation_alias="API_TOKEN")

    browser_headless: bool = Field(default=True, validation_alias="BROWSER_HEADLESS")
    browser_user_agent: str = Field(default="", validation_alias="BROWSER_USER_AGENT")
    browser_locale: str = Field(default="en-US", validation_alias="BROWSER_LOCALE")
    browser_base_url: str = Field(
        default="https://www.google.com/search?udm=50&aep=11&hl=en",
        validation_alias="BROWSER_BASE_URL",
    )
    browser_timeout_ms: int = Field(default=90_000, validation_alias="BROWSER_TIMEOUT_MS")
    answer_timeout_ms: int = Field(default=45_000, validation_alias="ANSWER_TIMEOUT_MS")
    max_concurrent_requests: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("BROWSER_WORKERS", "MAX_CONCURRENT_REQUESTS"),
    )
    request_queue_size: int = Field(default=8, ge=1, validation_alias="REQUEST_QUEUE_SIZE")
    request_log_max_rows: int = Field(default=200, ge=10, validation_alias="REQUEST_LOG_MAX_ROWS")

    browser_proxy_server: str = Field(default="", validation_alias="BROWSER_PROXY_SERVER")
    browser_proxy_username: str = Field(default="", validation_alias="BROWSER_PROXY_USERNAME")
    browser_proxy_password: str = Field(default="", validation_alias="BROWSER_PROXY_PASSWORD")
    browser_proxy_bypass: str = Field(default="", validation_alias="BROWSER_PROXY_BYPASS")

    @property
    def db_path(self) -> Path:
        return self.app_data_dir / "googleaisearch2api.sqlite3"

    def ensure_directories(self) -> None:
        self.app_data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    settings = AppSettings()
    settings.ensure_directories()
    return settings
