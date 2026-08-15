from pathlib import Path

import pytest
from pydantic import ValidationError

from googleaisearch2api.config import (
    AppSettings,
    ServiceConfig,
    ServiceConfigUpdate,
    parse_gemini_warp_proxies,
)


def test_browser_worker_settings_support_current_and_legacy_env_names(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        APP_DATA_DIR=tmp_path,
        BROWSER_WORKERS=3,
        REQUEST_QUEUE_SIZE=9,
        REQUEST_LOG_MAX_ROWS=250,
        GOOGLE_AI_BLOCKED_RETRY_COUNT=2,
        RESIN_STICKY_SESSION_ENABLED=True,
        PROXY_ALLOW_FALLBACK_TO_BASE=True,
        PROXY_AUTO_RECOVERY_ENABLED=True,
        PROXY_AUTO_RECOVERY_INTERVAL_SECONDS=600,
        PROXY_AUTO_RECOVERY_EXISTING_SESSION_LIMIT=50,
        PROXY_AUTO_RECOVERY_MAX_PROBES=4,
        PROXY_AUTO_RECOVERY_MIN_TRIGGER_INTERVAL_SECONDS=120,
        PROXY_AUTO_RECOVERY_SKIP_EGRESS=False,
        PROXY_AUTO_RECOVERY_SKIP_IPLARK=False,
        PROXY_AUTO_RECOVERY_FAST_IPAPI_EGRESS=True,
        PROXY_AUTO_RECOVERY_ALLOW_KNOWN_GOOGLE_BLOCKED_IP=False,
        PROXY_AUTO_RECOVERY_ALLOW_KNOWN_GOOGLE_BLOCKED_PREFIX=False,
        PROXY_AUTO_RECOVERY_SKIP_DUCK_CANARY=False,
        PROXY_AUTO_RECOVERY_CANARY_REPEATS=2,
    )
    assert settings.max_concurrent_requests == 3
    assert settings.request_queue_size == 9
    assert settings.request_log_max_rows == 250
    assert settings.google_ai_blocked_retry_count == 2
    assert settings.resin_sticky_session_enabled is True
    assert settings.proxy_allow_fallback_to_base is True
    assert settings.proxy_auto_recovery_enabled is True
    assert settings.proxy_auto_recovery_interval_seconds == 600
    assert settings.proxy_auto_recovery_existing_session_limit == 50
    assert settings.proxy_auto_recovery_max_probes == 4
    assert settings.proxy_auto_recovery_min_trigger_interval_seconds == 120
    assert settings.proxy_auto_recovery_skip_egress is False
    assert settings.proxy_auto_recovery_skip_iplark is False
    assert settings.proxy_auto_recovery_fast_ipapi_egress is True
    assert settings.proxy_auto_recovery_allow_known_google_blocked_ip is False
    assert settings.proxy_auto_recovery_allow_known_google_blocked_prefix is False
    assert settings.proxy_auto_recovery_skip_duck_canary is False
    assert settings.proxy_auto_recovery_canary_repeats == 2

    legacy_settings = AppSettings(
        _env_file=None,
        APP_DATA_DIR=tmp_path,
        MAX_CONCURRENT_REQUESTS=2,
        GOOGLE_BLOCKED_RETRIES=4,
    )
    assert legacy_settings.max_concurrent_requests == 2
    assert legacy_settings.google_ai_blocked_retry_count == 4

    default_settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert default_settings.google_ai_blocked_retry_count == 0
    assert default_settings.proxy_auto_recovery_enabled is False
    assert default_settings.proxy_auto_recovery_interval_seconds == 43_200
    assert default_settings.proxy_auto_recovery_run_on_startup is False
    assert default_settings.proxy_auto_recovery_existing_sessions is True
    assert default_settings.proxy_auto_recovery_existing_session_limit == 0
    assert default_settings.proxy_auto_recovery_max_probes == 25
    assert default_settings.proxy_auto_recovery_timeout_seconds == 300
    assert default_settings.proxy_auto_recovery_min_trigger_interval_seconds == 900
    assert default_settings.proxy_auto_recovery_skip_egress is True
    assert default_settings.proxy_auto_recovery_skip_iplark is True
    assert default_settings.proxy_auto_recovery_fast_ipapi_egress is False
    assert default_settings.proxy_auto_recovery_fast_http_prefilter is True
    assert default_settings.proxy_auto_recovery_fast_http_scan_limit == 0
    assert default_settings.proxy_auto_recovery_fast_http_workers == 16
    assert default_settings.proxy_auto_recovery_event_fast_http_scan_limit == 40
    assert default_settings.proxy_auto_recovery_allow_known_google_blocked_ip is True
    assert default_settings.proxy_auto_recovery_allow_known_google_blocked_prefix is True
    assert default_settings.proxy_auto_recovery_retry_retired is False
    assert default_settings.proxy_auto_recovery_skip_duck_canary is True
    assert default_settings.proxy_auto_recovery_canary_repeats == 1


def test_search_engine_accepts_gemini() -> None:
    config = ServiceConfig(search_engine="gemini")
    assert config.search_engine == "gemini"


def test_search_engine_rejects_google() -> None:
    with pytest.raises(ValidationError, match="search_engine must be one of"):
        ServiceConfig(search_engine="google")


def test_search_engine_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError, match="search_engine must be one of"):
        ServiceConfig(search_engine="bogus")


def test_search_engine_default_is_gemini(tmp_path: Path) -> None:
    settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert settings.search_engine == "gemini"
    assert ServiceConfig().search_engine == "gemini"
    update = ServiceConfigUpdate(
        default_model="google-search",
        api_token="secret-token",
        browser_headless=True,
        browser_locale="en-US",
        browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=en",
        browser_timeout_ms=90_000,
        answer_timeout_ms=45_000,
    )
    assert update.search_engine == "gemini"


def test_search_engine_accepts_gemini_upstream() -> None:
    config = ServiceConfig(search_engine="gemini-upstream")
    assert config.search_engine == "gemini-upstream"


def test_gemini_upstream_knobs_default_none(tmp_path: Path) -> None:
    settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert settings.gemini_upstream_base_url is None
    assert settings.gemini_upstream_api_key is None
    config = ServiceConfig.from_settings(settings)
    assert config.gemini_upstream_base_url is None
    assert config.gemini_upstream_api_key is None


def test_gemini_upstream_knobs_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_UPSTREAM_BASE_URL", "http://127.0.0.1:8081")
    monkeypatch.setenv("GEMINI_UPSTREAM_API_KEY", "sk-key")
    settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert settings.gemini_upstream_base_url == "http://127.0.0.1:8081"
    assert settings.gemini_upstream_api_key == "sk-key"
    config = ServiceConfig.from_settings(settings)
    assert config.gemini_upstream_base_url == "http://127.0.0.1:8081"
    assert config.gemini_upstream_api_key == "sk-key"


def test_service_config_update_accepts_gemini_upstream_fields() -> None:
    update = ServiceConfigUpdate(
        default_model="google-search",
        api_token="secret-token",
        browser_headless=True,
        browser_locale="en-US",
        browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=en",
        browser_timeout_ms=90_000,
        answer_timeout_ms=45_000,
        gemini_upstream_base_url="https://api.example.com/v1",
        gemini_upstream_api_key="sk-upstream-key",
        gemini_upstream_model="gemini-3.0-pro",
    )
    assert update.gemini_upstream_base_url == "https://api.example.com/v1"
    assert update.gemini_upstream_api_key == "sk-upstream-key"
    assert update.gemini_upstream_model == "gemini-3.0-pro"

    defaults = ServiceConfigUpdate(
        default_model="google-search",
        api_token="secret-token",
        browser_headless=True,
        browser_locale="en-US",
        browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=en",
        browser_timeout_ms=90_000,
        answer_timeout_ms=45_000,
    )
    assert defaults.gemini_upstream_base_url is None
    assert defaults.gemini_upstream_api_key is None
    assert defaults.gemini_upstream_model == "gemini-3.7-flash"


def test_gemini_fast_probe_knobs_default_and_env(tmp_path: Path, monkeypatch) -> None:
    default_settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert default_settings.gemini_fast_probe_timeout_s == 8.0
    assert default_settings.gemini_max_probe_sessions == 3

    monkeypatch.setenv("GEMINI_FAST_PROBE_TIMEOUT_S", "15.5")
    monkeypatch.setenv("GEMINI_MAX_PROBE_SESSIONS", "5")
    settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert settings.gemini_fast_probe_timeout_s == 15.5
    assert settings.gemini_max_probe_sessions == 5

    config = ServiceConfig.from_settings(settings)
    assert not hasattr(config, "gemini_fast_probe_timeout_s")
    assert not hasattr(config, "gemini_max_probe_sessions")

    monkeypatch.setenv("GEMINI_FAST_PROBE_TIMEOUT_S", "0")
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    monkeypatch.setenv("GEMINI_FAST_PROBE_TIMEOUT_S", "15.5")
    monkeypatch.setenv("GEMINI_MAX_PROBE_SESSIONS", "0")
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)


def test_ai_mode_http_enabled_defaults_false(tmp_path: Path) -> None:
    assert ServiceConfig().ai_mode_http_enabled is False
    default_settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert default_settings.ai_mode_http_enabled is False


def test_ai_mode_http_enabled_env_alias(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AI_MODE_HTTP_ENABLED", "true")
    settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert settings.ai_mode_http_enabled is True


def test_from_settings_maps_ai_mode_http_enabled(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        APP_DATA_DIR=tmp_path,
        AI_MODE_HTTP_ENABLED=True,
    )
    config = ServiceConfig.from_settings(settings)
    assert config.ai_mode_http_enabled is True


def test_pool_wait_timeout_covers_patchright_retry_worst_case() -> None:
    config = ServiceConfig(browser_timeout_ms=90_000, answer_timeout_ms=45_000)
    attempt_ms = (90_000 * 4) + 15_000 + 45_000 + 18_000
    assert config.pool_wait_timeout_ms() >= 2 * attempt_ms


def test_parse_gemini_warp_proxies_empty_and_blank() -> None:
    assert parse_gemini_warp_proxies("") == []
    assert parse_gemini_warp_proxies("  , , ") == []


def test_parse_gemini_warp_proxies_strips_and_splits() -> None:
    assert parse_gemini_warp_proxies("socks5h://a:1080, socks5h://b:1080") == [
        "socks5h://a:1080",
        "socks5h://b:1080",
    ]


def test_parse_gemini_warp_proxies_dedupes_preserving_order() -> None:
    assert parse_gemini_warp_proxies(
        "socks5h://a:1080,socks5h://b:1080,socks5h://a:1080"
    ) == ["socks5h://a:1080", "socks5h://b:1080"]


def test_gemini_warp_proxies_default_and_env(tmp_path: Path, monkeypatch) -> None:
    default_settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert default_settings.gemini_warp_proxies == ""

    monkeypatch.setenv("GEMINI_WARP_PROXIES", "socks5h://a:1080")
    settings = AppSettings(_env_file=None, APP_DATA_DIR=tmp_path)
    assert settings.gemini_warp_proxies == "socks5h://a:1080"


def test_browser_worker_hard_timeout_s_defaults_derived() -> None:
    config = ServiceConfig(browser_timeout_ms=90_000, answer_timeout_ms=45_000)
    assert config.browser_worker_hard_timeout_s == config.pool_wait_timeout_ms() / 1000 + 60

    explicit = ServiceConfig(
        browser_timeout_ms=90_000,
        answer_timeout_ms=45_000,
        browser_worker_hard_timeout_seconds=42,
    )
    assert explicit.browser_worker_hard_timeout_s == 42
