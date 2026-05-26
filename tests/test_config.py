from pathlib import Path

from googleaisearch2api.config import AppSettings


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
    )
    assert settings.max_concurrent_requests == 3
    assert settings.request_queue_size == 9
    assert settings.request_log_max_rows == 250
    assert settings.google_ai_blocked_retry_count == 2
    assert settings.resin_sticky_session_enabled is True
    assert settings.proxy_allow_fallback_to_base is True

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
