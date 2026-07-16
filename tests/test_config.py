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
    assert default_settings.proxy_auto_recovery_max_probes == 5
    assert default_settings.proxy_auto_recovery_timeout_seconds == 1_800
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
