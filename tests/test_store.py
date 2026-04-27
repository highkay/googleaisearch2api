from pathlib import Path

from googleaisearch2api.config import ServiceConfig, ServiceConfigUpdate
from googleaisearch2api.db import create_db_engine, create_session_factory, create_tables
from googleaisearch2api.schemas import Citation, GoogleAiResult
from googleaisearch2api.store import ConfigStore


def _make_store(tmp_path: Path, request_log_max_rows: int = 200) -> ConfigStore:
    db_path = tmp_path / "test.sqlite3"
    engine = create_db_engine(str(db_path))
    create_tables(engine)
    session_factory = create_session_factory(engine)
    return ConfigStore(
        session_factory,
        ServiceConfig(),
        request_log_max_rows=request_log_max_rows,
    )


def test_get_config_creates_default_row(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    config = store.get_config()
    assert config.default_model == "google-search"


def test_update_config_and_summary(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    update = ServiceConfigUpdate(
        default_model="google-search-v2",
        api_token="secret-token",
        browser_headless=False,
        browser_user_agent="Mozilla/5.0 TestBrowser",
        browser_locale="zh-CN",
        browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=zh-CN",
        browser_timeout_ms=120_000,
        answer_timeout_ms=60_000,
        browser_proxy_server="http://127.0.0.1:7890",
        browser_proxy_username="",
        browser_proxy_password="",
        browser_proxy_bypass="localhost",
    )
    config = store.update_config(update)
    assert config.default_model == "google-search-v2"
    assert config.browser_user_agent == "Mozilla/5.0 TestBrowser"
    assert config.proxy_enabled is True

    request_id = store.start_request(
        endpoint="/v1/chat/completions",
        model_name=config.default_model,
        prompt_preview="hello",
        client_ip="127.0.0.1",
        stream=False,
        config=config,
    )
    store.finish_request_success(
        request_id,
        GoogleAiResult(
            answer_text="world",
            citations=[Citation(title="OpenAI", url="https://openai.com")],
            final_url="https://www.google.com/search?udm=50",
            page_title="Google Search",
        ),
        duration_ms=1234,
    )

    summary = store.get_summary()
    recent = store.list_recent_requests(limit=5)

    assert summary.total_requests == 1
    assert summary.successful_requests == 1
    assert recent[0].response_preview == "world"


def test_request_logs_redact_secrets_and_trim_old_rows(tmp_path: Path) -> None:
    store = _make_store(tmp_path, request_log_max_rows=2)
    config = store.get_config()

    first_id = store.start_request(
        endpoint="/v1/chat/completions",
        model_name=config.default_model,
        prompt_preview="Authorization: Bearer sk-secret-token-value",
        client_ip="127.0.0.1",
        stream=False,
        config=config,
    )
    store.finish_request_error(
        first_id,
        "password=supersecret https://user:pass@example.com/path",
        duration_ms=10,
    )

    second_id = store.start_request(
        endpoint="/v1/chat/completions",
        model_name=config.default_model,
        prompt_preview="follow-up question",
        client_ip="127.0.0.1",
        stream=False,
        config=config,
    )
    store.finish_request_success(
        second_id,
        GoogleAiResult(
            answer_text="api_key=AIzaVerySecretValue1234567890",
            citations=[Citation(title="Example", url="https://example.com")],
            final_url="https://www.google.com/search?udm=50&aep=11&q=private+prompt&hl=en",
            page_title="Google Search",
        ),
        duration_ms=20,
    )

    before_trim = store.list_recent_requests(limit=5)
    assert "sk-secret-token-value" not in before_trim[1].prompt_preview
    assert "***" in before_trim[1].prompt_preview
    assert "supersecret" not in before_trim[1].error_message
    assert "https://***:***@" in before_trim[1].error_message

    third_id = store.start_request(
        endpoint="/v1/responses",
        model_name=config.default_model,
        prompt_preview="third request",
        client_ip="127.0.0.1",
        stream=False,
        config=config,
    )
    store.finish_request_success(
        third_id,
        GoogleAiResult(
            answer_text="third result",
            final_url="https://www.google.com/search?udm=50&aep=11&hl=en",
            page_title="Google Search",
        ),
        duration_ms=30,
    )

    recent = store.list_recent_requests(limit=5)

    assert len(recent) == 2
    assert all(item.id != first_id for item in recent)
    assert "AIzaVerySecretValue1234567890" not in recent[1].response_preview
    assert "***" in recent[1].response_preview
    assert "private+prompt" not in recent[1].final_url
    assert "q=" not in recent[1].final_url
