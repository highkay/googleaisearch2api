from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from googleaisearch2api.app import create_app
from googleaisearch2api.browser import GoogleAiBlockedError
from googleaisearch2api.config import DEFAULT_API_TOKEN, ServiceConfigUpdate, get_settings
from googleaisearch2api.schemas import Citation, GoogleAiResult


def _build_settings_form(**overrides: str) -> dict[str, str]:
    payload = {
        "default_model": "google-search",
        "search_engine": "google",
        "api_token": "",
        "browser_headless": "on",
        "browser_user_agent": "",
        "browser_locale": "en-US",
        "browser_base_url": "https://www.google.com/search?udm=50&aep=11&hl=en",
        "browser_timeout_ms": "90000",
        "answer_timeout_ms": "45000",
        "browser_proxy_server": "http://127.0.0.1:7890",
        "browser_proxy_username": "proxy-user",
        "browser_proxy_password": "",
        "browser_proxy_bypass": "localhost",
    }
    payload.update(overrides)
    return payload


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer secret-token"}


class FakePool:
    def __init__(
        self,
        answer_text: str = "Browser-backed answer.",
        outcomes: list[GoogleAiResult | Exception] | None = None,
    ) -> None:
        self.answer_text = answer_text
        self.outcomes = list(outcomes or [])
        self.prompts: list[str] = []
        self.configs: list = []
        self.blocked_retry_counts: list[int | None] = []
        self.reset_calls = 0
        self.closed = False

    def execute(
        self,
        config,
        prompt: str,
        *,
        blocked_retry_count: int | None = None,
    ) -> GoogleAiResult:
        self.configs.append(config)
        self.prompts.append(prompt)
        self.blocked_retry_counts.append(blocked_retry_count)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return GoogleAiResult(
            answer_text=self.answer_text,
            citations=[Citation(title="Source", url="https://example.com")],
            final_url="https://www.google.com/search?udm=50",
            page_title="Google Search",
        )

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


def _install_fake_pool(
    app,
    answer_text: str = "Browser-backed answer.",
    outcomes: list[GoogleAiResult | Exception] | None = None,
) -> FakePool:
    app.state.services.pool.close()
    pool = FakePool(answer_text=answer_text, outcomes=outcomes)
    app.state.services.pool = pool
    return pool


def _install_fake_duck_pool(
    app,
    answer_text: str = "Duck answer.",
    outcomes: list[GoogleAiResult | Exception] | None = None,
) -> FakePool:
    app.state.services.duck_pool.close()
    pool = FakePool(answer_text=answer_text, outcomes=outcomes)
    app.state.services.duck_pool = pool
    return pool


def _set_search_engine(app, search_engine: str) -> None:
    current = app.state.services.store.get_config()
    app.state.services.store.update_config(
        ServiceConfigUpdate(
            default_model=current.default_model,
            search_engine=search_engine,
            api_token=current.api_token,
            browser_headless=current.browser_headless,
            browser_user_agent=current.browser_user_agent,
            browser_locale=current.browser_locale,
            browser_base_url=current.browser_base_url,
            browser_timeout_ms=current.browser_timeout_ms,
            answer_timeout_ms=current.answer_timeout_ms,
            browser_proxy_server=current.browser_proxy_server,
            browser_proxy_username=current.browser_proxy_username,
            browser_proxy_password=current.browser_proxy_password,
            browser_proxy_bypass=current.browser_proxy_bypass,
            resin_sticky_session_enabled=current.resin_sticky_session_enabled,
        )
    )


@pytest.fixture
def test_app(tmp_path, monkeypatch) -> Iterator:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("API_TOKEN", "secret-token")
    monkeypatch.setenv("BROWSER_WORKERS", "1")
    monkeypatch.setenv("REQUEST_QUEUE_SIZE", "1")
    get_settings.cache_clear()
    app = create_app()
    try:
        yield app
    finally:
        get_settings.cache_clear()


def test_console_redirects_to_login_without_session(test_app) -> None:
    with TestClient(test_app) as client:
        response = client.get("/console", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/console/login?next=%2Fconsole"


def test_console_summary_requires_session_cookie(test_app) -> None:
    with TestClient(test_app) as client:
        response = client.get("/console/summary.json")

    assert response.status_code == 401
    assert response.json()["detail"] == "Console login required."


def test_console_login_sets_cookie_and_unlocks_console(test_app) -> None:
    with TestClient(test_app) as client:
        response = client.post(
            "/console/login",
            data={"console_token": "secret-token", "next": "/console"},
            follow_redirects=False,
        )
        page = client.get("/console")

    assert response.status_code == 303
    assert response.headers["location"] == "/console"
    assert "googleaisearch2api_console_token" in response.headers["set-cookie"]
    assert page.status_code == 200
    assert "Runtime Config" in page.text
    assert "Auto fallback" in page.text


def test_console_settings_preserve_and_clear_hidden_secrets(test_app) -> None:
    with TestClient(test_app) as client:
        test_app.state.services.store.update_config(
            ServiceConfigUpdate(
                default_model="google-search",
                api_token="secret-token",
                browser_headless=True,
                browser_user_agent="",
                browser_locale="en-US",
                browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=en",
                browser_timeout_ms=90_000,
                answer_timeout_ms=45_000,
                browser_proxy_server="http://127.0.0.1:7890",
                browser_proxy_username="proxy-user",
                browser_proxy_password="stored-pass",
                browser_proxy_bypass="localhost",
            )
        )
        client.post(
            "/console/login",
            data={"console_token": "secret-token", "next": "/console"},
            follow_redirects=False,
        )

        preserve_response = client.post(
            "/console/settings",
            data=_build_settings_form(search_engine="auto"),
            follow_redirects=False,
        )
        preserved = test_app.state.services.store.get_config()

        clear_response = client.post(
            "/console/settings",
            data=_build_settings_form(
                search_engine="auto",
                clear_browser_proxy_password="on",
            ),
            follow_redirects=False,
        )
        cleared = test_app.state.services.store.get_config()

    assert preserve_response.status_code == 303
    assert preserved.search_engine == "auto"
    assert preserved.api_token == "secret-token"
    assert preserved.browser_proxy_password == "stored-pass"
    assert clear_response.status_code == 303
    assert cleared.browser_proxy_password is None


def test_network_exposed_app_requires_non_default_api_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.setenv("API_TOKEN", DEFAULT_API_TOKEN)
    get_settings.cache_clear()
    app = create_app()

    try:
        with pytest.raises(RuntimeError, match="API_TOKEN must be set"):
            with TestClient(app):
                pass
    finally:
        get_settings.cache_clear()


def test_chat_completions_rejects_tool_message_role(test_app) -> None:
    with TestClient(test_app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=_auth_headers(),
            json={
                "model": "google-search",
                "messages": [{"role": "tool", "content": "tool output"}],
            },
        )

    assert response.status_code == 422


def test_chat_completions_rejects_image_parts(test_app) -> None:
    with TestClient(test_app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=_auth_headers(),
            json={
                "model": "google-search",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.jpg"},
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 422


def test_query_post_returns_tool_friendly_response_shape(test_app) -> None:
    with TestClient(test_app) as client:
        pool = _install_fake_pool(test_app, answer_text="Tool friendly answer.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={
                "model": "google-search",
                "query": "Question",
                "instructions": "Use verified facts only.",
                "context": [{"role": "assistant", "content": "Previous answer"}],
            },
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "query.result"
    assert payload["model"] == "google-search"
    assert payload["answer"] == "Tool friendly answer."
    assert payload["usage"]["total_tokens"] >= payload["usage"]["input_tokens"]
    assert payload["citations"][0]["url"] == "https://example.com"
    assert pool.prompts == [
        "System instructions:\n"
        "Use verified facts only.\n\n"
        "Conversation context:\n"
        "ASSISTANT: Previous answer\n\n"
        "User request:\n"
        "Question"
    ]
    assert recent[0].endpoint == "/query"
    assert recent[0].engine == "google"


def test_query_duck_engine_uses_duck_pool_only(test_app) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "duck")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck answer.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck answer."
    assert google_pool.prompts == []
    assert duck_pool.prompts == ["User request:\nQuestion"]
    assert duck_pool.blocked_retry_counts == [0]
    assert recent[0].engine == "duck"


def test_query_auto_falls_back_to_duck_when_google_is_unavailable(test_app) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(
            test_app,
            outcomes=[GoogleAiBlockedError("Google blocked this browser session.")],
        )
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck fallback.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=2)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck fallback."
    assert google_pool.prompts == ["User request:\nQuestion"]
    assert duck_pool.prompts == ["User request:\nQuestion"]
    assert [record.engine for record in recent] == ["duck", "google"]
    assert [record.status for record in recent] == ["ok", "error"]


def test_query_auto_falls_back_to_duck_when_google_answer_quality_fails(
    test_app,
) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(
            test_app,
            answer_text="You said: User request:\nQuestion",
        )
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck fallback.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=2)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck fallback."
    assert google_pool.prompts == ["User request:\nQuestion"]
    assert duck_pool.prompts == ["User request:\nQuestion"]
    assert [record.engine for record in recent] == ["duck", "google"]
    assert [record.status for record in recent] == ["ok", "error"]
    assert "quality check" in (recent[1].error_message or "")


def test_query_uses_active_sticky_proxy_session_when_enabled(test_app) -> None:
    with TestClient(test_app) as client:
        test_app.state.services.store.update_config(
            ServiceConfigUpdate(
                default_model="google-search",
                api_token="secret-token",
                browser_headless=True,
                browser_user_agent="",
                browser_locale="en-US",
                browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=en",
                browser_timeout_ms=90_000,
                answer_timeout_ms=45_000,
                browser_proxy_server="http://192.0.2.1:2260",
                browser_proxy_username="openai",
                browser_proxy_password="proxy-pass",
                browser_proxy_bypass="",
                resin_sticky_session_enabled=True,
            )
        )
        snapshot = test_app.state.services.proxy_session_store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name="user1",
            proxy_username="openai.user1",
            status="active",
        )
        test_app.state.services.proxy_session_store.update_egress(
            proxy_session_id=snapshot.id,
            ips=["203.0.113.10"],
            source="test",
        )
        test_app.state.services.proxy_session_store.mark_canary_success(snapshot.id)
        pool = _install_fake_pool(test_app, answer_text="Sticky answer.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert pool.configs[0].browser_proxy_username == "openai.user1"
    assert recent[0].resin_sticky_session_enabled is True
    assert recent[0].proxy_base_username == "openai"
    assert recent[0].proxy_username == "openai.user1"
    assert recent[0].proxy_primary_ip == "203.0.113.10"


def test_duck_query_can_use_duck_ok_session_in_google_cooldown(test_app) -> None:
    with TestClient(test_app) as client:
        test_app.state.services.store.update_config(
            ServiceConfigUpdate(
                default_model="google-search",
                search_engine="duck",
                api_token="secret-token",
                browser_headless=True,
                browser_user_agent="",
                browser_locale="en-US",
                browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=en",
                browser_timeout_ms=90_000,
                answer_timeout_ms=45_000,
                browser_proxy_server="http://192.0.2.1:2260",
                browser_proxy_username="openai",
                browser_proxy_password="proxy-pass",
                browser_proxy_bypass="",
                resin_sticky_session_enabled=True,
            )
        )
        snapshot = test_app.state.services.proxy_session_store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name="user1",
            proxy_username="openai.user1",
        )
        test_app.state.services.proxy_session_store.update_egress(
            proxy_session_id=snapshot.id,
            ips=["203.0.113.10"],
            source="test",
        )
        test_app.state.services.proxy_session_store.update_iplark_result(
            proxy_session_id=snapshot.id,
            quality_score=80,
            min_quality_score=0,
        )
        test_app.state.services.proxy_session_store.mark_session_cooldown(
            snapshot.id,
            reason="google blocked",
        )
        test_app.state.services.proxy_session_store.mark_duck_canary_success(snapshot.id)
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck answer.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)
        sessions = {
            item.proxy_username: item
            for item in test_app.state.services.proxy_session_store.list_proxy_sessions(
                limit=10
            )
        }

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck answer."
    assert google_pool.prompts == []
    assert duck_pool.configs[0].browser_proxy_username == "openai.user1"
    assert recent[0].engine == "duck"
    assert recent[0].proxy_username == "openai.user1"
    assert sessions["openai.user1"].status == "cooldown"
    assert sessions["openai.user1"].duck_canary_status == "ok"
    assert sessions["openai.user1"].request_success_count == 0


def test_query_reselects_sticky_proxy_session_after_google_block(test_app) -> None:
    with TestClient(test_app) as client:
        test_app.state.services.settings.google_ai_blocked_retry_count = 1
        test_app.state.services.store.update_config(
            ServiceConfigUpdate(
                default_model="google-search",
                api_token="secret-token",
                browser_headless=True,
                browser_user_agent="",
                browser_locale="en-US",
                browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=en",
                browser_timeout_ms=90_000,
                answer_timeout_ms=45_000,
                browser_proxy_server="http://192.0.2.1:2260",
                browser_proxy_username="openai",
                browser_proxy_password="proxy-pass",
                browser_proxy_bypass="",
                resin_sticky_session_enabled=True,
            )
        )
        first = test_app.state.services.proxy_session_store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name="user1",
            proxy_username="openai.user1",
            status="active",
        )
        second = test_app.state.services.proxy_session_store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name="user2",
            proxy_username="openai.user2",
            status="active",
        )
        test_app.state.services.proxy_session_store.update_egress(
            proxy_session_id=first.id,
            ips=["203.0.113.10"],
            source="test",
        )
        test_app.state.services.proxy_session_store.update_egress(
            proxy_session_id=second.id,
            ips=["203.0.113.20"],
            source="test",
        )
        test_app.state.services.proxy_session_store.mark_canary_success(first.id)
        test_app.state.services.proxy_session_store.mark_canary_success(second.id)
        blocked_error = GoogleAiBlockedError(
            "Google blocked the session while opening query page: "
            "our systems have detected unusual traffic from your computer network. "
            "ip address: 2606:c700:1:47:9e6b:ff:fe5e:b6f5"
        )
        pool = _install_fake_pool(
            test_app,
            answer_text="Recovered answer.",
            outcomes=[blocked_error],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=2)
        sessions = {
            snapshot.proxy_username: snapshot
            for snapshot in test_app.state.services.proxy_session_store.list_proxy_sessions(
                limit=10
            )
        }

    assert response.status_code == 200
    assert response.json()["answer"] == "Recovered answer."
    assert [config.browser_proxy_username for config in pool.configs] == [
        "openai.user1",
        "openai.user2",
    ]
    assert pool.blocked_retry_counts == [0, 0]
    assert pool.reset_calls == 1

    requests_by_proxy = {record.proxy_username: record for record in recent}
    assert requests_by_proxy["openai.user1"].status == "error"
    assert requests_by_proxy["openai.user1"].google_block_ips == [
        "2606:c700:1:47:9e6b:ff:fe5e:b6f5"
    ]
    assert requests_by_proxy["openai.user2"].status == "ok"
    assert sessions["openai.user1"].status == "cooldown"
    assert sessions["openai.user1"].request_block_count == 1
    assert sessions["openai.user2"].status == "active"
    assert sessions["openai.user2"].request_success_count == 1


def test_query_fails_fast_when_sticky_enabled_without_active_session(test_app) -> None:
    with TestClient(test_app) as client:
        test_app.state.services.store.update_config(
            ServiceConfigUpdate(
                default_model="google-search",
                api_token="secret-token",
                browser_headless=True,
                browser_user_agent="",
                browser_locale="en-US",
                browser_base_url="https://www.google.com/search?udm=50&aep=11&hl=en",
                browser_timeout_ms=90_000,
                answer_timeout_ms=45_000,
                browser_proxy_server="http://192.0.2.1:2260",
                browser_proxy_username="openai",
                browser_proxy_password="proxy-pass",
                browser_proxy_bypass="",
                resin_sticky_session_enabled=True,
            )
        )
        pool = _install_fake_pool(test_app)
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 503
    assert "No active sticky proxy session" in response.json()["detail"]
    assert pool.prompts == []
    assert recent[0].status == "error"
    assert recent[0].resin_sticky_session_enabled is True


def test_query_get_returns_tool_friendly_response_shape(test_app) -> None:
    with TestClient(test_app) as client:
        pool = _install_fake_pool(test_app, answer_text="GET answer.")
        response = client.get(
            "/query",
            headers=_auth_headers(),
            params={
                "q": "Question",
                "include_citations": False,
                "include_google_metadata": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "GET answer."
    assert "citations" not in payload
    assert "google_ai" not in payload
    assert pool.prompts == ["User request:\nQuestion"]


def test_query_get_rejects_blank_query_without_calling_pool(test_app) -> None:
    with TestClient(test_app) as client:
        pool = _install_fake_pool(test_app)
        response = client.get(
            "/query",
            headers=_auth_headers(),
            params={"q": "   "},
        )

    assert response.status_code == 422
    assert pool.prompts == []


def test_query_stream_returns_simple_sse_events(test_app) -> None:
    with TestClient(test_app) as client:
        _install_fake_pool(test_app, answer_text="Streaming answer.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={
                "model": "google-search",
                "query": "Question",
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: query.created" in response.text
    assert "event: answer.delta" in response.text
    assert '"delta": "Streaming answer."' in response.text
    assert "event: query.completed" in response.text


def test_query_rejects_empty_query_and_invalid_context_role(test_app) -> None:
    with TestClient(test_app) as client:
        pool = _install_fake_pool(test_app)
        empty_query = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "   "},
        )
        invalid_role = client.post(
            "/query",
            headers=_auth_headers(),
            json={
                "model": "google-search",
                "query": "hello",
                "context": [{"role": "tool", "content": "tool output"}],
            },
        )

    assert empty_query.status_code == 422
    assert invalid_role.status_code == 422
    assert pool.prompts == []


def test_responses_rejects_tools_field(test_app) -> None:
    with TestClient(test_app) as client:
        response = client.post(
            "/v1/responses",
            headers=_auth_headers(),
            json={
                "model": "google-search",
                "input": "hello",
                "tools": [{"type": "function", "name": "lookup"}],
            },
        )

    assert response.status_code == 422
