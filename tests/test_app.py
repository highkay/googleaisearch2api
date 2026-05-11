from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from googleaisearch2api.app import create_app
from googleaisearch2api.config import DEFAULT_API_TOKEN, ServiceConfigUpdate, get_settings
from googleaisearch2api.schemas import Citation, GoogleAiResult


def _build_settings_form(**overrides: str) -> dict[str, str]:
    payload = {
        "default_model": "google-search",
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
    def __init__(self, answer_text: str = "Browser-backed answer.") -> None:
        self.answer_text = answer_text
        self.prompts: list[str] = []
        self.closed = False

    def execute(self, config, prompt: str) -> GoogleAiResult:
        self.prompts.append(prompt)
        return GoogleAiResult(
            answer_text=self.answer_text,
            citations=[Citation(title="Source", url="https://example.com")],
            final_url="https://www.google.com/search?udm=50",
            page_title="Google Search",
        )

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _install_fake_pool(app, answer_text: str = "Browser-backed answer.") -> FakePool:
    app.state.services.pool.close()
    pool = FakePool(answer_text=answer_text)
    app.state.services.pool = pool
    return pool


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
            data=_build_settings_form(),
            follow_redirects=False,
        )
        preserved = test_app.state.services.store.get_config()

        clear_response = client.post(
            "/console/settings",
            data=_build_settings_form(clear_browser_proxy_password="on"),
            follow_redirects=False,
        )
        cleared = test_app.state.services.store.get_config()

    assert preserve_response.status_code == 303
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
