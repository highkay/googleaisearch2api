from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from googleaisearch2api.app import create_app
from googleaisearch2api.config import DEFAULT_API_TOKEN, ServiceConfigUpdate, get_settings


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
