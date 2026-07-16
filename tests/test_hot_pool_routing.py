from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from googleaisearch2api.app import create_app
from googleaisearch2api.config import ServiceConfigUpdate, get_settings
from googleaisearch2api.schemas import Citation, GoogleAiResult

from test_app import FakePool, FakeProxyAutoRecovery, _auth_headers, _install_fake_duck_pool


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


def test_auto_routes_to_duck_while_recovery_holds_gate(test_app) -> None:
    with TestClient(test_app) as client:
        test_app.state.services.store.update_config(
            ServiceConfigUpdate(
                default_model="google-search",
                search_engine="auto",
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
        active = test_app.state.services.proxy_session_store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name="user1",
            proxy_username="openai.user1",
            status="active",
        )
        test_app.state.services.proxy_session_store.mark_canary_success(active.id)
        google_pool = FakePool(
            outcomes=[
                GoogleAiResult(
                    answer_text="Should not run Google while recovery holds the gate.",
                    citations=[Citation(title="Source", url="https://example.com")],
                    final_url="https://www.google.com/search?udm=50",
                    page_title="Google",
                )
            ]
        )
        test_app.state.services.pool.close()
        test_app.state.services.pool = google_pool
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck while recovering.")
        recovery = FakeProxyAutoRecovery()
        recovery.running = True
        test_app.state.services.proxy_auto_recovery = recovery

        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck while recovering."
    assert google_pool.prompts == []
    assert duck_pool.prompts == ["User request:\nQuestion"]


def test_healthz_exposes_hot_pool_and_browser_gate(test_app) -> None:
    with TestClient(test_app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert "sticky_hot_pool_sessions" in payload
    assert "browser_gate" in payload
    assert "exclusive_holder" in payload["browser_gate"]
