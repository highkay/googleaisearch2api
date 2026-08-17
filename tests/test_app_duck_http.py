from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from test_app import _auth_headers, _install_fake_duck_pool, _set_search_engine

from googleaisearch2api.app import create_app
from googleaisearch2api.config import (
    ServiceConfigUpdate,
    get_settings,
)
from googleaisearch2api.duck_http import (
    DuckHttpBlockedError,
    DuckHttpRateLimitedError,
    DuckHttpRuntimeError,
)
from googleaisearch2api.fast_proxy_probe import FastProxyProbeResult
from googleaisearch2api.gemini_proxy_pool import GeminiWarpPool
from googleaisearch2api.schemas import Citation, GoogleAiResult


@pytest.fixture
def test_app(tmp_path, monkeypatch) -> Iterator:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("API_TOKEN", "secret-token")
    monkeypatch.setenv("BROWSER_WORKERS", "1")
    monkeypatch.setenv("REQUEST_QUEUE_SIZE", "1")
    # Isolate from the developer .env (recovery startup + real proxy would race SQLite).
    monkeypatch.setenv("PROXY_AUTO_RECOVERY_ENABLED", "false")
    monkeypatch.setenv("PROXY_AUTO_RECOVERY_RUN_ON_STARTUP", "false")
    monkeypatch.setenv("BROWSER_PROXY_SERVER", "")
    monkeypatch.setenv("BROWSER_PROXY_USERNAME", "")
    monkeypatch.setenv("BROWSER_PROXY_PASSWORD", "")
    monkeypatch.setenv("RESIN_STICKY_SESSION_ENABLED", "false")
    monkeypatch.setenv("GEMINI_UPSTREAM_BASE_URL", "")
    monkeypatch.setenv("GEMINI_UPSTREAM_API_KEY", "")
    monkeypatch.setenv("GEMINI_WARP_PROXIES", "")
    # Pin tests against the historical default; the production .env may set a
    # different value and BaseSettings reads the repo-root .env.
    monkeypatch.setenv("ANSWER_TIMEOUT_MS", "45000")
    # This suite owns the browserless duck HTTP path; the browser path suite
    # (test_app.py) pins DUCK_ENGINE=browser.
    monkeypatch.setenv("DUCK_ENGINE", "http")
    get_settings.cache_clear()
    app = create_app()
    try:
        yield app
    finally:
        get_settings.cache_clear()


def _set_duck_engine(test_app, value: str) -> None:
    test_app.state.services.settings.duck_engine = value


class FakeDuckHttpClient:
    """Stand-in for `app.DuckHttpClient` (browserless solver+chat, no pool)."""

    def __init__(
        self,
        answer_text: str = "Duck HTTP answer.",
        outcomes: list[GoogleAiResult | Exception] | None = None,
    ) -> None:
        self.answer_text = answer_text
        self.outcomes = list(outcomes or [])
        self.prompts: list[str] = []
        self.proxies: list[dict[str, str] | None] = []
        self.models: list[str | None] = []
        self.timeouts: list[float] = []

    def __call__(self, *, timeout_s: float = 60.0) -> FakeDuckHttpClient:
        self.timeouts.append(timeout_s)
        return self

    def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        proxies: dict[str, str] | None = None,
        session=None,
    ) -> GoogleAiResult:
        self.prompts.append(prompt)
        self.models.append(model)
        self.proxies.append(proxies)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return GoogleAiResult(
            answer_text=self.answer_text,
            citations=[Citation(title="Duck source", url="https://example.com")],
            final_url="https://duck.ai/",
            page_title="Duck.ai",
        )


def _install_fake_duck_http_client(
    monkeypatch,
    answer_text: str = "Duck HTTP answer.",
    outcomes: list[GoogleAiResult | Exception] | None = None,
) -> FakeDuckHttpClient:
    fake = FakeDuckHttpClient(answer_text=answer_text, outcomes=outcomes)
    monkeypatch.setattr("googleaisearch2api.app.DuckHttpClient", fake)
    return fake


def _install_fake_duck_probe(monkeypatch, ok_by_server: dict[str, bool]) -> None:
    def fake_probe(config, *, timeout_s: float = 8.0) -> FastProxyProbeResult:
        return FastProxyProbeResult(ok=ok_by_server.get(config.browser_proxy_server, False))

    monkeypatch.setattr("googleaisearch2api.app.probe_duck_http_fast", fake_probe)


def test_duck_http_engine_routes_to_http_client_when_enabled(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.DUCK_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "duck")
        _set_duck_engine(test_app, "http")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck browser answer.")
        duck_client = _install_fake_duck_http_client(monkeypatch, answer_text="Duck HTTP answer.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck HTTP answer."
    assert duck_pool.prompts == []
    assert len(duck_client.prompts) == 1
    assert "Question" in duck_client.prompts[0]
    assert "natural language" in duck_client.prompts[0].lower()
    assert duck_client.models == [None]
    assert duck_client.proxies == [None]
    assert duck_client.timeouts == [45.0]
    assert recent[0].engine == "duck"
    assert recent[0].status == "ok"


def test_duck_http_engine_maps_blocked_to_502(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.DUCK_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "duck")
        _set_duck_engine(test_app, "http")
        duck_client = _install_fake_duck_http_client(
            monkeypatch,
            outcomes=[
                DuckHttpBlockedError("Duck.ai refused the solved challenge (HTTP 418)."),
                DuckHttpBlockedError("Duck.ai refused the solved challenge (HTTP 418)."),
                DuckHttpBlockedError("Duck.ai refused the solved challenge (HTTP 418)."),
            ],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "Duck.ai challenge rejected"
    assert len(duck_client.prompts) == 3


def test_duck_http_engine_maps_rate_limited_to_503_and_opens_circuit(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.DUCK_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "duck")
        _set_duck_engine(test_app, "http")
        duck_client = _install_fake_duck_http_client(
            monkeypatch,
            outcomes=[DuckHttpRateLimitedError("Duck.ai rate limited the chat request.")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        cooldown = test_app.state.services.duck_circuit.remaining_seconds()

    assert response.status_code == 503
    assert response.json()["detail"] == "Duck.ai rate limited"
    assert cooldown > 0
    assert len(duck_client.prompts) == 1


def test_duck_http_engine_maps_runtime_error_to_502(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.DUCK_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "duck")
        _set_duck_engine(test_app, "http")
        duck_client = _install_fake_duck_http_client(
            monkeypatch,
            outcomes=[
                DuckHttpRuntimeError("Duck.ai chat request failed (HTTP 500)."),
                DuckHttpRuntimeError("Duck.ai chat request failed (HTTP 500)."),
                DuckHttpRuntimeError("Duck.ai chat request failed (HTTP 500)."),
            ],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )

    assert response.status_code == 502
    assert "Duck.ai http failed" in response.json()["detail"]
    assert len(duck_client.prompts) == 3


def test_duck_engine_browser_fallback_keeps_browser_pool(test_app) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "duck")
        _set_duck_engine(test_app, "browser")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck browser answer.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck browser answer."
    assert len(duck_pool.prompts) == 1
    assert recent[0].engine == "duck"
    assert recent[0].status == "ok"


def test_duck_http_engine_uses_warp_exit_when_configured(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.DUCK_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        test_app.state.services.duck_warp_pool = GeminiWarpPool(["socks5h://warpplus-us:1080"])
        _set_search_engine(test_app, "duck")
        _set_duck_engine(test_app, "http")
        _install_fake_duck_probe(monkeypatch, {"socks5h://warpplus-us:1080": True})
        duck_client = _install_fake_duck_http_client(monkeypatch)
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )

    assert response.status_code == 200
    assert duck_client.proxies[0] == {
        "http": "socks5h://warpplus-us:1080",
        "https": "socks5h://warpplus-us:1080",
    }


def test_duck_http_engine_rotates_sticky_session_on_block(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.DUCK_RETRY_DELAY_SEC", 0.0)
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
        test_app.state.services.proxy_session_store.mark_duck_canary_success(first.id)
        test_app.state.services.proxy_session_store.mark_duck_canary_success(second.id)
        _set_duck_engine(test_app, "http")
        duck_client = _install_fake_duck_http_client(
            monkeypatch,
            answer_text="Duck recovered.",
            outcomes=[DuckHttpBlockedError("Duck.ai refused the solved challenge (HTTP 418).")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck recovered."
    assert len(duck_client.prompts) == 2
    assert "openai.user1" in duck_client.proxies[0]["http"]
    assert "openai.user2" in duck_client.proxies[1]["http"]


def test_duck_warp_pool_uses_extended_cooldown(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("API_TOKEN", "secret-token")
    monkeypatch.setenv("BROWSER_WORKERS", "1")
    monkeypatch.setenv("REQUEST_QUEUE_SIZE", "1")
    monkeypatch.setenv("PROXY_AUTO_RECOVERY_ENABLED", "false")
    monkeypatch.setenv("PROXY_AUTO_RECOVERY_RUN_ON_STARTUP", "false")
    monkeypatch.setenv("GEMINI_WARP_PROXIES", "socks5h://warpplus-us:1080,socks5h://warpplus-gb:1080")
    monkeypatch.setenv("DUCK_WARP_PROXIES", "")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app):
        services = app.state.services
        try:
            assert services.duck_warp_pool is not None
            assert services.gemini_warp_pool is not None
            assert services.duck_warp_pool._cooldown_sec == 1800.0
            assert services.gemini_warp_pool._cooldown_sec == 300.0
        finally:
            services.pool.close()
            services.duck_pool.close()
    get_settings.cache_clear()


def test_run_duck_http_promotes_cold_pool_candidate_via_fast_probe(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("API_TOKEN", "secret-token")
    monkeypatch.setenv("BROWSER_WORKERS", "1")
    monkeypatch.setenv("REQUEST_QUEUE_SIZE", "1")
    monkeypatch.setenv("PROXY_AUTO_RECOVERY_ENABLED", "false")
    monkeypatch.setenv("PROXY_AUTO_RECOVERY_RUN_ON_STARTUP", "false")
    monkeypatch.setenv("GEMINI_WARP_PROXIES", "")
    monkeypatch.setenv("ANSWER_TIMEOUT_MS", "45000")
    monkeypatch.setenv("RESIN_STICKY_SESSION_ENABLED", "true")
    monkeypatch.setenv("BROWSER_PROXY_SERVER", "http://192.0.2.1:2260")
    monkeypatch.setenv("BROWSER_PROXY_USERNAME", "openai")
    monkeypatch.setenv("BROWSER_PROXY_PASSWORD", "proxy-pass")
    monkeypatch.setenv("DUCK_ENGINE", "http")
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        services = app.state.services
        _set_search_engine(app, "duck")
        services.proxy_session_store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name="user1",
            proxy_username="openai.user1",
            status="cooldown",
        )
        services.proxy_session_store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name="user2",
            proxy_username="openai.user2",
            status="cooldown",
        )

        def fake_probe(config, *, timeout_s: float = 8.0):
            return FastProxyProbeResult(ok=config.browser_proxy_username == "openai.user2")

        monkeypatch.setattr("googleaisearch2api.app.probe_duck_http_fast", fake_probe)
        duck_client = _install_fake_duck_http_client(monkeypatch, answer_text="Duck ok.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )

    assert response.status_code == 200
    assert "openai.user2" in duck_client.proxies[0]["http"]
