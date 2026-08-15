from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from googleaisearch2api.app import create_app
from googleaisearch2api.config import (
    DEFAULT_API_TOKEN,
    ServiceConfig,
    ServiceConfigUpdate,
    get_settings,
)
from googleaisearch2api.duck_ai import DuckAiTimeoutError
from googleaisearch2api.gemini_upstream import GeminiUpstreamRuntimeError
from googleaisearch2api.gemini_web import (
    GeminiWebBlockedError,
    GeminiWebRateLimitedError,
    GeminiWebRuntimeError,
)
from googleaisearch2api.schemas import Citation, GoogleAiResult


def _build_settings_form(**overrides: str) -> dict[str, str]:
    payload = {
        "default_model": "google-search",
        "search_engine": "gemini",
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


class FakeProxyAutoRecovery:
    def __init__(self) -> None:
        self.reasons: list[str] = []
        self.running = False

    def trigger_async(self, *, reason: str = "event") -> bool:
        self.reasons.append(reason)
        return True

    def is_running(self) -> bool:
        return self.running

    def status(self) -> dict[str, object]:
        return {"enabled": True, "running": self.running}

    def close(self) -> None:
        pass


class FakeGeminiClient:
    """Stand-in for `app.GeminiWebClient` (pure HTTP, no browser pool)."""

    def __init__(
        self,
        answer_text: str = "Gemini answer.",
        outcomes: list[GoogleAiResult | Exception] | None = None,
    ) -> None:
        self.answer_text = answer_text
        self.outcomes = list(outcomes or [])
        self.prompts: list[str] = []
        self.proxies: list[dict[str, str] | None] = []
        self.timeouts: list[float] = []
        self.models: list[str] = []
        self.cookies: list[str | None] = []
        self.sapisids: list[str | None] = []

    def __call__(self, *, timeout_s: float = 20.0) -> FakeGeminiClient:
        self.timeouts.append(timeout_s)
        return self

    def run(
        self,
        prompt: str,
        *,
        model: str = "gemini-3.7-flash",
        cookie: str | None = None,
        sapisid: str | None = None,
        session=None,
        proxies: dict[str, str] | None = None,
    ) -> GoogleAiResult:
        self.prompts.append(prompt)
        self.models.append(model)
        self.cookies.append(cookie)
        self.sapisids.append(sapisid)
        self.proxies.append(proxies)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return GoogleAiResult(
            answer_text=self.answer_text,
            citations=[Citation(title="Gemini source", url="https://example.com")],
            final_url="https://gemini.google.com/app",
            page_title="Gemini",
        )


def _install_fake_gemini_client(
    app,
    monkeypatch,
    answer_text: str = "Gemini answer.",
    outcomes: list[GoogleAiResult | Exception] | None = None,
) -> FakeGeminiClient:
    fake = FakeGeminiClient(answer_text=answer_text, outcomes=outcomes)
    monkeypatch.setattr("googleaisearch2api.app.GeminiWebClient", fake)
    return fake


class FakeGeminiUpstreamClient:
    """Stand-in for `app.GeminiUpstreamClient` (pure HTTP, no browser pool)."""

    def __init__(
        self,
        answer_text: str = "Gemini upstream answer.",
        outcomes: list[tuple[str, list[Citation]] | Exception] | None = None,
    ) -> None:
        self.answer_text = answer_text
        self.outcomes = list(outcomes or [])
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> FakeGeminiUpstreamClient:
        self.kwargs.append(kwargs)
        return self

    def run(self, prompt: str, *, model: str | None = None) -> tuple[str, list[Citation]]:
        self.prompts.append(prompt)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return self.answer_text, [
            Citation(title="Upstream source", url="https://upstream.example.com")
        ]


def _install_fake_gemini_upstream_client(
    app,
    monkeypatch,
    answer_text: str = "Gemini upstream answer.",
    outcomes: list[tuple[str, list[Citation]] | Exception] | None = None,
) -> FakeGeminiUpstreamClient:
    fake = FakeGeminiUpstreamClient(answer_text=answer_text, outcomes=outcomes)
    monkeypatch.setattr("googleaisearch2api.app.GeminiUpstreamClient", fake)
    return fake


def _set_gemini_upstream_base_url(
    app,
    monkeypatch,
    base_url: str | None,
) -> None:
    store = app.state.services.store
    original_get_config = store.get_config

    def patched_get_config() -> ServiceConfig:
        return original_get_config().model_copy(update={"gemini_upstream_base_url": base_url})

    monkeypatch.setattr(store, "get_config", patched_get_config)


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
    # Isolate tests from developer .env (recovery startup + real proxy would race SQLite).
    monkeypatch.setenv("PROXY_AUTO_RECOVERY_ENABLED", "false")
    monkeypatch.setenv("PROXY_AUTO_RECOVERY_RUN_ON_STARTUP", "false")
    monkeypatch.setenv("BROWSER_PROXY_SERVER", "")
    monkeypatch.setenv("BROWSER_PROXY_USERNAME", "")
    monkeypatch.setenv("BROWSER_PROXY_PASSWORD", "")
    monkeypatch.setenv("RESIN_STICKY_SESSION_ENABLED", "false")
    # Isolate auto routing from any developer-configured local gateway.
    monkeypatch.setenv("GEMINI_UPSTREAM_BASE_URL", "")
    monkeypatch.setenv("GEMINI_UPSTREAM_API_KEY", "")
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


def test_console_settings_persist_and_preserve_gemini_upstream_fields(test_app) -> None:
    with TestClient(test_app) as client:
        client.post(
            "/console/login",
            data={"console_token": "secret-token", "next": "/console"},
            follow_redirects=False,
        )

        save_response = client.post(
            "/console/settings",
            data=_build_settings_form(
                gemini_upstream_base_url="https://api.example.com/v1",
                gemini_upstream_api_key="sk-upstream-key",
                gemini_upstream_model="gemini-3.0-pro",
            ),
            follow_redirects=False,
        )
        saved = test_app.state.services.store.get_config()

        preserve_response = client.post(
            "/console/settings",
            data=_build_settings_form(
                gemini_upstream_base_url="",
                gemini_upstream_api_key="",
                # FastAPI swaps an empty-string Form value for its default, so
                # the blank-model fallback path is exercised with whitespace.
                gemini_upstream_model="  ",
            ),
            follow_redirects=False,
        )
        preserved = test_app.state.services.store.get_config()

    assert save_response.status_code == 303
    assert saved.gemini_upstream_base_url == "https://api.example.com/v1"
    assert saved.gemini_upstream_api_key == "sk-upstream-key"
    assert saved.gemini_upstream_model == "gemini-3.0-pro"
    assert preserve_response.status_code == 303
    assert preserved.gemini_upstream_base_url == "https://api.example.com/v1"
    assert preserved.gemini_upstream_api_key == "sk-upstream-key"
    assert preserved.gemini_upstream_model == "gemini-3.0-pro"


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


def test_query_post_returns_tool_friendly_response_shape(test_app, monkeypatch) -> None:
    with TestClient(test_app) as client:
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Tool friendly answer.",
        )
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
    assert google_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    assert "natural language" in gemini_client.prompts[0].lower()
    assert "Use verified facts only." in gemini_client.prompts[0]
    assert "Question" in gemini_client.prompts[0]
    assert recent[0].endpoint == "/query"
    assert recent[0].engine == "gemini"


def test_query_gemini_engine_dispatches_to_gemini_client(test_app, monkeypatch) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "gemini")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Gemini answer.",
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Gemini answer."
    assert payload["citations"][0]["url"] == "https://example.com"
    assert google_pool.prompts == []
    assert duck_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    assert "natural language" in gemini_client.prompts[0].lower()
    assert gemini_client.models == ["gemini-3.7-flash"]
    assert gemini_client.cookies == [None]
    assert gemini_client.sapisids == [None]
    assert gemini_client.proxies == [None]
    assert recent[0].engine == "gemini"
    assert recent[0].status == "ok"


def test_gemini_engine_retries_on_block_then_succeeds(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "gemini")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Gemini recovered.",
            outcomes=[GeminiWebBlockedError("Gemini web IP-blocked (test).")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=2)

    assert response.status_code == 200
    assert response.json()["answer"] == "Gemini recovered."
    assert len(gemini_client.prompts) == 2
    assert len(gemini_client.models) == 2
    assert [record.engine for record in recent] == ["gemini", "gemini"]
    assert [record.status for record in recent] == ["ok", "error"]


def test_gemini_engine_uses_sticky_proxy_selection(test_app, monkeypatch) -> None:
    with TestClient(test_app) as client:
        test_app.state.services.store.update_config(
            ServiceConfigUpdate(
                default_model="google-search",
                search_engine="gemini",
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
        test_app.state.services.proxy_session_store.mark_canary_success(snapshot.id)
        gemini_client = _install_fake_gemini_client(test_app, monkeypatch)
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert gemini_client.proxies[0] is not None
    assert "openai.user1" in gemini_client.proxies[0]["http"]
    assert "openai.user1" in gemini_client.proxies[0]["https"]
    assert recent[0].engine == "gemini"
    assert recent[0].proxy_username == "openai.user1"
    assert recent[0].status == "ok"


def test_gemini_engine_falls_back_to_base_proxy_when_no_session(test_app, monkeypatch) -> None:
    with TestClient(test_app) as client:
        test_app.state.services.store.update_config(
            ServiceConfigUpdate(
                default_model="google-search",
                search_engine="gemini",
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
        gemini_client = _install_fake_gemini_client(test_app, monkeypatch)
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert gemini_client.proxies[0] is not None
    assert gemini_client.proxies[0]["http"].startswith("http://openai:")
    assert ".user" not in gemini_client.proxies[0]["http"]
    assert recent[0].engine == "gemini"
    assert recent[0].proxy_username is None
    assert recent[0].status == "ok"


def test_query_engine_gemini_upstream_dispatches(test_app, monkeypatch) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "gemini-upstream")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck answer.")
        upstream_client = _install_fake_gemini_upstream_client(
            test_app,
            monkeypatch,
            answer_text="Gemini upstream answer.",
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Gemini upstream answer."
    assert payload["citations"][0]["url"] == "https://upstream.example.com"
    assert google_pool.prompts == []
    assert duck_pool.prompts == []
    assert len(upstream_client.prompts) == 1
    assert "natural language" in upstream_client.prompts[0].lower()
    assert len(upstream_client.kwargs) == 1
    assert upstream_client.kwargs[0]["base_url"] == ""
    assert upstream_client.kwargs[0]["api_key"] == ""
    assert upstream_client.kwargs[0]["model"] == "gemini-3.7-flash"
    assert recent[0].engine == "gemini-upstream"
    assert recent[0].status == "ok"


def test_query_auto_uses_gemini_in_process_when_upstream_configured(test_app, monkeypatch) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        _set_gemini_upstream_base_url(test_app, monkeypatch, "http://127.0.0.1:8081")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Gemini answer.",
        )
        upstream_client = _install_fake_gemini_upstream_client(
            test_app,
            monkeypatch,
            answer_text="Gemini upstream answer.",
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert response.json()["answer"] == "Gemini answer."
    assert google_pool.prompts == []
    assert duck_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    assert upstream_client.prompts == []
    assert upstream_client.kwargs == []
    assert recent[0].engine == "gemini"
    assert recent[0].status == "ok"


def test_query_auto_falls_back_to_duck_when_gemini_fails_despite_upstream_configured(
    test_app,
    monkeypatch,
) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        _set_gemini_upstream_base_url(test_app, monkeypatch, "http://127.0.0.1:8081")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck fallback.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            outcomes=[
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
            ],
        )
        upstream_client = _install_fake_gemini_upstream_client(
            test_app,
            monkeypatch,
            outcomes=[GeminiUpstreamRuntimeError("gateway unavailable (test).")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=4)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck fallback."
    assert google_pool.prompts == []
    assert upstream_client.prompts == []
    assert len(gemini_client.prompts) == 3
    assert len(duck_pool.prompts) == 1
    assert "Question" in duck_pool.prompts[0]
    assert [record.engine for record in recent] == ["duck", "gemini", "gemini", "gemini"]
    assert [record.status for record in recent] == ["ok", "error", "error", "error"]


def test_query_auto_uses_gemini_chain_when_upstream_not_configured(
    test_app,
    monkeypatch,
) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Gemini answer.",
        )
        upstream_client = _install_fake_gemini_upstream_client(test_app, monkeypatch)
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert response.json()["answer"] == "Gemini answer."
    assert google_pool.prompts == []
    assert duck_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    assert upstream_client.kwargs == []
    assert upstream_client.prompts == []
    assert recent[0].engine == "gemini"


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
    assert len(duck_pool.prompts) == 1
    assert "Question" in duck_pool.prompts[0]
    assert "natural language" in duck_pool.prompts[0].lower()
    assert duck_pool.blocked_retry_counts == [0]
    assert recent[0].engine == "duck"


def test_query_duck_engine_resets_duck_pool_after_timeout(test_app) -> None:
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "duck")
        duck_pool = _install_fake_duck_pool(
            test_app,
            outcomes=[DuckAiTimeoutError("Duck.ai chat input did not become ready.")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )

    assert response.status_code == 504
    assert duck_pool.reset_calls == 1


def test_query_duck_engine_retries_another_sticky_session_after_timeout(test_app) -> None:
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
        duck_pool = _install_fake_duck_pool(
            test_app,
            answer_text="Duck recovered.",
            outcomes=[DuckAiTimeoutError("Duck.ai chat input did not become ready.")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=2)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck recovered."
    assert [config.browser_proxy_username for config in duck_pool.configs] == [
        "openai.user1",
        "openai.user2",
    ]
    assert duck_pool.reset_calls == 1
    assert [record.status for record in recent] == ["ok", "error"]


def test_query_auto_falls_back_to_duck_when_gemini_is_rate_limited(
    test_app,
    monkeypatch,
) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            outcomes=[
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
            ],
        )
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck fallback.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=4)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck fallback."
    assert google_pool.prompts == []
    assert len(gemini_client.prompts) == 3
    assert len(duck_pool.prompts) == 1
    assert "Question" in duck_pool.prompts[0]
    assert "natural language" in duck_pool.prompts[0].lower()
    assert [record.engine for record in recent] == ["duck", "gemini", "gemini", "gemini"]
    assert [record.status for record in recent] == ["ok", "error", "error", "error"]


def test_query_auto_falls_back_to_duck_when_gemini_is_blocked(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            outcomes=[
                GeminiWebBlockedError("Gemini web IP-blocked (test)."),
                GeminiWebBlockedError("Gemini web IP-blocked (test)."),
                GeminiWebBlockedError("Gemini web IP-blocked (test)."),
            ],
        )
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck fallback.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=4)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck fallback."
    assert google_pool.prompts == []
    assert len(gemini_client.prompts) == 3
    assert "natural language" in gemini_client.prompts[0].lower()
    assert len(duck_pool.prompts) == 1
    assert "Question" in duck_pool.prompts[0]
    assert "natural language" in duck_pool.prompts[0].lower()
    assert [record.engine for record in recent] == ["duck", "gemini", "gemini", "gemini"]
    assert [record.status for record in recent] == ["ok", "error", "error", "error"]


def test_query_auto_merges_gemini_duck_errors_when_all_fail(
    test_app,
    monkeypatch,
) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        _install_fake_gemini_client(
            test_app,
            monkeypatch,
            outcomes=[
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
            ],
        )
        _install_fake_duck_pool(
            test_app,
            outcomes=[DuckAiTimeoutError("Duck.ai chat input did not become ready.")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=4)

    assert response.status_code == 504
    detail = response.json()["detail"]
    assert "Google" not in detail
    assert "Gemini web" in detail
    assert "Duck.ai" in detail
    assert google_pool.prompts == []
    assert [record.engine for record in recent] == ["duck", "gemini", "gemini", "gemini"]
    assert [record.status for record in recent] == ["error", "error", "error", "error"]


def test_query_auto_retries_sticky_sessions_before_duck_fallback(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
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
        third = test_app.state.services.proxy_session_store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name="user3",
            proxy_username="openai.user3",
            status="active",
        )
        test_app.state.services.proxy_session_store.mark_canary_success(first.id)
        test_app.state.services.proxy_session_store.mark_canary_success(second.id)
        test_app.state.services.proxy_session_store.mark_canary_success(third.id)
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            outcomes=[
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
                GeminiWebRateLimitedError("Gemini web is rate limited (test)."),
            ],
        )
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck fallback.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=4)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck fallback."
    assert google_pool.prompts == []
    assert len(gemini_client.prompts) == 3
    proxy_urls = [p["http"] for p in gemini_client.proxies]
    assert len(proxy_urls) == 3
    assert "openai.user1" in proxy_urls[0]
    assert "openai.user2" in proxy_urls[1]
    assert "openai.user3" in proxy_urls[2]
    assert len(duck_pool.prompts) == 1
    assert "Question" in duck_pool.prompts[0]
    assert "natural language" in duck_pool.prompts[0].lower()
    assert [record.engine for record in recent] == ["duck", "gemini", "gemini", "gemini"]
    assert [record.status for record in recent] == ["ok", "error", "error", "error"]


def test_query_auto_routes_directly_to_duck_when_sticky_pool_is_empty(test_app) -> None:
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
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_pool = _install_fake_duck_pool(test_app, answer_text="Duck direct fallback.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert response.json()["answer"] == "Duck direct fallback."
    assert google_pool.prompts == []
    assert len(duck_pool.prompts) == 1
    assert "Question" in duck_pool.prompts[0]
    assert "natural language" in duck_pool.prompts[0].lower()
    assert recent[0].engine == "duck"
    assert recent[0].status == "ok"


def test_query_auto_triggers_recovery_when_sticky_pool_is_empty(test_app) -> None:
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
        recovery = FakeProxyAutoRecovery()
        test_app.state.services.proxy_auto_recovery = recovery
        _install_fake_pool(test_app, answer_text="Google answer.")
        _install_fake_duck_pool(test_app, answer_text="Duck direct fallback.")
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )

    assert response.status_code == 200
    assert recovery.reasons == ["auto-pool-empty"]


def test_query_auto_falls_back_to_duck_when_gemini_answer_quality_fails(
    test_app,
    monkeypatch,
) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            outcomes=[
                GoogleAiResult(
                    answer_text="You said: User request:\nQuestion",
                    citations=[Citation(title="Gemini source", url="https://example.com")],
                    final_url="https://gemini.google.com/app",
                    page_title="Gemini",
                )
            ],
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
    assert google_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    assert len(duck_pool.prompts) == 1
    assert "Question" in duck_pool.prompts[0]
    assert "natural language" in duck_pool.prompts[0].lower()
    assert [record.engine for record in recent] == ["duck", "gemini"]
    assert [record.status for record in recent] == ["ok", "error"]
    assert "quality check" in (recent[1].error_message or "")


def test_query_auto_falls_back_to_duck_when_gemini_answer_is_empty(
    test_app,
    monkeypatch,
) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        duck_answer = (
            "1. 中微公司：刻蚀设备受益于先进制程扩产和国产替代。"
            "2. 北方华创：薄膜沉积、刻蚀和清洗设备覆盖关键环节。"
            "3. 沪硅产业：大硅片需求可能随先进制程供应链景气提升。"
            "4. 雅克科技：电子特气和前驱体材料与晶圆制造资本开支相关。"
            "5. 华海清科：CMP 设备环节受益于先进节点工艺复杂度提升。"
        )
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            outcomes=[
                GoogleAiResult(
                    answer_text="",
                    citations=[],
                    final_url="https://gemini.google.com/app",
                    page_title="Gemini",
                )
            ],
        )
        duck_pool = _install_fake_duck_pool(test_app, answer_text=duck_answer)
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={
                "model": "google-search",
                "query": "台积电 3nm 涨价 AI A股 受益股 OR 供应链 OR 半导体 最多返回 5 条",
            },
        )
        recent = test_app.state.services.store.list_recent_requests(limit=2)

    assert response.status_code == 200
    assert response.json()["answer"] == duck_answer
    assert google_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    assert len(duck_pool.prompts) == 1
    assert "台积电" in duck_pool.prompts[0]
    assert "最多返回 5 条" not in duck_pool.prompts[0]
    assert "请用自然语言完整回答" in duck_pool.prompts[0]
    assert [record.engine for record in recent] == ["duck", "gemini"]
    assert [record.status for record in recent] == ["ok", "error"]
    assert "empty answer" in (recent[1].error_message or "")


def test_query_simplifies_json_results_prompt_for_natural_language_answer(
    test_app,
    monkeypatch,
) -> None:
    prompt = (
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"","published_date":"YYYY-MM-DD"}]}。'
        '若找不到足够直接相关的结果，返回 {"results": []}。'
        "问题：PingAn 000001.SZ 最新公告 新闻 催化 风险 最多返回 5 条"
    )
    google_answer = (
        "平安银行近期公告重点包括经营数据和股东大会相关事项。"
        "来源可关注公司公告、交易所公告和证券时报等公开渠道。"
        "如果需要更精确的催化和风险，应继续核对交易所公告原文、公司投资者关系记录"
        "以及主流财经媒体报道，避免把转载或旧新闻当成最新事项。"
        "风险侧重点包括息差压力、资产质量波动和宏观需求变化；催化侧重点包括业绩披露、"
        "分红政策和监管口径变化。"
    )
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(test_app)
        duck_pool = _install_fake_duck_pool(test_app)
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text=google_answer,
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": prompt},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert response.json()["answer"] == google_answer
    assert google_pool.prompts == []
    assert duck_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    gemini_prompt = gemini_client.prompts[0]
    assert "PingAn 000001.SZ 最新公告 新闻 催化 风险" in gemini_prompt
    assert '{"results"' not in gemini_prompt
    assert "最多返回 5 条" not in gemini_prompt
    assert "请用自然语言" in gemini_prompt
    assert [record.engine for record in recent] == ["gemini"]
    assert [record.status for record in recent] == ["ok"]


def test_query_auto_naturalizes_empty_json_results_for_simplified_prompt(
    test_app,
    monkeypatch,
) -> None:
    prompt = (
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"","published_date":"YYYY-MM-DD"}]}。'
        '若找不到足够直接相关的结果，返回 {"results": []}。'
        "问题：PingAn 000001.SZ 最新公告 新闻 催化 风险 最多返回 5 条"
    )
    with TestClient(test_app) as client:
        _set_search_engine(test_app, "auto")
        google_pool = _install_fake_pool(test_app)
        duck_pool = _install_fake_duck_pool(test_app)
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text='{"results": []}',
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": prompt},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert "没有找到足够直接相关" in response.json()["answer"]
    assert google_pool.prompts == []
    assert duck_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    assert [record.engine for record in recent] == ["gemini"]
    assert [record.status for record in recent] == ["ok"]


def test_query_uses_active_sticky_proxy_session_when_enabled(test_app, monkeypatch) -> None:
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
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Sticky answer.",
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert google_pool.prompts == []
    assert "openai.user1" in gemini_client.proxies[0]["http"]
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
            for item in test_app.state.services.proxy_session_store.list_proxy_sessions(limit=10)
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


def test_query_reselects_sticky_proxy_session_after_gemini_block(test_app, monkeypatch) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
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
        test_app.state.services.proxy_session_store.mark_canary_success(first.id)
        test_app.state.services.proxy_session_store.mark_canary_success(second.id)
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Recovered answer.",
            outcomes=[GeminiWebBlockedError("Gemini web IP-blocked (test).")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=2)
        sessions = {
            item.proxy_username: item
            for item in test_app.state.services.proxy_session_store.list_proxy_sessions(limit=10)
        }

    assert response.status_code == 200
    assert response.json()["answer"] == "Recovered answer."
    assert google_pool.prompts == []
    proxy_urls = [p["http"] for p in gemini_client.proxies]
    assert len(proxy_urls) == 2
    assert "openai.user1" in proxy_urls[0]
    assert "openai.user2" in proxy_urls[1]

    requests_by_proxy = {record.proxy_username: record for record in recent}
    assert requests_by_proxy["openai.user1"].status == "error"
    assert requests_by_proxy["openai.user2"].status == "ok"
    assert sessions["openai.user1"].status == "cooldown"
    assert sessions["openai.user1"].request_block_count == 1
    assert sessions["openai.user2"].status == "active"
    assert sessions["openai.user2"].request_success_count == 1


def test_query_reselects_sticky_proxy_session_after_gemini_runtime_error(
    test_app,
    monkeypatch,
) -> None:
    monkeypatch.setattr("googleaisearch2api.app.GEMINI_RETRY_DELAY_SEC", 0.0)
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
        test_app.state.services.proxy_session_store.mark_canary_success(first.id)
        test_app.state.services.proxy_session_store.mark_canary_success(second.id)
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Recovered answer.",
            outcomes=[GeminiWebRuntimeError("Gemini web failed (test).")],
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=2)
        sessions = {
            item.proxy_username: item
            for item in test_app.state.services.proxy_session_store.list_proxy_sessions(limit=10)
        }

    assert response.status_code == 200
    assert response.json()["answer"] == "Recovered answer."
    assert google_pool.prompts == []
    proxy_urls = [p["http"] for p in gemini_client.proxies]
    assert len(proxy_urls) == 2
    assert "openai.user1" in proxy_urls[0]
    assert "openai.user2" in proxy_urls[1]
    requests_by_proxy = {record.proxy_username: record for record in recent}
    assert requests_by_proxy["openai.user1"].status == "error"
    assert requests_by_proxy["openai.user2"].status == "ok"
    assert sessions["openai.user1"].status == "active"
    assert sessions["openai.user1"].request_error_count == 1
    assert sessions["openai.user2"].request_success_count == 1


def test_query_default_engine_falls_back_to_base_proxy_without_active_session(
    test_app,
    monkeypatch,
) -> None:
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
        google_pool = _install_fake_pool(test_app)
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Base proxy answer.",
        )
        response = client.post(
            "/query",
            headers=_auth_headers(),
            json={"model": "google-search", "query": "Question"},
        )
        recent = test_app.state.services.store.list_recent_requests(limit=1)

    assert response.status_code == 200
    assert response.json()["answer"] == "Base proxy answer."
    assert google_pool.prompts == []
    assert gemini_client.proxies[0] is not None
    assert ".user" not in gemini_client.proxies[0]["http"]
    assert recent[0].status == "ok"
    assert recent[0].resin_sticky_session_enabled is True
    assert recent[0].proxy_username is None


def test_query_get_returns_tool_friendly_response_shape(test_app, monkeypatch) -> None:
    with TestClient(test_app) as client:
        google_pool = _install_fake_pool(test_app, answer_text="Google answer.")
        gemini_client = _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="GET answer.",
        )
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
    assert google_pool.prompts == []
    assert len(gemini_client.prompts) == 1
    assert "Question" in gemini_client.prompts[0]


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


def test_query_stream_returns_simple_sse_events(test_app, monkeypatch) -> None:
    with TestClient(test_app) as client:
        _install_fake_pool(test_app, answer_text="Google answer.")
        _install_fake_gemini_client(
            test_app,
            monkeypatch,
            answer_text="Streaming answer.",
        )
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


def test_healthz_reports_stuck_workers(test_app) -> None:
    with TestClient(test_app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stuck_workers"] == 0
    assert payload["poisoned_workers"] == 0
