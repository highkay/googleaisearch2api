"""Tests for the pure-HTTP Duck.ai engine (duck_http).

Given: a fake curl_cffi session that serves the /duckchat/v1/status challenge
header and a canned SSE chat stream (mirrors tests/test_gemini_web.py harness).
When: DuckHttpClient.run resolves + solves the challenge, POSTs the chat request
with the solved x-vqd-hash-1, and parses the `data:` SSE events.
Then: the result is a GoogleAiResult; 418/challenge failures raise
DuckHttpBlockedError, 429/rate-limit text raises DuckHttpRateLimitedError, and
empty/parse failures raise DuckHttpRuntimeError.
"""

from __future__ import annotations

import base64
import json
import re

import pytest

import googleaisearch2api.duck_http as dh
from googleaisearch2api.duck_ai import (
    DUCK_RATE_LIMIT_MARKERS,
    _is_rate_limited_text,
    _normalize_text,
    build_duck_search_prompt,
    extract_duck_answer_text,
)
from googleaisearch2api.duck_http import (
    DUCK_AI_URL,
    DUCK_CHAT_ENDPOINT,
    DUCK_STATUS_ENDPOINT,
    DuckHttpBlockedError,
    DuckHttpClient,
    DuckHttpRateLimitedError,
    DuckHttpRuntimeError,
    build_chat_body,
    build_chat_headers,
    parse_sse_events,
)
from googleaisearch2api.duck_solver import CHROME_USER_AGENT

DUCK_HOME_URL = DUCK_AI_URL
_FE_VERSION = "duckai-v1.2.3-deadbeef"
_HOME_HTML = (
    '<html><head><script src="/x.js" data-version-tag="duckai-v1.2.3" '
    'data-version-sha="deadbeef"></script></head></html>'
)

_CHALLENGE_SOURCE = "(async function(){return 1;})()"
_CHALLENGE_B64 = base64.b64encode(_CHALLENGE_SOURCE.encode()).decode()


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        text: str = "",
        url: str = "https://duck.ai/",
        headers: dict | None = None,
        lines: list[str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._text = text
        self.url = url
        self.headers = headers if headers is not None else {}
        self._lines = lines

    @property
    def text(self) -> str:
        return self._text or "\n".join(self._lines or [])

    def iter_lines(self) -> list[str]:
        return self._lines or []


class _FakeSession:
    def __init__(
        self,
        *,
        get_responses: dict[str, _FakeResponse] | None = None,
        post_response: _FakeResponse | Exception | None = None,
    ) -> None:
        self.get_responses = get_responses or {}
        self.post_response = post_response
        self.get_calls: list[tuple[str, dict]] = []
        self.post_calls: list[tuple[str, dict, dict, dict]] = []
        self.closed = False

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.get_calls.append((url, kwargs))
        if url not in self.get_responses:  # pragma: no cover - test setup error
            raise AssertionError(f"no fake get response registered for {url}")
        response = self.get_responses[url]
        if isinstance(response, Exception):
            raise response
        return response

    def post(
        self,
        url: str,
        json: object | None = None,
        data: object | None = None,
        headers: dict | None = None,
        **kwargs: object,
    ) -> _FakeResponse:
        self.post_calls.append((url, json if json is not None else data, headers or {}, kwargs))
        if isinstance(self.post_response, Exception):
            raise self.post_response
        assert self.post_response is not None
        return self.post_response

    def close(self) -> None:
        self.closed = True


def _status_response(challenge: str | None = _CHALLENGE_B64, status: int = 200) -> _FakeResponse:
    headers = {"x-vqd-hash-1": challenge} if challenge else {}
    return _FakeResponse(status, url=DUCK_STATUS_ENDPOINT, headers=headers)


def _solve_stub(monkeypatch) -> list[str]:
    """Monkeypatch the solver seam; returns the decoded-challenge capture list."""
    calls: list[str] = []
    monkeypatch.setattr(
        dh, "solve_vqd", lambda challenge: calls.append(challenge) or "solved-token"
    )
    monkeypatch.setattr(dh, "_cached_fe_version", None)
    return calls


def _session_for_run(
    *,
    post_status: int = 200,
    post_text: str = "",
    post_lines: list[str] | None = None,
) -> _FakeSession:
    return _FakeSession(
        get_responses={
            DUCK_STATUS_ENDPOINT: _status_response(),
            DUCK_HOME_URL: _FakeResponse(200, text=_HOME_HTML),
        },
        post_response=_FakeResponse(post_status, text=post_text, lines=post_lines),
    )


# ---- pure helpers ------------------------------------------------------------


def test_parse_sse_events_strips_data_prefix_maps_json_and_stops_at_done() -> None:
    raw = (
        'data: {"action":"message","message":"hello"}\n'
        "data: [DONE]\n"
        'data: {"action":"message","message":"ignored"}\n'
    )
    events = parse_sse_events(raw)
    assert [event["message"] for event in events] == ["hello"]


def test_parse_sse_events_skips_non_data_lines_and_bad_json() -> None:
    raw = 'event: ping\ndata: not-json\ndata: {"action":"success","message":null}\n'
    events = parse_sse_events(raw)
    assert len(events) == 1
    assert events[0]["action"] == "success"


def test_build_chat_headers_replicate_duck2api_post_chat_set() -> None:
    headers = build_chat_headers("solved-token", fe_version=_FE_VERSION)
    assert headers["accept"] == "text/event-stream"
    assert headers["accept-language"] == "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"
    assert headers["content-type"] == "application/json"
    assert headers["origin"] == "https://duck.ai"
    assert headers["referer"] == "https://duck.ai/"
    assert headers["user-agent"] == CHROME_USER_AGENT
    assert "149" in headers["sec-ch-ua"]
    assert headers["sec-ch-ua-mobile"] == "?0"
    assert headers["priority"] == "u=1, i"
    assert headers["x-vqd-hash-1"] == "solved-token"
    assert headers["x-fe-version"] == _FE_VERSION
    assert re.fullmatch(r"[0-9a-f]{32}", headers["x-ddg-journey-id"])
    signals = json.loads(base64.b64decode(headers["x-fe-signals"]))
    assert set(signals) == {"start", "events", "end"}


def test_build_chat_headers_omits_fe_version_when_unknown() -> None:
    headers = build_chat_headers("solved-token", fe_version=None)
    assert "x-fe-version" not in headers
    assert headers["x-vqd-hash-1"] == "solved-token"


def test_build_fe_signals_replays_onboarding_event_chain() -> None:
    signals = base64.b64decode(dh.build_fe_signals(now_ms=1000))
    payload = json.loads(signals)
    assert payload["start"] + payload["end"] == 1000  # start = now_ms - end
    deltas = [event["delta"] for event in payload["events"]]
    assert [event["name"] for event in payload["events"]] == [
        "onboarding_impression",
        "action",
        "onboarding_finish",
        "startNewChat_free",
    ]
    assert deltas == sorted(deltas)  # strictly increasing event chain
    assert payload["events"][1]["trusted"] is True


def test_build_chat_body_has_observed_required_fields() -> None:
    body = build_chat_body("hello", "gpt-5.4-mini")
    assert body["model"] == "gpt-5.4-mini"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["metadata"]["toolChoice"] == {
        "NewsSearch": False,
        "VideosSearch": False,
        "LocalSearch": False,
        "WeatherForecast": False,
    }
    assert body["canUseTools"] is True
    assert body["reasoningEffort"] == "none"
    assert body["canUseApproxLocation"] is None
    assert body["canDelegateImageGeneration"] is None
    durable = body["durableStream"]
    assert durable["messageId"] == "" and durable["conversationId"] == ""
    assert durable["publicKey"]["alg"] == "" and durable["publicKey"]["kty"] == ""


def test_fe_version_scrape_caches_and_falls_back_to_omitted() -> None:
    class _HomeResponse:
        text = _HOME_HTML

    class _Client:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, url: str, **kwargs: object) -> _HomeResponse:
            self.calls.append(url)
            return _HomeResponse()

    client = _Client()
    assert dh._resolve_fe_version(client, None) == _FE_VERSION
    assert dh._resolve_fe_version(client, None) == _FE_VERSION
    assert len(client.calls) == 1  # cached


def test_fe_version_scrape_fails_gracefully() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(dh, "_cached_fe_version", None)
    try:
        assert dh._resolve_fe_version(_BrokenClient(), None) is None
    finally:
        monkeypatch.undo()


class _BrokenClient:
    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        raise OSError("down")


# ---- client.run error classification -----------------------------------------


def test_run_418_raises_blocked(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(post_status=418, post_text='{"action":"error","status":418}')
    with pytest.raises(DuckHttpBlockedError):
        client.run("Say hello in 5 words.", session=session)


def test_run_429_raises_rate_limited(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(post_status=429, post_text='{"error":"too many"}')
    with pytest.raises(DuckHttpRateLimitedError):
        client.run("Say hello in 5 words.", session=session)


def test_run_rate_limit_body_text_raises_rate_limited(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(
        post_status=400,
        post_text="whoa there, too many requests — please take a short break and try again later",
    )
    with pytest.raises(DuckHttpRateLimitedError):
        client.run("Say hello in 5 words.", session=session)


def test_run_error_sse_event_418_raises_blocked(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(
        post_lines=['data: {"action":"error","status":418,"type":"ERR_CHALLENGE"}']
    )
    with pytest.raises(DuckHttpBlockedError):
        client.run("Say hello in 5 words.", session=session)


def test_run_error_sse_event_429_raises_rate_limited(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(
        post_lines=['data: {"action":"error","status":429,"type":"ERR_RATE"}']
    )
    with pytest.raises(DuckHttpRateLimitedError):
        client.run("Say hello in 5 words.", session=session)


def test_run_missing_challenge_header_raises_blocked(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _FakeSession(
        get_responses={
            DUCK_STATUS_ENDPOINT: _status_response(challenge=None),
            DUCK_HOME_URL: _FakeResponse(200, text=_HOME_HTML),
        },
        post_response=_FakeResponse(200, lines=["data: [DONE]"]),
    )
    with pytest.raises(DuckHttpBlockedError):
        client.run("Say hello in 5 words.", session=session)


def test_run_solve_failure_raises_blocked(monkeypatch) -> None:
    monkeypatch.setattr(dh, "_cached_fe_version", None)
    monkeypatch.setattr(
        dh, "solve_vqd", lambda challenge: (_ for _ in ()).throw(ValueError("boom"))
    )
    client = DuckHttpClient()
    session = _session_for_run()
    with pytest.raises(DuckHttpBlockedError):
        client.run("Say hello in 5 words.", session=session)


def test_run_status_endpoint_429_raises_rate_limited(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _FakeSession(
        get_responses={DUCK_STATUS_ENDPOINT: _status_response(status=429)},
        post_response=_FakeResponse(200, lines=["data: [DONE]"]),
    )
    with pytest.raises(DuckHttpRateLimitedError):
        client.run("Say hello in 5 words.", session=session)


def test_run_empty_sse_raises_runtime(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(post_lines=["data: [DONE]"])
    with pytest.raises(DuckHttpRuntimeError):
        client.run("Say hello in 5 words.", session=session)


def test_run_other_400_raises_runtime(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(post_status=400, post_text='{"action":"error","status":400}')
    with pytest.raises(DuckHttpRuntimeError):
        client.run("Say hello in 5 words.", session=session)


# ---- happy path --------------------------------------------------------------


def test_run_status_solve_chat_sse_returns_google_ai_result(monkeypatch) -> None:
    solved_calls = _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(
        post_lines=[
            'data: {"id":"msg_1","action":"success","model":"gpt-5.4-mini",'
            '"message":"Partly sunny"}',
            'data: {"id":"msg_1","action":"success","model":"gpt-5.4-mini",'
            '"message":" today"}',
            "data: [CHAT_TITLE:Weather]\n",
            "data: [DONE]",
        ]
    )
    result = client.run("Say hello in 5 words.", session=session)

    # solved baseline: challenge was decoded before being handed to the solver
    assert solved_calls == [_CHALLENGE_SOURCE]

    get_status = session.get_calls[0]
    assert get_status[0] == DUCK_STATUS_ENDPOINT
    assert get_status[1]["headers"]["x-vqd-accept"] == "1"

    post_url, post_body, post_headers, _ = session.post_calls[0]
    assert post_url == DUCK_CHAT_ENDPOINT
    assert post_headers["x-vqd-hash-1"] == "solved-token"
    assert post_headers["x-fe-version"] == _FE_VERSION
    assert post_headers["content-type"] == "application/json"
    assert post_body["messages"][0]["content"] == build_duck_search_prompt(
        "Say hello in 5 words."
    )

    assert "Partly sunny today" in result.answer_text
    assert result.citations == []
    assert result.final_url == "https://duck.ai/"
    assert result.page_title == "Duck.ai"
    assert result.body_excerpt


def test_run_uses_injected_session_without_closing_it(monkeypatch) -> None:
    _solve_stub(monkeypatch)
    client = DuckHttpClient()
    session = _session_for_run(
        post_lines=['data: {"id":"m","action":"success","message":"hi"}\n', "data: [DONE]\n"]
    )
    client.run("Say hello in 5 words.", session=session)
    assert session.closed is False


# ---- re-exports --------------------------------------------------------------


def test_re_exports_duck_ai_pure_helpers() -> None:
    assert dh.build_duck_search_prompt is build_duck_search_prompt
    assert dh.extract_duck_answer_text is extract_duck_answer_text
    assert dh._is_rate_limited_text is _is_rate_limited_text
    assert dh._normalize_text is _normalize_text
    assert dh.DUCK_RATE_LIMIT_MARKERS is DUCK_RATE_LIMIT_MARKERS


def test_re_exported_normalize_and_rate_limit_behave() -> None:
    assert dh._normalize_text("  a\r\nb ") == "a\nb"
    assert dh._is_rate_limited_text("please take a short break and retry") is True