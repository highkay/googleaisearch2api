"""Tests for the OpenAI-compatible gemini-upstream gateway client (pure HTTP)."""

from __future__ import annotations

import pytest

from googleaisearch2api.gemini_upstream import (
    GeminiUpstreamClient,
    GeminiUpstreamRuntimeError,
    extract_inline_citations,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self, post_response: _FakeResponse | Exception) -> None:
        self.post_response = post_response
        self.post_calls: list[tuple[str, dict[str, object]]] = []
        self.close_calls = 0

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.post_calls.append((url, kwargs))
        if isinstance(self.post_response, Exception):
            raise self.post_response
        return self.post_response

    def close(self) -> None:
        self.close_calls += 1


def _install_fake_session(monkeypatch, session: _FakeSession) -> None:
    monkeypatch.setattr("curl_cffi.requests.Session", lambda: session)


def _ok_payload(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def test_extract_inline_citations_dedupes_skips_non_http_and_falls_back_title() -> None:
    text = (
        "[A](https://a.com) [A dup](https://a.com) [B](https://b.com) "
        "[not-http](ftp://c.com) [   ](https://d.com) [ spaced title ](https://e.com)"
    )
    citations = extract_inline_citations(text)
    assert [c.url for c in citations] == [
        "https://a.com",
        "https://b.com",
        "https://d.com",
        "https://e.com",
    ]
    assert citations[0].title == "A"
    assert citations[2].title == "https://d.com"
    assert citations[3].title == "spaced title"


def test_extract_inline_citations_returns_empty_for_plain_text() -> None:
    assert extract_inline_citations("no links here") == []


def test_run_builds_correct_url_headers_and_body(monkeypatch) -> None:
    session = _FakeSession(_FakeResponse(200, _ok_payload("ok answer")))
    _install_fake_session(monkeypatch, session)
    client = GeminiUpstreamClient(
        base_url="http://127.0.0.1:8081/",
        api_key="sk-key",
        timeout_s=12.5,
        model="gemini-3.7-flash",
    )
    answer, citations = client.run("hello")
    assert answer == "ok answer"
    assert citations == []
    assert session.close_calls == 1
    url, kwargs = session.post_calls[0]
    assert url == "http://127.0.0.1:8081/v1/chat/completions"
    assert kwargs["headers"] == {
        "Authorization": "Bearer sk-key",
        "Content-Type": "application/json",
    }
    assert kwargs["json"] == {
        "model": "gemini-3.7-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert kwargs["timeout"] == 12.5


def test_run_omits_authorization_header_without_api_key_and_honors_model_override(
    monkeypatch,
) -> None:
    session = _FakeSession(_FakeResponse(200, _ok_payload("answer")))
    _install_fake_session(monkeypatch, session)
    client = GeminiUpstreamClient(base_url="http://127.0.0.1:8081")
    answer, _ = client.run("hello", model="gemini-2.5-pro")
    assert answer == "answer"
    _, kwargs = session.post_calls[0]
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["json"]["model"] == "gemini-2.5-pro"


def test_run_extracts_and_dedupes_citations_from_answer(monkeypatch) -> None:
    content = (
        "Answer with sources: [First](https://a.com/1) and [Second](https://b.com/2) "
        "and [First again](https://a.com/1) and [Fallback](https://c.com/3)."
    )
    session = _FakeSession(_FakeResponse(200, _ok_payload(content)))
    _install_fake_session(monkeypatch, session)
    client = GeminiUpstreamClient(base_url="http://127.0.0.1:8081")
    answer, citations = client.run("what is x?")
    assert answer == content
    assert [(c.title, c.url) for c in citations] == [
        ("First", "https://a.com/1"),
        ("Second", "https://b.com/2"),
        ("Fallback", "https://c.com/3"),
    ]


def test_run_raises_when_upstream_returns_error_body(monkeypatch) -> None:
    session = _FakeSession(_FakeResponse(200, {"error": {"message": "model overloaded"}}))
    _install_fake_session(monkeypatch, session)
    client = GeminiUpstreamClient(base_url="http://127.0.0.1:8081")
    with pytest.raises(GeminiUpstreamRuntimeError, match="model overloaded"):
        client.run("hi")
    assert session.close_calls == 1


def test_run_raises_with_status_when_non_200(monkeypatch) -> None:
    session = _FakeSession(_FakeResponse(503, {}))
    _install_fake_session(monkeypatch, session)
    client = GeminiUpstreamClient(base_url="http://127.0.0.1:8081")
    with pytest.raises(GeminiUpstreamRuntimeError, match="HTTP 503"):
        client.run("hi")


def test_run_raises_when_response_has_no_choices(monkeypatch) -> None:
    session = _FakeSession(_FakeResponse(200, {"choices": []}))
    _install_fake_session(monkeypatch, session)
    client = GeminiUpstreamClient(base_url="http://127.0.0.1:8081")
    with pytest.raises(GeminiUpstreamRuntimeError, match="choices"):
        client.run("hi")


def test_run_wraps_network_exception_in_runtime_error(monkeypatch) -> None:
    session = _FakeSession(RuntimeError("connection refused"))
    _install_fake_session(monkeypatch, session)
    client = GeminiUpstreamClient(base_url="http://127.0.0.1:8081")
    with pytest.raises(GeminiUpstreamRuntimeError, match="connection refused") as excinfo:
        client.run("hi")
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert session.close_calls == 1


def test_extract_inline_citations_captures_bare_urls_and_markdown() -> None:
    text = (
        "Fact one [Source A](https://a.com/x#:~:text=z). "
        "Fact two bare https://b.com/y. "
        "Trailing https://c.com/z, and https://a.com/x again."
    )
    citations = extract_inline_citations(text)
    assert [c.url for c in citations] == [
        "https://a.com/x",
        "https://b.com/y",
        "https://c.com/z",
    ]
    assert citations[0].title == "Source A"
    assert citations[1].title == "https://b.com/y"


def test_extract_inline_citations_dedupes_and_normalizes() -> None:
    citations = extract_inline_citations("See https://d.com/1. and [Title](https://d.com/1#frag).")
    assert len(citations) == 1
    assert citations[0].url == "https://d.com/1"
