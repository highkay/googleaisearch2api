"""Tests for the pure-HTTP Google AI Mode /async/folif transport.

No live network access: fetch_folif is driven through fake sessions only.
"""

from __future__ import annotations

import pytest

from googleaisearch2api.ai_mode_http import (
    AiModeTokens,
    build_folif_url,
    classify_folif_response,
    extract_answer_from_folif,
    fetch_folif,
)
from googleaisearch2api.config import ServiceConfig

FOLIF_PREFIX = "https://www.google.com/search/async/folif?"

ENABLEJS_SHELL_BODY = (
    "<!DOCTYPE html><html><body><noscript>"
    '<meta content="0;url=/httpservice/retry/enablejs?sei=abc" '
    'http-equiv="refresh">'
    "Please click here if you are not redirected within a few seconds."
    "</noscript></body></html>"
)

SG_REL_BODY = (
    "<html><body><div>emsg=sg_rel: AI Mode unavailable on this exit,"
    " the async endpoint answered with a restricted shell page</div></body></html>"
)

ANSWER_BODY = (
    '<div class="n6owBd" data-mstk="42">'
    "<p>Here is a conservative AI Mode answer fragment for the TTL probe.</p>"
    "</div>"
)


class _FakeResponse:
    def __init__(
        self, status_code: int, text: str, url: str = "https://www.google.com/search"
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.url = url


class _FakeSession:
    """Records get() arguments and returns a canned response (or raises)."""

    def __init__(self, response: _FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[str] = []
        self.last_headers: dict[str, str] = {}
        self.last_proxies: dict[str, str] | None = None
        self.last_timeout: float = 0.0
        self.last_allow_redirects: bool = True

    def get(
        self,
        url: str,
        *,
        proxies: dict[str, str] | None = None,
        timeout: float = 0.0,
        allow_redirects: bool = True,
        headers: dict[str, str] | None = None,
        **_ignored: object,
    ) -> _FakeResponse:
        self.calls.append(url)
        self.last_headers = headers or {}
        self.last_proxies = proxies
        self.last_timeout = timeout
        self.last_allow_redirects = allow_redirects
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self) -> None:
        return None


def test_build_folif_url_includes_all_tokens() -> None:
    tokens = AiModeTokens(
        ei="ei000",
        stkp="stkp111",
        elrc="elrc444",
        xsrf_folif_token="xsrf789",
        srtst="srt123",
        garc="garc456",
        sca_esv="sca222",
        mstk="mstk333",
    )
    assert build_folif_url(tokens, "hello world") == (
        FOLIF_PREFIX
        + "ei=ei000&yv=3&udm=50&hl=en&stkp=stkp111&cs=0&csuir=0&elrc=elrc444"
        + "&q=hello+world&async=_fmt%3Aadl%2C_xsrf%3Axsrf789"
        + "&srtst=srt123&garc=garc456&sca_esv=sca222&mstk=mstk333"
    )


def test_build_folif_url_omits_none_tokens() -> None:
    tokens = AiModeTokens(xsrf_folif_token="xsrf789", stkp="stkp111")
    url = build_folif_url(tokens, "ping")
    assert url == (
        FOLIF_PREFIX
        + "yv=3&udm=50&hl=en&stkp=stkp111&cs=0&csuir=0&q=ping"
        + "&async=_fmt%3Aadl%2C_xsrf%3Axsrf789"
    )
    assert "ei=" not in url
    assert "elrc=" not in url
    assert "srtst=" not in url
    assert "garc=" not in url
    assert "sca_esv=" not in url
    assert "mstk=" not in url


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (AiModeTokens(xsrf_folif_token="xsrf", stkp="stkp"), True),
        (AiModeTokens(xsrf_folif_token="xsrf", stkp=None), False),
        (AiModeTokens(xsrf_folif_token=None, stkp="stkp"), False),
        (AiModeTokens(), False),
    ],
)
def test_is_complete_requires_xsrf_folif_token_and_stkp(
    tokens: AiModeTokens, expected: bool
) -> None:
    assert tokens.is_complete() is expected


@pytest.mark.parametrize(
    "body",
    [
        ENABLEJS_SHELL_BODY,
        SG_REL_BODY,
        "x" * 60 + "sg_rel" + "y" * 60,
    ],
)
def test_classify_returns_shell_for_enablejs_and_sg_rel_bodies(body: str) -> None:
    result = classify_folif_response(200, body, "https://www.google.com/search")
    assert result.kind == "shell"
    assert result.body == body


def test_classify_returns_blocked_for_sorry_redirect() -> None:
    body = "Redirecting you to the sorry interstitial after abuse detection." * 3
    result = classify_folif_response(
        200, body, "https://www.google.com/sorry/index?continue=search"
    )
    assert result.kind == "blocked"
    assert result.final_url == "https://www.google.com/sorry/index?continue=search"


def test_classify_returns_blocked_for_403_status() -> None:
    body = "Access denied by Google for this network." * 4
    result = classify_folif_response(403, body, "https://www.google.com/search")
    assert result.kind == "blocked"
    assert result.status_code == 403


def test_classify_returns_blocked_for_blocked_marker_in_body() -> None:
    body = "Our systems have detected unusual traffic from your computer network." * 2
    result = classify_folif_response(200, body, "https://www.google.com/search")
    assert result.kind == "blocked"


@pytest.mark.parametrize("body", ["", "   \n  ", "x" * 49])
def test_classify_returns_empty_for_short_body(body: str) -> None:
    result = classify_folif_response(200, body, "https://www.google.com/search")
    assert result.kind == "empty"
    assert result.answer_text == ""


@pytest.mark.parametrize(
    "body",
    [
        '<div class="n6owBd">Answer fragment for the n6owBd marker, padded past fifty chars.</div>',
        '<div class="pTRUV">Answer fragment for the pTRUV marker, padded past fifty chars.</div>',
        '<div data-mstk="7">Answer for the data-mstk marker, padded well past fifty chars.</div>',
    ],
)
def test_classify_returns_answer_for_answer_fragment(body: str) -> None:
    result = classify_folif_response(200, body, "https://www.google.com/search")
    assert result.kind == "answer"
    assert result.answer_text == body


def test_extract_answer_from_folif_returns_body_when_data_mstk_present() -> None:
    body = '<div data-mstk="1">answer</div>'
    assert extract_answer_from_folif(body) == body


def test_extract_answer_from_folif_returns_empty_when_marker_absent() -> None:
    assert extract_answer_from_folif("<div>no marker here</div>") == ""


def test_fetch_folif_uses_proxy_and_folif_url_prefix() -> None:
    tokens = AiModeTokens(ei="ei000", stkp="stkp111", elrc="elrc444", xsrf_folif_token="xsrf789")
    session = _FakeSession(_FakeResponse(200, ANSWER_BODY, FOLIF_PREFIX + "q=ping"))
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")

    result = fetch_folif(tokens, "ping", config=config, session=session)

    assert session.calls == [build_folif_url(tokens, "ping")]
    assert session.last_proxies == {
        "http": "http://Default:x@127.0.0.1:2260",
        "https": "http://Default:x@127.0.0.1:2260",
    }
    assert session.last_timeout == 15.0
    assert session.last_allow_redirects is True
    assert session.last_headers["Referer"] == "https://www.google.com/search"
    assert session.last_headers["Accept-Language"] == "en-US,en;q=0.9"
    assert result.kind == "answer"
    assert result.body == ANSWER_BODY
    assert result.answer_text == ANSWER_BODY


def test_fetch_folif_creates_session_with_chrome131_impersonate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_impersonates: list[str] = []

    class _StubSession:
        def __init__(self, impersonate: str) -> None:
            created_impersonates.append(impersonate)

        def get(self, url: str, **_ignored: object) -> _FakeResponse:
            return _FakeResponse(200, ANSWER_BODY, url)

        def close(self) -> None:
            return None

    class _StubRequestsModule:
        Session = _StubSession

    monkeypatch.setattr("curl_cffi.requests", _StubRequestsModule, raising=False)
    tokens = AiModeTokens(xsrf_folif_token="xsrf789", stkp="stkp111")
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")

    result = fetch_folif(tokens, "ping", config=config)

    assert created_impersonates == ["chrome131"]
    assert result.kind == "answer"
    assert result.body == ANSWER_BODY


def test_fetch_folif_network_exception_returns_error_kind() -> None:
    session = _FakeSession(RuntimeError("connection reset by peer"))
    tokens = AiModeTokens(xsrf_folif_token="xsrf789", stkp="stkp111")
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")

    result = fetch_folif(tokens, "ping", config=config, session=session)

    assert session.calls == [build_folif_url(tokens, "ping")]
    assert result.kind == "error"
    assert result.status_code == 0
    assert result.body == ""
    assert result.final_url == ""
