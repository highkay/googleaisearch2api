"""Tests for the pure-HTTP Gemini web fallback (StreamGenerate protocol)."""

from __future__ import annotations

import json
import re

import pytest

from googleaisearch2api.gemini_web import (
    GeminiWebBlockedError,
    GeminiWebClient,
    GeminiWebRateLimitedError,
    GeminiWebRuntimeError,
    build_f_req,
    build_inner_json,
    build_stream_generate_url,
    extract_answer_text,
    extract_citations,
    parse_response_frames,
)


def _make_source(url: str, title: str) -> list[object]:
    return [url, title, "https://encrypted-tbn0.gstatic.com/favicon", "snippet"]


def _make_chunk(sources: list[object]) -> list[object]:
    return [None, None, sources]


def _make_frame(texts: list[str], sources: list[object]) -> str:
    chunks = [_make_chunk(sources)]
    node: list[object] = [None, texts, [None, chunks]]
    inner: list[object] = [None] * 80
    inner[4] = [node]
    first: list[object] = ["wrb.fr", None, json.dumps(inner)]
    return json.dumps([first])


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        text: str,
        url: str = "https://gemini.google.com/app",
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.url = url
        self.headers: dict[str, str] = {}


class _FakeSession:
    def __init__(
        self,
        get_response: _FakeResponse | Exception | None = None,
        post_response: _FakeResponse | Exception | None = None,
    ) -> None:
        self.get_response = get_response
        self.post_response = post_response
        self.get_calls: list[tuple[str, dict]] = []
        self.post_calls: list[tuple[str, dict, dict, dict]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.get_calls.append((url, kwargs))
        if isinstance(self.get_response, Exception):
            raise self.get_response
        assert self.get_response is not None
        return self.get_response

    def post(
        self,
        url: str,
        data: dict | None = None,
        headers: dict | None = None,
        **kwargs: object,
    ) -> _FakeResponse:
        self.post_calls.append((url, data or {}, headers or {}, kwargs))
        if isinstance(self.post_response, Exception):
            raise self.post_response
        assert self.post_response is not None
        return self.post_response

    def close(self) -> None:
        return None


def test_build_stream_generate_url_has_bl_hl_reqid_rt() -> None:
    url = build_stream_generate_url(bl="boq_assistant-bard-web-server_20260716.08_p0")
    assert "StreamGenerate" in url
    assert "bl=boq_assistant-bard-web-server_20260716.08_p0" in url
    assert "hl=en" in url
    assert "rt=c" in url
    assert "_reqid=" in url


def test_build_inner_json_uses_102_slot_non_temporary_chat() -> None:
    inner = build_inner_json("hello")
    assert len(inner) == 102
    assert inner[41] == [2]
    assert inner[45] is None
    assert inner[79] == 1  # gemini-3.7-flash -> FAST mode
    assert inner[17] == [[0]]
    assert inner[0] == ["hello", 0, None, None, None, None, 0]
    assert inner[1] == ["en"]


def test_build_inner_json_applies_model_think_and_extra_fields() -> None:
    inner = build_inner_json("hi", model_id=3, think_mode=2, extra_fields={31: 2, 80: 3})
    assert inner[79] == 3
    assert inner[17] == [[2]]
    assert inner[31] == 2
    assert inner[80] == 3


def test_build_f_req_wraps_inner_json_string() -> None:
    f_req = build_f_req("hello world")
    outer = json.loads(f_req)
    assert outer[0] is None
    inner = json.loads(outer[1])  # outer[1] is a JSON *string*, not a list
    assert isinstance(inner, list)
    assert len(inner) == 102
    assert inner[0] == ["hello world", 0, None, None, None, None, 0]
    assert inner[1] == ["en"]


def test_parse_response_frames_splits_lines_and_returns_lists() -> None:
    raw = ")]}'\n" + _make_frame(["hello"], [_make_source("https://a.com/x", "A")]) + "\nnot-json\n"
    frames = parse_response_frames(raw)
    assert len(frames) == 1
    assert isinstance(frames[0], list)


def test_extract_answer_text_returns_last_nonempty() -> None:
    raw = _make_frame(["hello"], []) + "\n" + _make_frame(["hello world", "final answer"], [])
    frames = parse_response_frames(raw)
    assert extract_answer_text(frames) == "final answer"


def test_extract_citations_maps_url_title_strips_fragment_dedupes() -> None:
    sources = [
        _make_source("https://en.wikipedia.org/wiki/X#:~:text=abc", "X - Wikipedia"),
        _make_source("https://en.wikipedia.org/wiki/X#:~:text=abc", "X - Wikipedia"),
        _make_source("https://example.com/y", "Y"),
    ]
    raw = _make_frame(["answer"], sources)
    citations = extract_citations(parse_response_frames(raw))
    assert len(citations) == 2
    assert citations[0].url == "https://en.wikipedia.org/wiki/X"
    assert citations[0].title == "X - Wikipedia"
    assert citations[1].url == "https://example.com/y"


def test_blocked_when_sorry_redirect() -> None:
    client = GeminiWebClient()
    session = _FakeSession(
        get_response=_FakeResponse(200, "boq_assistant-bard-web-server_20260716.08_p0"),
        post_response=_FakeResponse(200, "", url="https://www.google.com/sorry/index?continue=abc"),
    )
    with pytest.raises(GeminiWebBlockedError):
        client.run("hi", session=session)


def test_rate_limited_when_empty_body() -> None:
    client = GeminiWebClient()
    session = _FakeSession(
        get_response=_FakeResponse(200, "boq_assistant-bard-web-server_20260716.08_p0"),
        post_response=_FakeResponse(200, ""),
    )
    with pytest.raises(GeminiWebRateLimitedError):
        client.run("hi", session=session)


def test_run_posts_f_req_and_returns_result_with_citations() -> None:
    client = GeminiWebClient()
    session = _FakeSession(
        get_response=_FakeResponse(200, "boq_assistant-bard-web-server_20260716.08_p0"),
        post_response=_FakeResponse(
            200, _make_frame(["the answer"], [_make_source("https://a.com/1", "A")])
        ),
    )
    result = client.run("what is x?", session=session)
    assert result.answer_text == "the answer"
    assert result.citations[0].url == "https://a.com/1"
    assert session.post_calls[0][1]["f.req"]


def test_no_frames_raises_runtime_error() -> None:
    client = GeminiWebClient()
    session = _FakeSession(
        get_response=_FakeResponse(200, "boq_assistant-bard-web-server_20260716.08_p0"),
        post_response=_FakeResponse(
            200, "some plain text that is long enough but is not a frame at all ..."
        ),
    )
    with pytest.raises(GeminiWebRuntimeError):
        client.run("hi", session=session)


def test_resolve_model_maps_names_to_modes() -> None:
    from googleaisearch2api.gemini_web_models import resolve_model

    name, mode, think, error, extra = resolve_model("gemini-3.7-flash")
    assert name == "gemini-3.7-flash"
    assert mode == 1
    assert think == 0
    assert error is None
    assert extra is None
    assert resolve_model("gemini-3.5-flash-thinking")[1] == 2
    assert resolve_model("gemini-3.1-pro")[1] == 3
    name, mode, think, error, extra = resolve_model("gemini-3.7-flash@think=2")
    assert name == "gemini-3.7-flash"
    assert think == 2
    assert error is None


def test_resolve_model_falls_back_on_unknown_and_keeps_enhanced_extra() -> None:
    from googleaisearch2api.gemini_web_models import resolve_model

    name, mode, think, error, extra = resolve_model("totally-unknown-model")
    assert name == "gemini-3.7-flash"
    assert mode == 1
    assert error is None
    assert extra is None
    name, mode, think, error, extra = resolve_model("gemini-3.1-pro-enhanced")
    assert mode == 3
    assert extra == {31: 2, 80: 3}


def test_is_block_response_detects_status_and_markers() -> None:
    from googleaisearch2api.gemini_web import is_block_response

    assert is_block_response(403, None, b"") is True
    assert is_block_response(200, None, b"unusual traffic from your computer network") is True
    assert is_block_response(200, None, b"normal answer") is False


def test_make_sapisidhash_format() -> None:
    from googleaisearch2api.gemini_web import make_sapisidhash

    assert re.fullmatch(r"SAPISIDHASH \d+_[0-9a-f]{40}", make_sapisidhash("abc"))


def test_build_headers_adds_auth_fields_only_when_present() -> None:
    from googleaisearch2api.gemini_web import _build_headers

    headers = _build_headers()
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert headers["Origin"] == "https://gemini.google.com"
    assert headers["Referer"] == "https://gemini.google.com/app"
    assert headers["X-Same-Domain"] == "1"
    assert "User-Agent" in headers
    assert "Cookie" not in headers
    assert "Authorization" not in headers

    with_auth = _build_headers(cookie="SID=x", sapisid="abc")
    assert with_auth["Cookie"] == "SID=x"
    assert with_auth["Authorization"].startswith("SAPISIDHASH ")


def test_run_raises_blocked_on_302_redirect_to_sorry() -> None:
    client = GeminiWebClient()
    session = _FakeSession(
        get_response=_FakeResponse(200, "boq_assistant-bard-web-server_20260716.08_p0"),
        post_response=_FakeResponse(302, "", url="https://www.google.com/sorry/index"),
    )
    with pytest.raises(GeminiWebBlockedError):
        client.run("hi", session=session)
