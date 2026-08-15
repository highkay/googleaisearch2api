from __future__ import annotations

from googleaisearch2api.config import ServiceConfig
from googleaisearch2api.fast_proxy_probe import (
    build_proxy_url,
    probe_gemini_http_fast,
    probe_proxy_http_fast,
)
from googleaisearch2api.gemini_web import (
    GeminiWebBlockedError,
    GeminiWebRateLimitedError,
    GeminiWebRuntimeError,
)
from googleaisearch2api.schemas import GoogleAiResult


class _FakeResponse:
    def __init__(self, status_code: int, text: str, url: str = "https://example.com") -> None:
        self.status_code = status_code
        self.text = text
        self.url = url


class _FakeSession:
    def __init__(self, responses: dict[str, _FakeResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> _FakeResponse:
        self.calls.append(url)
        payload = self.responses[url]
        if isinstance(payload, Exception):
            raise payload
        return payload

    def close(self) -> None:
        return None


def test_build_proxy_url_injects_sticky_username() -> None:
    config = ServiceConfig(
        browser_proxy_server="http://Default:secret@192.168.1.18:2260",
        browser_proxy_username="Default.user12",
    )
    assert (
        build_proxy_url(config)
        == "http://Default.user12:secret@192.168.1.18:2260"
    )


def test_probe_proxy_http_fast_rejects_google_block_shell() -> None:
    session = _FakeSession(
        {
            "https://api.ipify.org?format=json": _FakeResponse(
                200, '{"ip":"203.0.113.10"}'
            ),
            "https://api64.ipify.org?format=json": _FakeResponse(
                200, '{"ip":"2001:db8::1"}'
            ),
            "https://www.google.com/search?udm=50&aep=11&hl=en&q=ping": _FakeResponse(
                200,
                "Our systems have detected unusual traffic from your computer network.",
            ),
        }
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_proxy_http_fast(config, session=session)

    assert result.ok is False
    assert result.google_blocked is True
    assert "203.0.113.10" in result.ips
    assert "2001:db8::1" in result.ips


def test_probe_proxy_http_fast_accepts_clean_proxy() -> None:
    session = _FakeSession(
        {
            "https://api.ipify.org?format=json": _FakeResponse(
                200, '{"ip":"198.51.100.7"}'
            ),
            "https://api64.ipify.org?format=json": _FakeResponse(
                200, '{"ip":"198.51.100.7"}'
            ),
            "https://www.google.com/search?udm=50&aep=11&hl=en&q=ping": _FakeResponse(
                200,
                "<html><body>Search results without bot markers</body></html>",
            ),
        }
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_proxy_http_fast(config, session=session)

    assert result.ok is True
    assert result.primary_ip == "198.51.100.7"
    assert result.google_blocked is False


def test_probe_proxy_http_fast_allows_enablejs_shell_for_browser_canary() -> None:
    # Pure HTTP to the AI search URL normally returns the enablejs retry shell
    # even for healthy sticky exits; L0 must not treat that as an IP block.
    session = _FakeSession(
        {
            "https://api.ipify.org?format=json": _FakeResponse(
                200, '{"ip":"203.0.113.55"}'
            ),
            "https://api64.ipify.org?format=json": _FakeResponse(
                200, '{"ip":"203.0.113.55"}'
            ),
            "https://www.google.com/search?udm=50&aep=11&hl=en&q=ping": _FakeResponse(
                200,
                (
                    '<!DOCTYPE html><html><body><noscript>'
                    '<meta content="0;url=/httpservice/retry/enablejs?sei=abc" '
                    'http-equiv="refresh">'
                    "Please click here if you are not redirected within a few seconds."
                    "</noscript></body></html>"
                ),
            ),
        }
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_proxy_http_fast(config, session=session)

    assert result.ok is True
    assert result.google_blocked is False
    assert result.primary_ip == "203.0.113.55"


def test_probe_proxy_http_fast_still_rejects_sorry_interstitial() -> None:
    session = _FakeSession(
        {
            "https://api.ipify.org?format=json": _FakeResponse(
                200, '{"ip":"203.0.113.77"}'
            ),
            "https://api64.ipify.org?format=json": _FakeResponse(
                200, '{"ip":"203.0.113.77"}'
            ),
            "https://www.google.com/search?udm=50&aep=11&hl=en&q=ping": _FakeResponse(
                200,
                "https://www.google.com/sorry/index?continue=https://www.google.com/",
                url="https://www.google.com/sorry/index?continue=search",
            ),
        }
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_proxy_http_fast(config, session=session)

    assert result.ok is False
    assert result.google_blocked is True


def _install_fake_gemini_probe(monkeypatch, outcome: GoogleAiResult | Exception) -> None:
    class _FakeGeminiClient:
        def __init__(self, timeout_s: float) -> None:
            self.timeout_s = timeout_s

        def run(self, prompt: str, **kwargs: object) -> GoogleAiResult:
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr(
        "googleaisearch2api.fast_proxy_probe.GeminiWebClient", _FakeGeminiClient
    )


def test_probe_gemini_http_fast_accepts_valid_request(monkeypatch) -> None:
    _install_fake_gemini_probe(
        monkeypatch,
        GoogleAiResult(
            answer_text="pong",
            citations=[],
            final_url="https://gemini.google.com/app",
            page_title="",
        ),
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_gemini_http_fast(config)

    assert result.ok is True
    assert result.reason is None


def test_probe_gemini_http_fast_rejects_block(monkeypatch) -> None:
    _install_fake_gemini_probe(monkeypatch, GeminiWebBlockedError("blocked"))
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_gemini_http_fast(config)

    assert result.ok is False
    assert result.google_blocked is True
    assert "blocked" in result.reason


def test_probe_gemini_http_fast_rejects_rate_limited(monkeypatch) -> None:
    _install_fake_gemini_probe(monkeypatch, GeminiWebRateLimitedError("rate limited"))
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_gemini_http_fast(config)

    assert result.ok is False
    assert "rate limited" in result.reason


def test_probe_gemini_http_fast_ok_false_when_unconfigured() -> None:
    config = ServiceConfig()
    result = probe_gemini_http_fast(config)

    assert result.ok is False
    assert result.reason == "proxy is not configured"


def test_probe_gemini_http_fast_rejects_runtime_error(monkeypatch) -> None:
    _install_fake_gemini_probe(monkeypatch, GeminiWebRuntimeError("timeout"))
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_gemini_http_fast(config)

    assert result.ok is False
    assert "timeout" in result.reason
