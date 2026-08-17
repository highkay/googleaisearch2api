from __future__ import annotations

from googleaisearch2api.config import ServiceConfig
from googleaisearch2api.fast_proxy_probe import probe_duck_http_fast

DUCK_STATUS_URL = "https://duck.ai/duckchat/v1/status"
EGRESS_URL = "https://api.ipify.org?format=json"


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


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


def _install_fake_probe_client(monkeypatch, session: _FakeSession) -> _FakeSession:
    monkeypatch.setattr(
        "googleaisearch2api.fast_proxy_probe._probe_client",
        lambda impersonate, session_arg: (session, False),
    )
    return session


def _challenge_response(challenge: str | None) -> _FakeResponse:
    headers = {"x-vqd-hash-1": challenge} if challenge is not None else {}
    return _FakeResponse(200, headers=headers)


def test_probe_duck_http_fast_ok_when_status_200_with_challenge(monkeypatch) -> None:
    session = _install_fake_probe_client(
        monkeypatch,
        _FakeSession(
            {
                DUCK_STATUS_URL: _challenge_response("dmVyaWZpZWQtY2hhbGxlbmdl"),
                EGRESS_URL: _FakeResponse(200, text='{"ip":"198.51.100.9"}'),
            }
        ),
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_duck_http_fast(config)

    assert result.ok is True
    assert result.reason is None
    assert session.calls[0] == DUCK_STATUS_URL
    assert result.raw.get("egress_ip") == "198.51.100.9"


def test_probe_duck_http_fast_reports_rate_limited_on_429(monkeypatch) -> None:
    session = _install_fake_probe_client(
        monkeypatch,
        _FakeSession({DUCK_STATUS_URL: _FakeResponse(429, text="rate limited")}),
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_duck_http_fast(config)

    assert result.ok is False
    assert result.reason == "duck probe rate limited"
    assert session.calls == [DUCK_STATUS_URL]


def test_probe_duck_http_fast_fails_when_challenge_header_missing(monkeypatch) -> None:
    _install_fake_probe_client(
        monkeypatch,
        _FakeSession({DUCK_STATUS_URL: _challenge_response(None)}),
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_duck_http_fast(config)

    assert result.ok is False
    assert result.reason == "duck probe: challenge missing"


def test_probe_duck_http_fast_fails_on_transport_error(monkeypatch) -> None:
    _install_fake_probe_client(
        monkeypatch,
        _FakeSession({DUCK_STATUS_URL: TimeoutError("connection timed out")}),
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_duck_http_fast(config)

    assert result.ok is False
    assert "TimeoutError" in (result.reason or "")


def test_probe_duck_http_fast_never_solves_challenge(monkeypatch) -> None:
    # Status-only contract: the probe must never exercise the wasmrt solver.
    def exploding_solve(challenge: str) -> str:
        raise AssertionError("probe must not solve the x-vqd challenge")

    monkeypatch.setattr("googleaisearch2api.duck_solver.solve_vqd", exploding_solve)
    _install_fake_probe_client(
        monkeypatch,
        _FakeSession(
            {
                DUCK_STATUS_URL: _challenge_response("bmV2ZXItc29sdmVk"),
                EGRESS_URL: _FakeResponse(200, text='{"ip":"198.51.100.9"}'),
            }
        ),
    )
    config = ServiceConfig(browser_proxy_server="http://Default:x@127.0.0.1:2260")
    result = probe_duck_http_fast(config)

    assert result.ok is True


def test_probe_duck_http_fast_ok_false_when_unconfigured() -> None:
    config = ServiceConfig()
    result = probe_duck_http_fast(config)

    assert result.ok is False
    assert result.reason == "proxy is not configured"
