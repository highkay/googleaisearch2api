from __future__ import annotations

from patchright.sync_api import Error as PatchrightError

from googleaisearch2api.config import ServiceConfig
from googleaisearch2api.egress import _build_launch_kwargs, _probe_endpoint


class _FailingPage:
    def goto(self, endpoint: str, *, wait_until: str, timeout: int) -> None:
        raise PatchrightError("net::ERR_TUNNEL_CONNECTION_FAILED")


class _JsonBody:
    def inner_text(self, *, timeout: int) -> str:
        assert timeout == 5_000
        return '{"ip":"203.0.113.10","org":"AS64500 Example Network"}'


class _JsonPage:
    def goto(self, endpoint: str, *, wait_until: str, timeout: int) -> None:
        assert endpoint == "https://ipinfo.io/json"
        assert wait_until == "domcontentloaded"
        assert timeout == 1_000

    def locator(self, selector: str) -> _JsonBody:
        assert selector == "body"
        return _JsonBody()


def test_egress_endpoint_failure_returns_error_payload() -> None:
    payload = _probe_endpoint(_FailingPage(), "https://ipinfo.io/json", 1_000)

    assert "error" in payload
    assert "ERR_TUNNEL_CONNECTION_FAILED" in payload["error"]


def test_egress_endpoint_parses_json_payload() -> None:
    payload = _probe_endpoint(_JsonPage(), "https://ipinfo.io/json", 1_000)

    assert payload == {"ip": "203.0.113.10", "org": "AS64500 Example Network"}


def test_egress_wraps_authenticated_socks_proxy_with_local_http_bridge() -> None:
    config = ServiceConfig(
        browser_proxy_server="socks5h://openai:secret@192.168.1.18:2260",
        browser_proxy_bypass="localhost,127.0.0.1",
    )

    launch_kwargs, bridge = _build_launch_kwargs(config)
    try:
        proxy = launch_kwargs["proxy"]
        assert proxy["server"].startswith("http://127.0.0.1:")
        assert proxy["bypass"] == "localhost,127.0.0.1"
        assert "username" not in proxy
        assert "password" not in proxy
    finally:
        assert bridge is not None
        bridge.stop()


def test_egress_keeps_http_proxy_credentials_in_launch_kwargs() -> None:
    config = ServiceConfig(
        browser_proxy_server="http://openai:secret@192.168.1.18:2260",
        browser_proxy_bypass="localhost,127.0.0.1",
    )

    launch_kwargs, bridge = _build_launch_kwargs(config)

    assert bridge is None
    assert launch_kwargs["proxy"] == {
        "server": "http://192.168.1.18:2260",
        "username": "openai",
        "password": "secret",
        "bypass": "localhost,127.0.0.1",
    }
