from __future__ import annotations

from googleaisearch2api.config import ServiceConfig
from googleaisearch2api.egress import _build_launch_kwargs


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
