from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from patchright.sync_api import sync_playwright

from .browser import DEFAULT_BROWSER_CHANNEL, resolve_browser_proxy, resolve_browser_user_agent
from .config import ServiceConfig
from .proxy_bridge import (
    LocalSocksProxyBridge,
    build_socks_proxy_target,
    is_socks_proxy_server,
)
from .proxy_sessions import normalize_ip_vector

EGRESS_ENDPOINTS = (
    "https://api.ipify.org?format=json",
    "https://api64.ipify.org?format=json",
    "https://ipinfo.io/json",
)


@dataclass(frozen=True, slots=True)
class EgressProbeResult:
    ips: list[str]
    primary_ip: str | None
    asn: str | None = None
    organization: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


def _parse_body_json(body_text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(body_text.strip())
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _extract_asn(org: str | None) -> str | None:
    if not org:
        return None
    first = org.split(maxsplit=1)[0].strip()
    if first.upper().startswith("AS"):
        return first
    return None


def _build_launch_kwargs(
    config: ServiceConfig,
) -> tuple[dict[str, Any], LocalSocksProxyBridge | None]:
    kwargs: dict[str, Any] = {
        "headless": config.browser_headless,
        "channel": DEFAULT_BROWSER_CHANNEL,
    }
    browser_user_agent = resolve_browser_user_agent(config)
    if browser_user_agent:
        kwargs["args"] = [f"--user-agent={browser_user_agent}"]
    browser_proxy = resolve_browser_proxy(config)
    if browser_proxy:
        proxy_server = browser_proxy["server"]
        if is_socks_proxy_server(proxy_server):
            target = build_socks_proxy_target(
                proxy_server,
                username=browser_proxy.get("username"),
                password=browser_proxy.get("password"),
            )
            bridge = LocalSocksProxyBridge(target)
            bridge.start()
            kwargs["proxy"] = {
                "server": bridge.server_url,
                "bypass": browser_proxy.get("bypass"),
            }
            return kwargs, bridge
        kwargs["proxy"] = browser_proxy
    return kwargs, None


def probe_egress(config: ServiceConfig, *, timeout_ms: int = 30_000) -> EgressProbeResult:
    raw: dict[str, Any] = {}
    ips: list[str] = []
    organization: str | None = None
    asn: str | None = None

    launch_kwargs, bridge = _build_launch_kwargs(config)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(**launch_kwargs)
            context = browser.new_context(locale=config.browser_locale)
            page = context.new_page()
            try:
                for endpoint in EGRESS_ENDPOINTS:
                    page.goto(endpoint, wait_until="domcontentloaded", timeout=timeout_ms)
                    body = page.locator("body").inner_text(timeout=5_000)
                    payload = _parse_body_json(body)
                    raw[endpoint] = payload or {"body": body[:500]}
                    ip = payload.get("ip")
                    if isinstance(ip, str):
                        ips.append(ip)
                    org = payload.get("org")
                    if isinstance(org, str) and org:
                        organization = org
                        asn = _extract_asn(org)
            finally:
                context.close()
                browser.close()
    finally:
        if bridge is not None:
            bridge.stop()

    vector = normalize_ip_vector(ips)
    return EgressProbeResult(
        ips=vector,
        primary_ip=vector[0] if vector else None,
        asn=asn,
        organization=organization,
        raw=raw,
    )
