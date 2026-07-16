from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from .browser import resolve_browser_proxy
from .config import ServiceConfig
from .proxy_sessions import normalize_ip_vector

EGRESS_ENDPOINTS = (
    "https://api.ipify.org?format=json",
    "https://api64.ipify.org?format=json",
)
GOOGLE_AI_PROBE_URL = "https://www.google.com/search?udm=50&aep=11&hl=en&q=ping"
DEFAULT_IMPERSONATE = "chrome131"
DEFAULT_TIMEOUT_S = 8.0

BLOCKED_BODY_MARKERS = (
    "unusual traffic",
    "this network is blocked due to unaddressed abuse complaints",
    "malicious behavior",
    "this page checks to see if it's really a human",
    "not a robot",
    "captcha",
    "enablejs",
    "please click here if you are not redirected",
)


@dataclass(frozen=True, slots=True)
class FastProxyProbeResult:
    ok: bool
    reason: str | None = None
    ips: list[str] = field(default_factory=list)
    primary_ip: str | None = None
    google_status: int | None = None
    google_blocked: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_proxy_url(config: ServiceConfig) -> str | None:
    """Build a curl/libcurl proxy URL including sticky username auth."""
    proxy = resolve_browser_proxy(config)
    if not proxy:
        return None
    server = (proxy.get("server") or "").strip()
    if not server:
        return None
    parsed = urlsplit(server)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid proxy server: {server}")

    username = proxy.get("username")
    password = proxy.get("password")
    host = parsed.hostname
    port = parsed.port
    if port is None:
        netloc = host
    elif ":" in host and not host.startswith("["):
        netloc = f"[{host}]:{port}"
    else:
        netloc = f"{host}:{port}"

    if username:
        userinfo = quote(str(username), safe="")
        if password is not None:
            userinfo = f"{userinfo}:{quote(str(password), safe='')}"
        netloc = f"{userinfo}@{netloc}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _extract_ip(payload: Any) -> str | None:
    if isinstance(payload, dict):
        value = payload.get("ip")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            # plain-text IP body
            return text if " " not in text and len(text) < 64 else None
        return _extract_ip(loaded)
    return None


def _body_looks_blocked(body: str) -> bool:
    text = (body or "").casefold()
    return any(marker in text for marker in BLOCKED_BODY_MARKERS)


def probe_proxy_http_fast(
    config: ServiceConfig,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    impersonate: str = DEFAULT_IMPERSONATE,
    check_google: bool = True,
    session: Any | None = None,
) -> FastProxyProbeResult:
    """Cheap L0 proxy screen using curl_cffi (no browser).

    Pass criteria:
    - at least one egress IP endpoint succeeds
    - optional Google AI URL is reachable and body is not an obvious block/enablejs shell
    """
    try:
        from curl_cffi import requests as curl_requests
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise RuntimeError(
            "curl_cffi is required for fast proxy probes; install project dependencies."
        ) from exc

    proxy_url = build_proxy_url(config)
    if not proxy_url:
        return FastProxyProbeResult(ok=False, reason="proxy is not configured")

    raw: dict[str, Any] = {"proxy_scheme": urlsplit(proxy_url).scheme}
    ips: list[str] = []
    owns_session = session is None
    client = session or curl_requests.Session(impersonate=impersonate)
    proxies = {"http": proxy_url, "https": proxy_url}

    try:
        for endpoint in EGRESS_ENDPOINTS:
            try:
                response = client.get(
                    endpoint,
                    proxies=proxies,
                    timeout=timeout_s,
                    allow_redirects=True,
                )
                body_text = response.text
                raw[endpoint] = {
                    "status": response.status_code,
                    "body": body_text[:300],
                }
                if response.status_code >= 400:
                    continue
                ip = _extract_ip(body_text)
                if ip:
                    ips.append(ip)
            except Exception as exc:
                raw[endpoint] = {"error": f"{type(exc).__name__}: {exc}"[:300]}

        vector = normalize_ip_vector(ips)
        if not vector:
            return FastProxyProbeResult(
                ok=False,
                reason="fast http egress probe failed (no IP)",
                raw=raw,
            )

        google_status: int | None = None
        google_blocked = False
        if check_google:
            try:
                response = client.get(
                    GOOGLE_AI_PROBE_URL,
                    proxies=proxies,
                    timeout=timeout_s,
                    allow_redirects=True,
                )
                google_status = int(response.status_code)
                body_text = response.text or ""
                raw["google_ai"] = {
                    "status": google_status,
                    "final_url": str(getattr(response, "url", "") or "")[:300],
                    "body_excerpt": body_text[:400],
                }
                if google_status in {401, 403, 429, 503} or _body_looks_blocked(body_text):
                    google_blocked = True
            except Exception as exc:
                raw["google_ai"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                return FastProxyProbeResult(
                    ok=False,
                    reason=f"fast http google probe failed: {type(exc).__name__}",
                    ips=vector,
                    primary_ip=vector[0],
                    raw=raw,
                )

        if google_blocked:
            return FastProxyProbeResult(
                ok=False,
                reason=(
                    f"fast http google probe blocked "
                    f"(status={google_status})"
                ),
                ips=vector,
                primary_ip=vector[0],
                google_status=google_status,
                google_blocked=True,
                raw=raw,
            )

        return FastProxyProbeResult(
            ok=True,
            ips=vector,
            primary_ip=vector[0],
            google_status=google_status,
            google_blocked=False,
            raw=raw,
        )
    finally:
        if owns_session:
            try:
                client.close()
            except Exception:
                pass
