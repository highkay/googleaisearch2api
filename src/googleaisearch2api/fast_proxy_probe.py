from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from .browser import resolve_browser_proxy
from .config import ServiceConfig
from .gemini_web import (
    GeminiWebBlockedError,
    GeminiWebClient,
    GeminiWebRateLimitedError,
    GeminiWebRuntimeError,
)
from .proxy_sessions import normalize_ip_vector

EGRESS_ENDPOINTS = (
    "https://api.ipify.org?format=json",
    "https://api64.ipify.org?format=json",
)
GOOGLE_AI_PROBE_URL = "https://www.google.com/search?udm=50&aep=11&hl=en&q=ping"
DUCK_PROBE_URL = "https://duck.ai/duckchat/v1/status"
DUCK_PROBE_CHALLENGE_HEADER = "x-vqd-hash-1"
DEFAULT_IMPERSONATE = "chrome131"
DEFAULT_TIMEOUT_S = 8.0

# IP-level Google blocks only. Do NOT treat enablejs / noscript shells as blocked:
# pure HTTP (httpx/curl_cffi) routinely receives the enablejs retry page even for
# healthy sticky exits; only a real browser can complete that path (see AGENTS.md).
BLOCKED_BODY_MARKERS = (
    "unusual traffic",
    "this network is blocked due to unaddressed abuse complaints",
    "malicious behavior",
    "this page checks to see if it's really a human",
    "not a robot",
    "captcha",
    "sorry/index",
    "/sorry/",
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


def _probe_client(impersonate: str, session: Any | None) -> tuple[Any, bool]:
    """Create or reuse a curl_cffi session; returns (client, owns_session)."""
    try:
        from curl_cffi import requests as curl_requests
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise RuntimeError(
            "curl_cffi is required for fast proxy probes; install project dependencies."
        ) from exc
    if session is not None:
        return session, False
    return curl_requests.Session(impersonate=impersonate), True


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
    - optional Google AI URL is reachable and body is not an obvious IP-level block
      (enablejs / JS shells are allowed; browser canary owns that path)
    """
    client, owns_session = _probe_client(impersonate, session)

    proxy_url = build_proxy_url(config)
    if not proxy_url:
        return FastProxyProbeResult(ok=False, reason="proxy is not configured")

    raw: dict[str, Any] = {"proxy_scheme": urlsplit(proxy_url).scheme}
    ips: list[str] = []
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


def probe_gemini_http_fast(
    config: ServiceConfig,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> FastProxyProbeResult:
    """Cheap L0 probe: a lightweight StreamGenerate request via GeminiWebClient.

    A tiny-prompt StreamGenerate POST proves the sticky exit can actually
    complete a Gemini web request — the homepage GET alone is too weak a signal
    for flaky exits that answer GET but hang on the POST.
    """
    proxy_url = build_proxy_url(config)
    if not proxy_url:
        return FastProxyProbeResult(ok=False, reason="proxy is not configured")

    proxies = {"http": proxy_url, "https": proxy_url}
    client = GeminiWebClient(timeout_s=timeout_s)
    try:
        result = client.run("ping", proxies=proxies)
    except GeminiWebBlockedError as exc:
        return FastProxyProbeResult(
            ok=False,
            reason=f"gemini probe blocked: {exc}",
            google_blocked=True,
        )
    except GeminiWebRateLimitedError as exc:
        return FastProxyProbeResult(ok=False, reason=f"gemini probe rate limited: {exc}")
    except GeminiWebRuntimeError as exc:
        return FastProxyProbeResult(ok=False, reason=f"gemini probe failed: {exc}")
    return FastProxyProbeResult(
        ok=True,
        raw={"answer_preview": result.answer_text[:200]},
    )


def probe_duck_http_fast(
    config: ServiceConfig,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> FastProxyProbeResult:
    """Status-only Duck.ai L0 screen: GET /duckchat/v1/status and expect the
    x-vqd-hash-1 challenge header. Solves nothing and never chats, so a
    candidate exit is screened without burning the ~640ms solver budget.
    """
    proxy_url = build_proxy_url(config)
    if not proxy_url:
        return FastProxyProbeResult(ok=False, reason="proxy is not configured")

    client, owns_session = _probe_client(DEFAULT_IMPERSONATE, None)
    raw: dict[str, Any] = {"proxy_scheme": urlsplit(proxy_url).scheme}
    proxies = {"http": proxy_url, "https": proxy_url}
    try:
        try:
            response = client.get(
                DUCK_PROBE_URL,
                headers={"x-vqd-accept": "1"},
                proxies=proxies,
                timeout=timeout_s,
            )
        except Exception as exc:
            return FastProxyProbeResult(
                ok=False,
                reason=f"duck probe failed: {type(exc).__name__}",
                raw=raw,
            )

        status_code = int(getattr(response, "status_code", 0) or 0)
        response_headers = getattr(response, "headers", None) or {}
        challenge = next(
            (
                value
                for key, value in response_headers.items()
                if key.lower() == DUCK_PROBE_CHALLENGE_HEADER
            ),
            None,
        )
        raw["duck_status"] = {"status": status_code, "challenge": bool(challenge)}
        if status_code == 429:
            return FastProxyProbeResult(ok=False, reason="duck probe rate limited", raw=raw)
        if status_code != 200:
            return FastProxyProbeResult(
                ok=False,
                reason=f"duck probe failed (status={status_code})",
                raw=raw,
            )
        if not challenge:
            return FastProxyProbeResult(ok=False, reason="duck probe: challenge missing", raw=raw)

        # Best-effort egress IP for the caller's debug payload only; it never gates ok-ness.
        try:
            egress = client.get(
                EGRESS_ENDPOINTS[0],
                proxies=proxies,
                timeout=timeout_s,
                allow_redirects=True,
            )
            raw["egress_ip"] = _extract_ip(getattr(egress, "text", ""))
        except Exception as exc:
            raw["egress_ip_error"] = f"{type(exc).__name__}"[:120]
        return FastProxyProbeResult(ok=True, raw=raw)
    finally:
        if owns_session:
            try:
                client.close()
            except Exception:
                pass
