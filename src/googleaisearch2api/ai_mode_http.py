"""Pure-HTTP transport for Google AI Mode's /async/folif endpoint.

This module is intentionally a primitive: it builds the folif URL and
classifies raw HTTP responses, with no browser or app wiring.  The folif
protocol itself is UNVERIFIED internally (see AGENTS.md), so no Google
internal endpoint is treated as a stable contract here.

Shared by:
- a future hybrid (HTTP + browser) runner
- a TTL probe script
"""

from __future__ import annotations

import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from .config import ServiceConfig
from .fast_proxy_probe import (
    BLOCKED_BODY_MARKERS,
    DEFAULT_IMPERSONATE,
    build_proxy_url,
)

FOLIF_BASE_URL = "https://www.google.com/search"
DEFAULT_FOLIF_TIMEOUT_S = 15.0

# JS-required shells. NOT IP blocks: pure HTTP routinely receives the enablejs
# retry page even for healthy sticky exits (see AGENTS.md / commit 1a2ee81).
SHELL_MARKERS: tuple[str, ...] = ("emsg=sg_rel", "sg_rel", "enablejs", "noscript")

# IP-level Google blocks, shared with the L0 fast proxy probe.
FOLIF_BLOCKED_MARKERS: tuple[str, ...] = BLOCKED_BODY_MARKERS

_ANSWER_MARKERS: tuple[str, ...] = ("n6owBd", "pTRUV", "data-mstk")


@dataclass(frozen=True, slots=True)
class AiModeTokens:
    srtst: str | None = None
    garc: str | None = None
    xsrf_folif_token: str | None = None
    ei: str | None = None
    stkp: str | None = None
    sca_esv: str | None = None
    mstk: str | None = None
    # Not a "credential" but part of the folif query contract (see
    # /tmp/opencode/xd06_engine.py folif param assembly).
    elrc: str | None = None
    cookies: dict[str, str] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return bool(self.xsrf_folif_token and self.stkp)


def build_folif_url(
    tokens: AiModeTokens,
    prompt: str,
    *,
    base_url: str = FOLIF_BASE_URL,
) -> str:
    """Assemble the /async/folif URL in Google's own parameter order."""
    params: list[tuple[str, str]] = []
    if tokens.ei:
        params.append(("ei", tokens.ei))
    params.extend((("yv", "3"), ("udm", "50"), ("hl", "en")))
    if tokens.stkp:
        params.append(("stkp", tokens.stkp))
    params.extend((("cs", "0"), ("csuir", "0")))
    if tokens.elrc:
        params.append(("elrc", tokens.elrc))
    params.append(("q", prompt))
    params.append(("async", f"_fmt:adl,_xsrf:{tokens.xsrf_folif_token or ''}"))
    for key in ("srtst", "garc", "sca_esv", "mstk"):
        value = getattr(tokens, key)
        if value:
            params.append((key, value))
    query = urllib.parse.urlencode(params)
    return f"{base_url}/async/folif?{query}"


@dataclass(frozen=True, slots=True)
class FolifResult:
    kind: Literal["answer", "shell", "empty", "blocked", "error"]
    status_code: int
    body: str
    final_url: str
    answer_text: str = ""


def classify_folif_response(status_code: int, body: str, final_url: str) -> FolifResult:
    """Classify a raw folif HTTP response into a FolifResult.

    Shell classification runs BEFORE blocked classification: enablejs/SG_REL
    shells are not IP blocks (commit 1a2ee81) and must never be reported as
    blocked.
    """
    text = body or ""
    low = text.casefold()

    if any(marker.casefold() in low for marker in SHELL_MARKERS):
        return FolifResult(kind="shell", status_code=status_code, body=body, final_url=final_url)

    if (
        status_code in {403}
        or any(marker.casefold() in low for marker in FOLIF_BLOCKED_MARKERS)
        or "/sorry" in final_url
    ):
        return FolifResult(kind="blocked", status_code=status_code, body=body, final_url=final_url)

    if not text or len(text.strip()) < 50:
        return FolifResult(kind="empty", status_code=status_code, body=body, final_url=final_url)

    if status_code == 200 and any(marker in text for marker in _ANSWER_MARKERS):
        return FolifResult(
            kind="answer",
            status_code=status_code,
            body=body,
            final_url=final_url,
            answer_text=body,
        )

    return FolifResult(kind="error", status_code=status_code, body=body, final_url=final_url)


def extract_answer_from_folif(body: str) -> str:
    """TODO(M1): the TTL probe will calibrate the real answer marker.

    Deliberately conservative placeholder: only data-mstk has been observed on
    live folif payloads so far.  Until M1 lands, return the whole stripped body
    when that marker is present and otherwise decline to guess.
    """
    if "data-mstk" in body:
        return body.strip()
    return ""


def fetch_folif(
    tokens: AiModeTokens,
    prompt: str,
    *,
    config: ServiceConfig,
    timeout_s: float = DEFAULT_FOLIF_TIMEOUT_S,
    session: Any | None = None,
) -> FolifResult:
    """GET the folif endpoint via curl_cffi and classify the raw response."""
    try:
        from curl_cffi import requests as curl_requests
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise RuntimeError(
            "curl_cffi is required for folif HTTP transport; install project dependencies."
        ) from exc

    url = build_folif_url(tokens, prompt)
    proxy_url = build_proxy_url(config)
    proxies: dict[str, str] | None = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    headers = {
        "Referer": "https://www.google.com/search",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Upgrade-Insecure-Requests": "1",
    }

    owns_session = session is None
    client = session or curl_requests.Session(impersonate=DEFAULT_IMPERSONATE)

    try:
        response = client.get(
            url,
            proxies=proxies,
            timeout=timeout_s,
            allow_redirects=True,
            headers=headers,
        )
        return classify_folif_response(
            status_code=int(response.status_code),
            body=getattr(response, "text", "") or "",
            final_url=str(getattr(response, "url", "") or ""),
        )
    except Exception as exc:
        logger.warning("folif http transport failed: {}: {}", type(exc).__name__, exc)
        return FolifResult(kind="error", status_code=0, body="", final_url="")
    finally:
        if owns_session:
            try:
                client.close()
            except Exception:
                pass
