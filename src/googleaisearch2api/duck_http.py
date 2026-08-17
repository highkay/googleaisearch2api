"""Pure-HTTP Duck.ai engine client (curl_cffi + the wasmrt x-vqd challenge solver).

Browserless counterpart of ``duck_ai.py``: GET ``/duckchat/v1/status``
(``x-vqd-accept: 1``) for the ``x-vqd-hash-1`` challenge, solve it with
``duck_solver.solve_vqd``, then POST ``/duckchat/v1/chat`` and read the
``data:`` SSE stream (wire formats live in ``duck_http_protocol``). Solver
acceptance was proven live 2026-08-17: the full protocol set earns a 200
``action=success`` SSE stream on both WARP and 2260 egress.

Only stdlib + the sibling package at import time; curl_cffi is imported lazily
inside ``run`` (mirrors ``gemini_web.GeminiWebClient``).
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from .duck_ai import (  # noqa: F401 - re-exported for callers of this module
    DUCK_RATE_LIMIT_MARKERS,
    _is_rate_limited_text,
    _normalize_text,
    build_duck_search_prompt,
    extract_duck_answer_text,
)
from .duck_http_protocol import (  # noqa: F401 - re-exported for callers of this module
    _BASE_BROWSER_HEADERS,
    DUCK_AI_URL,
    DUCK_CHAT_ENDPOINT,
    DUCK_MODELS_ENDPOINT,
    DUCK_STATUS_ENDPOINT,
    build_chat_body,
    build_chat_headers,
    build_fe_signals,
    parse_sse_events,
    status_headers,
)
from .duck_solver import decode_challenge, solve_vqd
from .schemas import GoogleAiResult

#: Wire model id observed on the live /duckchat/v1/models catalog (free tier,
#: entityHasAccess=true); the server maps it to ``gpt-5.4-mini-2026-03-17``.
DEFAULT_MODEL = "gpt-5.4-mini"
#: Closest curl_cffi impersonation target to the solver's Chrome/149 UA.
DEFAULT_IMPERSONATE = "chrome146"
DEFAULT_TIMEOUT_S = 60.0
_FE_FETCH_TIMEOUT_S = 15.0

_FE_TAG_RE = re.compile(r'data-version-tag="([^"]+)"')
_FE_SHA_RE = re.compile(r'data-version-sha="([^"]+)"')


class DuckHttpRuntimeError(RuntimeError):
    """Generic Duck.ai HTTP failure (transport/protocol/parse)."""


class DuckHttpBlockedError(DuckHttpRuntimeError):
    """Challenge gate failed: HTTP 418, missing/unsolvable challenge, or ERR_CHALLENGE."""


class DuckHttpRateLimitedError(DuckHttpRuntimeError):
    """Rate-limited: HTTP 429 or rate-limit markers in the response body."""


_cached_fe_version: str | None = None


def _resolve_fe_version(client: Any, proxies: dict[str, str] | None) -> str | None:
    """Scrape the duck.ai frontend build version; cache successes at module level.

    Failures are deliberately NOT cached (a transient proxy error would otherwise
    omit ``x-fe-version`` forever); omitting the header is safe.
    """
    global _cached_fe_version
    if _cached_fe_version is not None:
        return _cached_fe_version
    try:
        response = client.get(
            DUCK_AI_URL,
            proxies=proxies,
            timeout=_FE_FETCH_TIMEOUT_S,
            headers={
                **_BASE_BROWSER_HEADERS,
                "accept": "text/html",
                "content-type": "application/json",
            },
        )
        text = getattr(response, "text", "") or ""
        tag_match = _FE_TAG_RE.search(text)
        sha_match = _FE_SHA_RE.search(text)
        if tag_match and sha_match:
            _cached_fe_version = f"{tag_match.group(1)}-{sha_match.group(1)}"
    except Exception as exc:
        logger.warning("Duck.ai frontend version fetch failed: {}: {}", type(exc).__name__, exc)
    return _cached_fe_version


def _classify_error_event(status: object, event_kind: object) -> None:
    code = status if isinstance(status, int) else None
    kind = str(event_kind) if event_kind else ""
    if code == 418 or "CHALLENGE" in kind.upper():
        raise DuckHttpBlockedError(
            f"Duck.ai chat rejected the request (action=error status={code} type={kind})"
        )
    if code == 429:
        raise DuckHttpRateLimitedError("Duck.ai chat rate limited (action=error status=429)")
    raise DuckHttpRuntimeError(f"Duck.ai chat error event (status={code} type={kind})")


class DuckHttpClient:
    """Pure-HTTP Duck.ai client (curl_cffi transport + wasmrt challenge solver)."""

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s

    def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        proxies: dict[str, str] | None = None,
        session: Any | None = None,
    ) -> GoogleAiResult:
        prompt = prompt.strip()
        if not prompt:
            raise DuckHttpRuntimeError("Prompt is empty.")
        submitted_prompt = build_duck_search_prompt(prompt)
        model = model or DEFAULT_MODEL

        try:
            from curl_cffi import requests as curl_requests
        except ImportError as exc:
            raise DuckHttpRuntimeError("curl_cffi is required for Duck.ai HTTP transport") from exc

        owns_session = session is None
        client = session or curl_requests.Session(impersonate=DEFAULT_IMPERSONATE)

        try:
            solved_hash = self._solve_challenge(client, proxies)
            fe_version = _resolve_fe_version(client, proxies)
            headers = build_chat_headers(solved_hash, fe_version=fe_version)
            response = client.post(
                DUCK_CHAT_ENDPOINT,
                json=build_chat_body(submitted_prompt, model),
                headers=headers,
                proxies=proxies,
                timeout=self.timeout_s,
                stream=True,
            )
            status = int(getattr(response, "status_code", 0) or 0)
            if status != 200:
                body_text = _safe_body_text(response)
                if status == 418 or _mentions_challenge(body_text):
                    raise DuckHttpBlockedError(
                        f"Duck.ai refused the solved challenge (HTTP {status}): {body_text[:200]}"
                    )
                if status == 429 or _is_rate_limited_text(body_text):
                    raise DuckHttpRateLimitedError(
                        f"Duck.ai rate limited the chat request (HTTP {status})."
                    )
                raise DuckHttpRuntimeError(
                    f"Duck.ai chat request failed (HTTP {status}): {body_text[:200]}"
                )

            raw = _read_sse_text(response)
            if not raw.strip():
                raise DuckHttpRuntimeError("Duck.ai chat returned an empty SSE body.")
            if _is_rate_limited_text(raw):
                raise DuckHttpRateLimitedError("Duck.ai returned a rate-limit page body.")

            events = parse_sse_events(raw)
            parts: list[str] = []
            for event in events:
                if event.get("action") == "error":
                    _classify_error_event(event.get("status"), event.get("type"))
                message = event.get("message") if event.get("action") == "success" else None
                if isinstance(message, str) and message.strip():
                    parts.append(message)
            accumulated = "".join(parts).strip()
            if not accumulated:
                raise DuckHttpRuntimeError("Duck.ai chat stream carried no answer message.")

            answer = extract_duck_answer_text(accumulated, prompt) or accumulated
            return GoogleAiResult(
                answer_text=answer,
                citations=[],
                final_url=f"{DUCK_AI_URL}/",
                page_title="Duck.ai",
                body_excerpt=accumulated[:800],
            )
        except (DuckHttpBlockedError, DuckHttpRateLimitedError, DuckHttpRuntimeError):
            raise
        except Exception as exc:
            raise DuckHttpRuntimeError(f"Duck.ai HTTP request failed: {exc}") from exc
        finally:
            if owns_session:
                try:
                    client.close()
                except Exception:
                    pass

    def _solve_challenge(self, client: Any, proxies: dict[str, str] | None) -> str:
        response = client.get(
            DUCK_STATUS_ENDPOINT,
            headers=status_headers(),
            proxies=proxies,
            timeout=self.timeout_s,
        )
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 429 or _is_rate_limited_text(_safe_body_text(response)):
            raise DuckHttpRateLimitedError(
                f"Duck.ai rate limited the challenge status request (HTTP {status})."
            )
        if status != 200:
            raise DuckHttpRuntimeError(
                f"Duck.ai challenge status request failed (HTTP {status})."
            )

        headers = getattr(response, "headers", None) or {}
        challenge_b64 = headers.get("x-vqd-hash-1") or headers.get("X-Vqd-Hash-1")
        if not challenge_b64:
            raise DuckHttpBlockedError(
                "Duck.ai status response carried no x-vqd-hash-1 challenge header."
            )
        try:
            challenge_source = decode_challenge(challenge_b64)
            return solve_vqd(challenge_source)
        except (ValueError, TypeError) as exc:
            raise DuckHttpBlockedError(f"Could not solve the Duck.ai challenge: {exc}") from exc


def _safe_body_text(response: Any) -> str:
    try:
        return getattr(response, "text", "") or ""
    except Exception:
        return ""


def _read_sse_text(response: Any) -> str:
    parts: list[str] = []
    for chunk in response.iter_lines():
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", "replace")
        if chunk is not None:
            parts.append(str(chunk))
    return "\n".join(parts)


def _mentions_challenge(body_text: str) -> bool:
    lowered = (body_text or "").lower()
    return "err_challenge" in lowered or "challenge" in lowered