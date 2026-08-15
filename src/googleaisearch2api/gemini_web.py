"""Pure-HTTP Gemini web app fallback (reverse-engineered StreamGenerate protocol).

Anonymous-or-authenticated StreamGenerate client, search-grounded, with
structured citations.  The protocol is UNVERIFIED internally (see AGENTS.md):
this module is a standalone primitive with no app wiring, and the 102-slot
inner request is a faithful port of the gemini-web2api reference
(github.com/Sophomoresty/gemini-web2api, commit
6824ccaaa65768d9a4befc33a417e268f134b252).  Treat the slot layout as unstable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from typing import Any

from loguru import logger

from .gemini_web_models import resolve_model
from .schemas import Citation, GoogleAiResult

GEMINI_STREAM_ENDPOINT = (
    "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
)
GEMINI_APP_URL = "https://gemini.google.com/app"
DEFAULT_HL = "en"
DEFAULT_BL_FALLBACK = "boq_assistant-bard-web-server_20260716.08_p0"
DEFAULT_IMPERSONATE = "chrome146"

_BL_RE = re.compile(r"boq_assistant-bard-web-server_\d+\.\d+_p\d+")
_CODE_ARTIFACT_RE = re.compile(
    r"(?s)```(?:python|javascript|text)\?code_(?:reference|stdout)"
    r"&code_event_index=\d+\n.*?```\n?"
)

_BLOCK_STATUS = {302, 303, 307, 308, 403, 405, 429, 503}
_BLOCK_MARKERS = (
    "google.com/sorry",
    "www.google.com/sorry",
    "unusual traffic",
    "recaptcha",
    "our systems have detected",
    "sorry/index",
)


class GeminiWebRuntimeError(RuntimeError):
    """Generic Gemini web failure (protocol/transport)."""


class GeminiWebBlockedError(GeminiWebRuntimeError):
    """IP hard-blocked: redirected to Google's /sorry interstitial."""


class GeminiWebRateLimitedError(GeminiWebRuntimeError):
    """Rate-limited: empty/short response body."""


def build_stream_generate_url(*, bl: str, hl: str = DEFAULT_HL) -> str:
    reqid = int(time.time()) % 1_000_000
    return f"{GEMINI_STREAM_ENDPOINT}?bl={bl}&hl={hl}&_reqid={reqid}&rt=c"


def build_inner_json(
    prompt: str,
    *,
    model_id: int = 1,
    think_mode: int = 0,
    extra_fields: dict[int, object] | None = None,
) -> list[object]:
    """Build the 102-slot inner request array (non-temporary chat)."""
    inner: list[object] = [None] * 102
    inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [2]  # non-temporary chat
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for slot, value in extra_fields.items():
            inner[slot] = value
    return inner


def build_f_req(
    prompt: str,
    *,
    model_id: int = 1,
    think_mode: int = 0,
    extra_fields: dict[int, object] | None = None,
) -> str:
    """Build the `f.req` form value: `[null, "<inner JSON string>"]`."""
    inner = build_inner_json(
        prompt, model_id=model_id, think_mode=think_mode, extra_fields=extra_fields
    )
    return json.dumps([None, json.dumps(inner)])


def make_sapisidhash(sapisid: str) -> str:
    """Build the `SAPISIDHASH <ts>_<sha1>` Authorization value."""
    ts = int(time.time())
    digest = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{digest}"


def is_block_response(
    status_code: int, headers: dict[str, str] | None = None, body: bytes | str = b""
) -> bool:
    """Heuristic: Google captcha / rate-limit / method-trap response."""
    if status_code in _BLOCK_STATUS:
        if status_code not in (302, 303, 307, 308):
            return True
        loc = (headers or {}).get("Location") or (headers or {}).get("location") or ""
        text = loc
        if body:
            text += " " + (
                body.decode("utf-8", "replace")
                if isinstance(body, (bytes, bytearray))
                else str(body)
            )
        lowered = text.lower()
        if any(m in lowered for m in _BLOCK_MARKERS) or "sorry" in lowered:
            return True
        if "streamgenerate" not in lowered and "gemini.google.com" not in lowered:
            return True
        return "google.com" in lowered and "sorry" in lowered
    if not body:
        return False
    sample = body[:4000]
    text = (
        sample.decode("utf-8", "replace") if isinstance(sample, (bytes, bytearray)) else str(sample)
    )
    return any(m in text.lower() for m in _BLOCK_MARKERS)


def _build_headers(cookie: str = "", sapisid: str = "") -> dict[str, str]:
    """Mandatory StreamGenerate headers; Cookie/Authorization only when present."""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": GEMINI_APP_URL,
        "X-Same-Domain": "1",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        ),
    }
    if cookie:
        headers["Cookie"] = cookie
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    return headers


def parse_response_frames(raw: str) -> list[object]:
    """Split a newline-delimited `wrb.fr` stream into parsed JSON frames."""
    if raw.startswith(")]}'"):
        raw = raw[4:]
    frames: list[object] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            frames.append(value)
    return frames


def _inner_from_frame(frame: object) -> object | None:
    if not isinstance(frame, list) or not frame:
        return None
    first = frame[0]
    if not isinstance(first, list) or len(first) < 3:
        return None
    inner_str = first[2]
    if not isinstance(inner_str, str) or not inner_str:
        return None
    try:
        return json.loads(inner_str)
    except json.JSONDecodeError:
        return None


def extract_answer_text(frames: list[object]) -> str:
    """Extract the answer text (last non-empty cumulative text chunk)."""
    texts: list[str] = []
    for frame in frames:
        inner = _inner_from_frame(frame)
        if not isinstance(inner, list) or len(inner) <= 4:
            continue
        parts = inner[4]
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, list) or len(part) < 2:
                continue
            text_list = part[1]
            if not isinstance(text_list, list):
                continue
            for text in text_list:
                if isinstance(text, str) and text.strip():
                    texts.append(text)
    for text in reversed(texts):
        if text.strip():
            return _clean_gemini_text(text)
    return ""


def _clean_gemini_text(text: str) -> str:
    return _CODE_ARTIFACT_RE.sub("", text).strip()


def extract_citations(frames: list[object]) -> list[Citation]:
    """Extract structured citations from grounding chunks (inner[4][0][2][1])."""
    citations: list[Citation] = []
    seen: set[str] = set()
    for frame in frames:
        inner = _inner_from_frame(frame)
        for chunk in _grounding_chunks(inner):
            if not isinstance(chunk, list) or len(chunk) <= 2:
                continue
            sources = chunk[2]
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, list) or len(source) < 2:
                    continue
                url = source[0]
                if not isinstance(url, str):
                    continue
                url = _strip_fragment(url)
                if not url or url in seen:
                    continue
                title = source[1]
                title = title if isinstance(title, str) else ""
                seen.add(url)
                citations.append(Citation(title=title, url=url))
    return citations


def _grounding_chunks(inner: object) -> list[object]:
    if not isinstance(inner, list) or len(inner) <= 4:
        return []
    node_list = inner[4]
    if not isinstance(node_list, list) or not node_list:
        return []
    node = node_list[0]
    if not isinstance(node, list) or len(node) <= 2:
        return []
    grounding = node[2]
    if not isinstance(grounding, list) or len(grounding) <= 1:
        return []
    chunks = grounding[1]
    return chunks if isinstance(chunks, list) else []


def _strip_fragment(url: str) -> str:
    if "#:~:text=" in url:
        url = url.split("#:~:text=")[0]
    return url.split("#", 1)[0]


_cached_bl: str | None = None
_BL_FETCH_TIMEOUT_S = 15.0


def _resolve_bl(client: Any, proxies: dict[str, str] | None) -> str:
    global _cached_bl
    if _cached_bl is not None:
        return _cached_bl
    try:
        response = client.get(
            GEMINI_APP_URL,
            proxies=proxies,
            timeout=_BL_FETCH_TIMEOUT_S,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        text = getattr(response, "text", "") or ""
        match = _BL_RE.search(text)
        _cached_bl = match.group(0) if match else DEFAULT_BL_FALLBACK
    except Exception as exc:
        logger.warning("Gemini web bl fetch failed: {}: {}", type(exc).__name__, exc)
        _cached_bl = DEFAULT_BL_FALLBACK
    return _cached_bl


class GeminiWebClient:
    """Pure-HTTP Gemini web client (StreamGenerate protocol)."""

    def __init__(self, *, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s

    def run(
        self,
        prompt: str,
        *,
        model: str = "gemini-3.7-flash",
        cookie: str | None = None,
        sapisid: str | None = None,
        session: Any | None = None,
        proxies: dict[str, str] | None = None,
    ) -> GoogleAiResult:
        try:
            from curl_cffi import requests as curl_requests
        except ImportError as exc:
            raise GeminiWebRuntimeError("curl_cffi is required for Gemini web transport") from exc

        _, model_id, think_mode, resolve_error, extra_fields = resolve_model(model)
        if resolve_error is not None or model_id is None or think_mode is None:
            raise GeminiWebRuntimeError(resolve_error or f"cannot resolve model {model!r}")

        owns_session = session is None
        client = session or curl_requests.Session(impersonate=DEFAULT_IMPERSONATE)

        try:
            bl = _resolve_bl(client, proxies)
            url = build_stream_generate_url(bl=bl)
            response = client.post(
                url,
                data={
                    "f.req": build_f_req(
                        prompt, model_id=model_id, think_mode=think_mode, extra_fields=extra_fields
                    )
                },
                headers=_build_headers(cookie or "", sapisid or ""),
                proxies=proxies,
                timeout=self.timeout_s,
                allow_redirects=False,
            )
            final_url = str(getattr(response, "url", "") or "")
            status = int(getattr(response, "status_code", 0) or 0)
            text = getattr(response, "text", "") or ""
            response_headers = getattr(response, "headers", None)
            if is_block_response(status, response_headers, text) or "/sorry" in final_url:
                raise GeminiWebBlockedError(f"Gemini web IP-blocked (HTTP {status} -> {final_url})")
            if not text or len(text.strip()) < 10:
                raise GeminiWebRateLimitedError("Gemini web returned an empty/rate-limited body")
            frames = parse_response_frames(text)
            answer = extract_answer_text(frames)
            citations = extract_citations(frames)
            if not answer and not citations:
                raise GeminiWebRuntimeError("no answer frame in Gemini web response")
            return GoogleAiResult(
                answer_text=answer,
                citations=citations,
                final_url=final_url,
                page_title="",
                body_excerpt=answer[:800],
            )
        except (GeminiWebBlockedError, GeminiWebRateLimitedError, GeminiWebRuntimeError):
            raise
        except Exception as exc:
            raise GeminiWebRuntimeError(f"Gemini web request failed: {exc}") from exc
        finally:
            if owns_session:
                try:
                    client.close()
                except Exception:
                    pass
