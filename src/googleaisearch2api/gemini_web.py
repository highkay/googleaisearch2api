"""Pure-HTTP Gemini web app fallback (reverse-engineered StreamGenerate protocol).

Anonymous (no cookie), search-grounded, with structured citations.  The
protocol is UNVERIFIED internally (see AGENTS.md): this module is a standalone
primitive with no app wiring, and the 80-slot inner request is a faithful port
of the gemini-web2api reference
(github.com/Sophomoresty/gemini-web2api, commit
6824ccaaa65768d9a4befc33a417e268f134b252).  Treat the slot layout as unstable.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from loguru import logger

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


class GeminiWebRuntimeError(RuntimeError):
    """Generic Gemini web failure (protocol/transport)."""


class GeminiWebBlockedError(GeminiWebRuntimeError):
    """IP hard-blocked: redirected to Google's /sorry interstitial."""


class GeminiWebRateLimitedError(GeminiWebRuntimeError):
    """Rate-limited: empty/short response body."""


def build_stream_generate_url(*, bl: str, hl: str = DEFAULT_HL) -> str:
    reqid = int(time.time()) % 1_000_000
    return f"{GEMINI_STREAM_ENDPOINT}?bl={bl}&hl={hl}&_reqid={reqid}&rt=c"


def build_inner_json(prompt: str) -> list[object]:
    """Build the 80-slot inner request array (temporary chat, no thinking).

    No model override is set: anonymous requests fall back to the server's
    Flash-class default, mirroring the reference implementation.
    """
    inner: list[object] = [None] * 80
    inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[0]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    inner[41] = [1]  # temporary chat
    inner[45] = 1
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    return inner


def build_f_req(prompt: str) -> str:
    """Build the `f.req` form value: `[null, "<inner JSON string>"]`."""
    outer = [None, json.dumps(build_inner_json(prompt))]
    return json.dumps(outer)


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


class GeminiWebClient:
    """Pure-HTTP Gemini web client (anonymous, no cookie)."""

    def __init__(self, *, timeout_s: float = 20.0) -> None:
        self.timeout_s = timeout_s

    def run(
        self,
        prompt: str,
        *,
        session: Any | None = None,
        proxies: dict[str, str] | None = None,
    ) -> GoogleAiResult:
        try:
            from curl_cffi import requests as curl_requests
        except ImportError as exc:
            raise GeminiWebRuntimeError("curl_cffi is required for Gemini web transport") from exc

        owns_session = session is None
        client = session or curl_requests.Session(impersonate=DEFAULT_IMPERSONATE)

        try:
            bl = self._fetch_bl(client, proxies)
            url = build_stream_generate_url(bl=bl)
            headers = {
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "Origin": "https://gemini.google.com",
                "Referer": GEMINI_APP_URL,
                "X-Same-Domain": "1",
                "Accept-Language": "en-US,en;q=0.9",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/146.0.0.0 Safari/537.36"
                ),
            }
            response = client.post(
                url,
                data={"f.req": build_f_req(prompt)},
                headers=headers,
                proxies=proxies,
                timeout=self.timeout_s,
            )
            final_url = str(getattr(response, "url", "") or "")
            status = int(getattr(response, "status_code", 0) or 0)
            if "/sorry" in final_url or status in {301, 302}:
                raise GeminiWebBlockedError("Gemini web IP-blocked (redirected to /sorry)")
            text = getattr(response, "text", "") or ""
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

    def _fetch_bl(self, client: Any, proxies: dict[str, str] | None) -> str:
        try:
            response = client.get(
                GEMINI_APP_URL,
                proxies=proxies,
                timeout=self.timeout_s,
                headers={
                    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                },
            )
            text = getattr(response, "text", "") or ""
            match = _BL_RE.search(text)
            if match:
                return match.group(0)
        except Exception as exc:
            logger.warning("Gemini web bl fetch failed: {}: {}", type(exc).__name__, exc)
        return DEFAULT_BL_FALLBACK
