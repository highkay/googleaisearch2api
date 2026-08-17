"""Wire-format builders for the pure-HTTP Duck.ai engine.

Every header/body field here was proven live (2026-08-17, WARP + 2260 egress):
removing ``x-fe-signals``, the ``sec-ch-ua`` family, or the body's non-minimal
fields flipped the chat POST back to 418; the full set earns a 200 SSE stream.
"""

from __future__ import annotations

import base64
import json
import secrets
import time
from typing import Any

from .duck_solver import CHROME_USER_AGENT

DUCK_AI_URL = "https://duck.ai"
DUCK_STATUS_ENDPOINT = "https://duck.ai/duckchat/v1/status"
DUCK_CHAT_ENDPOINT = "https://duck.ai/duckchat/v1/chat"
DUCK_MODELS_ENDPOINT = "https://duck.ai/duckchat/v1/models"

# Chrome/149 client-hint + fetch-metadata set shared by every duck.ai request
# (Duck2api createHeader; observed-required: without it the chat POST is 418).
_BASE_BROWSER_HEADERS: dict[str, str] = {
    "accept-language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "origin": DUCK_AI_URL,
    "referer": f"{DUCK_AI_URL}/",
    "user-agent": CHROME_USER_AGENT,
}


def build_chat_body(prompt: str, model: str) -> dict[str, Any]:
    """Chat request body. Observed-required: the minimal OpenAI shape alone -> 418."""
    return {
        "model": model,
        "metadata": {
            "toolChoice": {
                "NewsSearch": False,
                "VideosSearch": False,
                "LocalSearch": False,
                "WeatherForecast": False,
            }
        },
        "messages": [{"role": "user", "content": prompt}],
        "canUseTools": True,
        "reasoningEffort": "none",
        "canUseApproxLocation": None,
        "canDelegateImageGeneration": None,
        "durableStream": {
            "messageId": "",
            "conversationId": "",
            "publicKey": {
                "alg": "",
                "e": "",
                "ext": False,
                "key_ops": None,
                "kty": "",
                "n": "",
                "use": "",
            },
        },
    }


def build_fe_signals(now_ms: int | None = None) -> str:
    """Base64 ``x-fe-signals`` event log (observed-required: absent -> 418).

    Replays the duck.ai frontend event sequence between page load and chat send
    (onboarding_impression -> action -> onboarding_finish -> startNewChat_free)
    with plausible random deltas, exactly as Duck2api's CreateFESignals does.
    """
    impression = 50 + secrets.randbelow(100)
    action = impression + 5000 + secrets.randbelow(25000)
    finish = action + 1000 + secrets.randbelow(9000)
    start_chat = finish + 10 + secrets.randbelow(90)
    end = start_chat + secrets.randbelow(10)
    start = (now_ms if now_ms is not None else int(time.time() * 1000)) - end
    payload = {
        "start": start,
        "events": [
            {"name": "onboarding_impression", "delta": impression},
            {"name": "action", "delta": action, "trusted": True},
            {"name": "onboarding_finish", "delta": finish},
            {"name": "startNewChat_free", "delta": start_chat},
        ],
        "end": end,
    }
    return base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def build_chat_headers(solved_hash: str, *, fe_version: str | None) -> dict[str, str]:
    """POST /duckchat/v1/chat header set (Duck2api postChat, live-verified).

    Every header besides ``x-vqd-hash-1`` is part of the acceptance surface;
    ``x-fe-version`` is omitted only when the homepage scrape failed (Duck2api
    ships the same fallback).
    """
    headers = {
        **_BASE_BROWSER_HEADERS,
        "accept": "text/event-stream",
        "content-type": "application/json",
        "priority": "u=1, i",
        "x-ddg-journey-id": secrets.token_hex(16),
        "x-fe-signals": build_fe_signals(),
        "x-vqd-hash-1": solved_hash,
    }
    if fe_version:
        headers["x-fe-version"] = fe_version
    return headers


def status_headers() -> dict[str, str]:
    """GET /duckchat/v1/status header set (accept: */* + x-vqd-accept: 1)."""
    return {
        **_BASE_BROWSER_HEADERS,
        "accept": "*/*",
        "content-type": "application/json",
        "x-vqd-accept": "1",
    }


def parse_sse_events(raw: str) -> list[dict[str, Any]]:
    """Parse a raw ``data:`` SSE stream into event dicts.

    Skips non-``data:`` lines, undecodable payloads (e.g. ``[CHAT_TITLE:...]``),
    and blank lines; a bare ``[DONE]`` payload terminates parsing.
    """
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events