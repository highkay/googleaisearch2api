"""Pure-HTTP client for a local OpenAI-compatible gemini-web2api gateway.

The operator runs a gemini-web2api gateway (e.g. at http://127.0.0.1:8081)
that exposes gemini models with NATIVE web search through a plain
OpenAI-compatible /v1/chat/completions endpoint.  Answers come back with
inline markdown source links `[Title](https://url)` which this module lifts
into structured citations.
"""

from __future__ import annotations

import re

from .schemas import Citation


class GeminiUpstreamRuntimeError(RuntimeError):
    """Generic gemini-upstream gateway failure (transport/protocol)."""


CITATION_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_BARE_URL_RE = re.compile(r"https?://[^\s)\]\"'<>\u4e00-\u9fff]+")


def _normalize_url(url: str) -> str:
    url = url.split("#", 1)[0]
    return url.rstrip(".,;:!?")


def extract_inline_citations(answer_text: str) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[str] = set()

    def _add(title: str, url: str) -> None:
        url = _normalize_url(url)
        if not url or url in seen:
            return
        seen.add(url)
        citations.append(Citation(title=title.strip() or url, url=url))

    # Markdown links first (they carry a human title), then any bare URLs the
    # markdown pass missed. Both forms appear in the gateway's answers.
    for title, url in CITATION_LINK_RE.findall(answer_text):
        _add(title, url)
    for url in _BARE_URL_RE.findall(answer_text):
        _add("", url)
    return citations


class GeminiUpstreamClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout_s: float = 120.0,
        model: str = "gemini-3.7-flash",
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.model = model

    def run(self, prompt: str, *, model: str | None = None) -> tuple[str, list[Citation]]:
        from curl_cffi import requests as curl_requests

        client = curl_requests.Session()
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            body = {
                "model": model or self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = client.post(
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=self.timeout_s,
            )
            payload = response.json()
            if response.status_code != 200 or "error" in payload:
                message = f"HTTP {response.status_code}"
                if isinstance(payload, dict):
                    error = payload.get("error")
                    if isinstance(error, dict) and error.get("message"):
                        message = str(error["message"])
                raise GeminiUpstreamRuntimeError(f"gemini upstream error: {message}")
            choices = payload.get("choices")
            if not choices:
                raise GeminiUpstreamRuntimeError(
                    "gemini upstream error: missing choices in response"
                )
            content = choices[0]["message"]["content"]
            return content, extract_inline_citations(content)
        except GeminiUpstreamRuntimeError:
            raise
        except Exception as exc:
            raise GeminiUpstreamRuntimeError(str(exc)) from exc
        finally:
            client.close()
