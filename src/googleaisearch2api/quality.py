from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_WHITESPACE_RE = re.compile(r"\s+")
_MULTI_ITEM_REQUEST_RE = re.compile(
    r"(最多|至少)?\s*(返回|列出|给出|提供|找出|推荐)\s*\d+\s*(条|个|项|只|家|篇|则)"
    r"|top\s*\d+|up to\s*\d+|at most\s*\d+",
    re.IGNORECASE,
)
_GENERIC_CLARIFICATION_PHRASES = (
    "what would you like to know",
    "could you please clarify",
    "please clarify",
    "please specify",
    "can you clarify",
    "你想要关于",
    "你想了解",
    "请说明具体",
    "请具体说明",
    "需要哪方面",
    "哪方面信息",
)


@dataclass(frozen=True, slots=True)
class AnswerQuality:
    ok: bool
    reason: str | None = None


def _normalize(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip()).casefold()


def _citation_url(citation: object) -> str:
    if isinstance(citation, dict):
        return str(citation.get("url") or "")
    return str(getattr(citation, "url", "") or "")


def _has_usable_citation(citations: Sequence[object] | None) -> bool:
    return any(_citation_url(citation).strip() for citation in citations or [])


def _prompt_requests_multiple_items(prompt_text: str) -> bool:
    return bool(_MULTI_ITEM_REQUEST_RE.search(prompt_text))


def assess_google_answer_quality(
    prompt: str,
    answer: str,
    citations: Sequence[object] | None = None,
) -> AnswerQuality:
    prompt_text = _normalize(prompt)
    answer_text = _normalize(answer)
    if not answer_text:
        return AnswerQuality(False, "empty answer")

    if answer_text.startswith(("you said", "you asked", "你说", "你问的是", "你刚才说")):
        return AnswerQuality(False, "answer appears to echo the prompt")

    if len(prompt_text) >= 12:
        if answer_text == prompt_text or answer_text.startswith(prompt_text):
            return AnswerQuality(False, "answer appears to echo the prompt")
        if prompt_text in answer_text and len(answer_text) <= len(prompt_text) + 80:
            return AnswerQuality(False, "answer mostly contains the prompt")

    if len(answer_text) < 220 and any(
        phrase in answer_text for phrase in _GENERIC_CLARIFICATION_PHRASES
    ):
        return AnswerQuality(False, "answer is a generic clarification")

    if _prompt_requests_multiple_items(prompt_text) and len(answer_text) < 120:
        return AnswerQuality(False, "answer is too short for the requested list")

    if len(answer_text) < 80 and not _has_usable_citation(citations):
        return AnswerQuality(False, "short answer has no usable citations")

    return AnswerQuality(True)
