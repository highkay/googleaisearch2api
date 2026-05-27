from __future__ import annotations

import re
from dataclasses import dataclass

_WHITESPACE_RE = re.compile(r"\s+")
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


def assess_google_answer_quality(prompt: str, answer: str) -> AnswerQuality:
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

    return AnswerQuality(True)
