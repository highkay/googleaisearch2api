from __future__ import annotations

import re

_JSON_RESULTS_HINT_RE = re.compile(r'(?is)\bjson\b|["“”]results["“”]|输出格式固定')
_QUESTION_LABEL_RE = re.compile(
    r"(?:^|[\n。])\s*(?:问题|查询|搜索请求)\s*[:：]\s*(?P<question>.+)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SITE_OPERATOR_RE = re.compile(r"(?i)\bsite:\S+")
_HARD_DATE_WINDOW_RE = re.compile(
    r"(?is)(?:时间范围必须限制在|时间范围优先|时间范围)[^。；;\n]{0,120}"
)
_MAX_RESULTS_RE = re.compile(r"(?is)(?:最多返回|最多引用约|最多引用)\s*\d+\s*条[^。；;\n]{0,40}")
_STRICT_SOURCE_RE = re.compile(
    r"(?is)(?:只保留直接相关[^。；;\n]{0,40}|限定站点[：:][^。；;\n]+|补充要求[：:][^。；;\n]+)"
)
_SEARCH_WRAPPER_RE = re.compile(
    r"(?is)^(?:搜索并用自然语言简要回答[^：:\n]*[：:]\s*|"
    r"Search and answer briefly in natural language[^:\n]*:\s*)"
)
_USER_REQUEST_PREFIX_RE = re.compile(r"(?is)^user request\s*:\s*")


def simplify_search_prompt(prompt: str) -> str:
    """Collapse JSON-results wrapper prompts into a direct natural-language search ask."""
    stripped = prompt.strip()
    if not stripped or not _JSON_RESULTS_HINT_RE.search(stripped):
        return stripped

    match = _QUESTION_LABEL_RE.search(stripped)
    if match is None:
        return stripped

    question = _compact_text(match.group("question"))
    if not question:
        return stripped

    if _CJK_RE.search(question):
        return (
            "搜索并用自然语言简要回答，列出关键发现、来源和日期；"
            "如果没有足够直接相关的信息，直接说明未找到：\n"
            f"{question}"
        )
    return (
        "Search and answer briefly in natural language with key findings, sources, "
        "and dates. If there is not enough directly relevant information, say so:\n"
        f"{question}"
    )


def adapt_prompt_for_engine(prompt: str, *, engine: str) -> str:
    """Engine-specific prompt shaping after shared simplify.

    Google can keep retrieval-shaped asks. Duck.ai and Gemini web are
    conversational NL assistants: keyword piles, site: operators, and hard
    SERP constraints cause keyword echo and weak answers in production
    (2026-07-17 request_logs).
    """
    simplified = simplify_search_prompt(prompt)
    if (engine or "").strip().lower() not in {"duck", "gemini"}:
        return simplified
    return _naturalize_for_duck(simplified)


def _naturalize_for_duck(prompt: str) -> str:
    stripped = prompt.strip()
    if not stripped:
        return stripped

    body = stripped
    wrapper_match = _SEARCH_WRAPPER_RE.match(stripped)
    if wrapper_match is not None:
        body = stripped[wrapper_match.end() :].strip()
    body = _USER_REQUEST_PREFIX_RE.sub("", body).strip()

    body = _SITE_OPERATOR_RE.sub(" ", body)
    body = _HARD_DATE_WINDOW_RE.sub(" ", body)
    body = _MAX_RESULTS_RE.sub(" ", body)
    body = _STRICT_SOURCE_RE.sub(" ", body)
    body = _compact_text(body)
    if not body:
        body = _compact_text(stripped)

    # Already a clear question — keep tone, still re-wrap with Duck-friendly instructions.
    if _looks_like_natural_question(body):
        question = body
    else:
        question = _keyword_pile_to_question(body)

    if _CJK_RE.search(question):
        return (
            "请用自然语言完整回答下面的问题：说明关键事实，并在有把握时给出可核对的来源名称、"
            "日期和链接；如果没有直接证据，就明确说未找到，不要堆砌搜索词或站点运算符。\n"
            f"{question}"
        )
    return (
        "Answer the following in natural language with complete sentences. When confident, "
        "include verifiable source names, dates, and links. If there is no direct evidence, "
        "say so clearly — do not dump search keywords or site operators.\n"
        f"{question}"
    )


def _looks_like_natural_question(text: str) -> bool:
    if "?" in text or "？" in text:
        return True
    if re.search(r"(请|如何|为什么|是否|什么|哪些|怎么)", text):
        return True
    # Short keyword piles without sentence punctuation are not NL questions.
    if "。" in text or "，" in text or "," in text:
        # Still may be a comma-separated keyword list; require a verb-ish cue or length.
        if re.search(r"(是|有|发生|发布|影响|说明|介绍|概述)", text):
            return True
    tokens = text.split()
    return len(tokens) >= 12 and (" " in text) and not _mostly_keyword_tokens(tokens)


def _mostly_keyword_tokens(tokens: list[str]) -> bool:
    if len(tokens) < 4:
        return True
    short = sum(1 for token in tokens if len(token) <= 6)
    return short / len(tokens) >= 0.7


def _keyword_pile_to_question(body: str) -> str:
    # Preserve order; turn a retrieval bag into one ask without inventing facts.
    if _CJK_RE.search(body):
        return (
            f"请围绕「{body}」用完整句子说明目前能核对到的公开信息："
            "关键主体、时间线索、发生了什么、以及依据来源；没有直接材料就直接说没有。"
        )
    return (
        f"Please explain in complete sentences what publicly verifiable information is "
        f"available about: {body}. Cover the main subject, timing, what happened, and "
        "sources when available; say clearly if nothing direct is found."
    )


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
