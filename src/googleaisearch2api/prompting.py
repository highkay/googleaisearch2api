from __future__ import annotations

import re

_JSON_RESULTS_HINT_RE = re.compile(r'(?is)\bjson\b|["“”]results["“”]|输出格式固定')
_QUESTION_LABEL_RE = re.compile(
    r"(?:^|[\n。])\s*(?:问题|查询|搜索请求)\s*[:：]\s*(?P<question>.+)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


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


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
