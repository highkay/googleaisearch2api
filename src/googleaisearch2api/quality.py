from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

_WHITESPACE_RE = re.compile(r"\s+")
_MULTI_ITEM_REQUEST_RE = re.compile(
    r"(最多|至少)?\s*(返回|列出|给出|提供|找出|推荐)\s*\d+\s*(条|个|项|只|家|篇|则)"
    r"|top\s*\d+|up to\s*\d+|at most\s*\d+",
    re.IGNORECASE,
)
_JSON_RESULTS_REQUEST_RE = re.compile(
    r"json\s*(对象|object)|\"results\"|输出格式固定|return\s+(only\s+)?a\s+json",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_RANGE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s*(?:至|到|~|—|–|\s+-\s+|through|to)\s*"
    r"(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_SINGLE_DAY_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?:当天|当日)")
_INTERNAL_SOURCE_LABEL_RE = re.compile(r"^[A-Za-z]+Result$")
_MALFORMED_STOCK_CODE_RE = re.compile(
    r"[（(][^）)]{0,40}\b\d{6}\b\s*/\s*0x[a-z0-9]*\s*[）)]",
    re.IGNORECASE,
)
_CITATION_MARKER_ARTIFACT_RE = re.compile(r"\[\s*\d+\s*\]")
_FOLLOW_UP_TAIL_PHRASES = (
    "如果您想进一步了解",
    "如果你想进一步了解",
    "如果您想深入了解",
    "如果你想深入了解",
    "若需我",
    "如需我",
    "如果需要，我可以",
    "would you like me to",
    "i can also help",
)
_SOURCE_LABEL_HINTS = (
    "财联社",
    "东方财富",
    "新浪",
    "腾讯",
    "网易",
    "搜狐",
    "凤凰",
    "证券时报",
    "证券日报",
    "中国证券报",
    "上海证券报",
    "第一财经",
    "每日经济新闻",
    "界面新闻",
    "澎湃新闻",
    "华尔街见闻",
    "央视",
    "新华社",
    "人民网",
    "环球网",
    "路透",
    "彭博",
    "Reuters",
    "Bloomberg",
    "CNBC",
)
_SOURCE_LABEL_SUFFIXES = (
    "网",
    "新闻",
    "财经",
    "时报",
    "日报",
    "证券报",
    "经济报",
    "快讯",
)
_GOOGLE_UTILITY_HOSTS = {
    "accounts.google.com",
    "myactivity.google.com",
    "policies.google.com",
    "support.google.com",
}
_ALLOWED_GOOGLE_SOURCE_HOSTS = {
    "books.google.com",
    "news.google.com",
    "patents.google.com",
    "scholar.google.com",
}
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
_STOCK_PROMPT_PHRASES = (
    "a股",
    "受益股",
    "个股",
    "股票",
    "证券",
    "ticker",
    "stock",
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


def _contains_follow_up_tail(answer_text: str) -> bool:
    return any(phrase.casefold() in answer_text for phrase in _FOLLOW_UP_TAIL_PHRASES)


def _standalone_source_label_count(answer: str) -> int:
    return sum(
        1
        for line in answer.splitlines()
        if _is_standalone_source_label(line.strip())
    )


def _is_standalone_source_label(line: str) -> bool:
    if not line or len(line) > 80:
        return False
    if ":" in line or "：" in line:
        return False
    if any(punctuation in line for punctuation in "。，,；;！？?（）()"):
        return False
    if any(hint in line for hint in _SOURCE_LABEL_HINTS):
        return True
    return len(line) <= 20 and line.endswith(_SOURCE_LABEL_SUFFIXES)


def _prompt_is_stock_related(prompt_text: str) -> bool:
    return any(phrase in prompt_text for phrase in _STOCK_PROMPT_PHRASES)


def _contains_malformed_stock_code(prompt_text: str, answer: str) -> bool:
    if not _prompt_is_stock_related(prompt_text):
        return False
    return bool(_MALFORMED_STOCK_CODE_RE.search(answer))


def _contains_citation_marker_artifact(value: object) -> bool:
    return bool(_CITATION_MARKER_ARTIFACT_RE.search(str(value or "")))


def _prompt_requests_multiple_items(prompt_text: str) -> bool:
    return bool(_MULTI_ITEM_REQUEST_RE.search(prompt_text))


def _prompt_requests_json_results(prompt_text: str) -> bool:
    return bool(_JSON_RESULTS_REQUEST_RE.search(prompt_text)) and "results" in prompt_text


def _is_usable_result_url(value: object) -> bool:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.hostname or ""
    if host in _GOOGLE_UTILITY_HOSTS:
        return False
    if host.endswith(".google.com") and host not in _ALLOWED_GOOGLE_SOURCE_HOSTS:
        return False
    return True


def _extract_requested_date_range(prompt: str) -> tuple[str, str] | None:
    range_match = _DATE_RANGE_RE.search(prompt)
    if range_match:
        start_date, end_date = range_match.groups()
        return (start_date, end_date) if start_date <= end_date else (end_date, start_date)

    single_day_match = _SINGLE_DAY_RE.search(prompt)
    if single_day_match:
        date = single_day_match.group(1)
        return date, date

    return None


def _published_date_is_in_range(published_date: str, date_range: tuple[str, str]) -> bool:
    start_date, end_date = date_range
    return start_date <= published_date <= end_date


def _published_date_is_future(published_date: str) -> bool:
    try:
        return date.fromisoformat(published_date) > date.today()
    except ValueError:
        return False


def normalize_answer_for_prompt(prompt: str, answer: str) -> str:
    prompt_text = _normalize(prompt)
    if not _prompt_requests_json_results(prompt_text):
        return answer

    date_range = _extract_requested_date_range(prompt)
    try:
        payload = json.loads(answer.strip())
    except json.JSONDecodeError:
        return answer
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return answer

    filtered_results: list[object] = []
    changed = False
    for item in payload["results"]:
        if isinstance(item, dict):
            published_date = str(item.get("published_date") or "").strip()
            if _DATE_RE.match(published_date):
                if _published_date_is_future(published_date):
                    changed = True
                    continue
                if date_range is not None and not _published_date_is_in_range(
                    published_date,
                    date_range,
                ):
                    changed = True
                    continue
        filtered_results.append(item)

    if not changed:
        return answer

    normalized_payload = dict(payload)
    normalized_payload["results"] = filtered_results
    return json.dumps(normalized_payload, ensure_ascii=False)


def _assess_json_results_answer(prompt: str, answer: str) -> AnswerQuality:
    prompt_text = _normalize(prompt)
    try:
        payload = json.loads(answer.strip())
    except json.JSONDecodeError:
        return AnswerQuality(False, "answer is not valid JSON for requested results")

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return AnswerQuality(False, "answer JSON does not contain a results list")

    if not payload["results"]:
        return AnswerQuality(False, "answer JSON results list is empty")

    if _contains_malformed_stock_code(prompt_text, answer):
        return AnswerQuality(False, "answer contains malformed stock code")

    date_range = _extract_requested_date_range(prompt)
    for item in payload["results"]:
        if not isinstance(item, dict):
            return AnswerQuality(False, "answer JSON result item is not an object")
        for field in ("title", "content", "source", "url", "published_date"):
            if not str(item.get(field) or "").strip():
                return AnswerQuality(False, f"answer JSON result is missing {field}")
        if _contains_citation_marker_artifact(item.get("content")):
            return AnswerQuality(False, "answer JSON result contains citation marker artifacts")
        if not _is_usable_result_url(item.get("url")):
            return AnswerQuality(False, "answer JSON result has an unusable URL")
        if not _DATE_RE.match(str(item.get("published_date") or "").strip()):
            return AnswerQuality(False, "answer JSON result has an invalid published_date")
        published_date = str(item.get("published_date") or "").strip()
        if _published_date_is_future(published_date):
            return AnswerQuality(False, "answer JSON result published_date is in the future")
        if date_range is not None and not _published_date_is_in_range(
            published_date,
            date_range,
        ):
            return AnswerQuality(
                False,
                "answer JSON result published_date is outside requested date range",
            )
        source = str(item.get("source") or "").strip()
        if _INTERNAL_SOURCE_LABEL_RE.match(source):
            return AnswerQuality(False, "answer JSON result source is an internal label")

    return AnswerQuality(True)


def assess_search_answer_quality(
    prompt: str,
    answer: str,
    citations: Sequence[object] | None = None,
) -> AnswerQuality:
    prompt_text = _normalize(prompt)
    answer_text = _normalize(answer)
    raw_answer = answer.strip()
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

    if _prompt_requests_json_results(prompt_text):
        return _assess_json_results_answer(prompt, raw_answer)

    if _contains_follow_up_tail(answer_text):
        return AnswerQuality(False, "answer contains a follow-up prompt tail")

    if _standalone_source_label_count(raw_answer) >= 2:
        return AnswerQuality(False, "answer contains standalone source labels")

    if _contains_malformed_stock_code(prompt_text, raw_answer):
        return AnswerQuality(False, "answer contains malformed stock code")

    if _prompt_requests_multiple_items(prompt_text) and len(answer_text) < 120:
        return AnswerQuality(False, "answer is too short for the requested list")

    if len(answer_text) < 80 and not _has_usable_citation(citations):
        return AnswerQuality(False, "short answer has no usable citations")

    return AnswerQuality(True)


def assess_google_answer_quality(
    prompt: str,
    answer: str,
    citations: Sequence[object] | None = None,
) -> AnswerQuality:
    return assess_search_answer_quality(prompt, answer, citations)
