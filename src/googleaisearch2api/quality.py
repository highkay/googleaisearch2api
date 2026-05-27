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
_YEAR_DATE_RANGE_RE = re.compile(
    r"\b\d{4}(?:-\d{2}-\d{2})?\s*(?:至|到|~|—|–|\s+-\s+|through|to)\s*"
    r"\d{4}(?:-\d{2}-\d{2})?\b",
    re.IGNORECASE,
)
_NON_SPECIFIC_URL_PATH_RE = re.compile(
    r"(?:^|/)(?:tags?|topics?|channels?|categories|keywords?|search|lists?|"
    r"specials?|subjects?|zt)(?:/|$)",
    re.IGNORECASE,
)
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
_RAW_URL_RE = re.compile(r"https?://[^\s<>\]\)）\"']+")
_DOMAIN_ONLY_LINE_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$",
    re.IGNORECASE,
)
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
_PLACEHOLDER_RESULT_HOSTS = {
    "0.0.0.0",
    "127.0.0.1",
    "::1",
    "example.com",
    "example.net",
    "example.org",
    "localhost",
}
_PLACEHOLDER_RESULT_HOST_SUFFIXES = (
    ".example.com",
    ".example.net",
    ".example.org",
    ".invalid",
    ".localhost",
    ".test",
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
_STOCK_PROMPT_PHRASES = (
    "a股",
    "受益股",
    "个股",
    "股票",
    "证券",
    "ticker",
    "stock",
)
_GENERIC_JSON_SOURCE_LABEL_PHRASES = (
    "综合行业报道",
    "综合行业资讯",
    "综合报道",
    "综合消息",
    "行业报道",
    "行业快报",
    "行业研究报告",
    "产业研究报告",
    "市场分析",
    "战略观察",
    "research report",
    "market analysis",
    "industry report",
)
_AGGREGATE_JSON_SOURCE_LABEL_PHRASES = (
    "综合",
    "汇总",
    "整理",
    "转载",
    "multiple sources",
    "various sources",
)
_COMPOSITE_JSON_SOURCE_LABEL_SEPARATORS = (
    " / ",
    "/",
    "|",
    "｜",
    "、",
)
_AGGREGATE_RAW_SOURCE_LABEL_PHRASES = (
    "综合",
    "汇总",
    "整理",
    "multiple sources",
    "various sources",
    "多家",
    "汇编",
    "专题",
)
_NON_SPECIFIC_RAW_DATE_PHRASES = (
    "见页",
    "页面无明确",
    "页内无明确",
    "未标注",
    "未注明",
    "无明确出版",
    "无明确日期",
    "相关时间点",
    "多篇",
    "多条",
    "汇编",
    "专题",
    "示例",
    "unknown",
    "not specified",
    "no explicit date",
)
_SEARCH_RESULTS_TAIL_MARKERS = {
    "search results",
    "搜索结果",
}


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


def _standalone_hostname_count(answer: str) -> int:
    return sum(
        1
        for line in answer.splitlines()
        if _DOMAIN_ONLY_LINE_RE.match(line.strip())
    )


def _contains_search_results_tail(answer: str) -> bool:
    return any(
        line.strip().casefold() in _SEARCH_RESULTS_TAIL_MARKERS
        for line in answer.splitlines()
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
    host = (parsed.hostname or "").casefold()
    if host in _PLACEHOLDER_RESULT_HOSTS:
        return False
    if any(host.endswith(suffix) for suffix in _PLACEHOLDER_RESULT_HOST_SUFFIXES):
        return False
    if host in _GOOGLE_UTILITY_HOSTS:
        return False
    if host.endswith(".google.com") and host not in _ALLOWED_GOOGLE_SOURCE_HOSTS:
        return False
    return True


def _is_specific_result_url(value: object) -> bool:
    parsed = urlparse(str(value or "").strip())
    path = parsed.path.strip("/")
    if not path and not parsed.query:
        return False
    if _NON_SPECIFIC_URL_PATH_RE.search(path):
        return False
    if parsed.path.casefold().rstrip("/") in {"/tag", "/tags", "/topic", "/topics"}:
        return False
    query = parsed.query.casefold()
    return not (path in {"search", "s"} and query)


def _extract_raw_urls(answer: str) -> list[str]:
    urls: list[str] = []
    for match in _RAW_URL_RE.finditer(answer):
        url = match.group(0).rstrip(".,，。；;:：、")
        if _is_usable_result_url(url):
            urls.append(url)
    return urls


def _contains_non_specific_raw_url(answer: str) -> bool:
    return any(not _is_specific_result_url(url) for url in _extract_raw_urls(answer))


def _contains_duplicate_raw_url(answer: str) -> bool:
    seen: set[str] = set()
    for url in _extract_raw_urls(answer):
        if url in seen:
            return True
        seen.add(url)
    return False


def _iter_raw_source_labels(answer: str) -> list[str]:
    labels: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("来源：", "来源:")):
            labels.append(line.split("：", 1)[-1].split(":", 1)[-1].strip())
            continue
        if "http://" not in line and "https://" not in line:
            continue
        parts = re.split(r"\s+[—–-]\s+", line)
        if len(parts) >= 4:
            labels.append(parts[1].strip())
    return labels


def _contains_aggregate_raw_source_label(answer: str) -> bool:
    for source in _iter_raw_source_labels(answer):
        normalized = _normalize(source)
        if any(phrase in normalized for phrase in _AGGREGATE_RAW_SOURCE_LABEL_PHRASES):
            return True
        if any(separator in source for separator in _COMPOSITE_JSON_SOURCE_LABEL_SEPARATORS):
            return True
    return False


def _contains_non_specific_raw_date(answer: str) -> bool:
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        date_text = ""
        if line.startswith(("日期：", "日期:", "date:", "Date:")):
            date_text = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        elif "http://" in line or "https://" in line:
            parts = re.split(r"\s+[—–-]\s+", line)
            if len(parts) >= 4:
                date_text = parts[-2].strip()
        if not date_text:
            continue
        normalized = _normalize(date_text)
        if _YEAR_DATE_RANGE_RE.search(date_text):
            return True
        if any(
            phrase in normalized for phrase in _NON_SPECIFIC_RAW_DATE_PHRASES
        ):
            return True
    return False


def _is_generic_json_source_label(value: object) -> bool:
    source = _normalize(str(value or ""))
    return any(phrase in source for phrase in _GENERIC_JSON_SOURCE_LABEL_PHRASES)


def _is_aggregate_json_source_label(value: object) -> bool:
    raw_source = str(value or "").strip()
    source = _normalize(raw_source)
    if any(phrase in source for phrase in _AGGREGATE_JSON_SOURCE_LABEL_PHRASES):
        return True
    return any(separator in raw_source for separator in _COMPOSITE_JSON_SOURCE_LABEL_SEPARATORS)


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
    seen_urls: set[str] = set()
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
        url = str(item.get("url") or "").strip()
        if not _is_specific_result_url(url):
            return AnswerQuality(False, "answer JSON result URL is not specific")
        if url in seen_urls:
            return AnswerQuality(False, "answer JSON result URL is duplicated")
        seen_urls.add(url)
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
        if _is_generic_json_source_label(source):
            return AnswerQuality(False, "answer JSON result source is generic")
        if _is_aggregate_json_source_label(source):
            return AnswerQuality(False, "answer JSON result source is aggregate")

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

    if _contains_search_results_tail(raw_answer):
        return AnswerQuality(False, "answer contains search results UI tail")

    if _standalone_source_label_count(raw_answer) >= 2:
        return AnswerQuality(False, "answer contains standalone source labels")

    if _standalone_hostname_count(raw_answer) >= 2:
        return AnswerQuality(False, "answer contains standalone hostnames")

    if _contains_aggregate_raw_source_label(raw_answer):
        return AnswerQuality(False, "answer contains aggregate source labels")

    if _contains_non_specific_raw_date(raw_answer):
        return AnswerQuality(False, "answer contains non-specific publication dates")

    if _prompt_requests_multiple_items(prompt_text) and _contains_non_specific_raw_url(
        raw_answer
    ):
        return AnswerQuality(False, "answer contains non-specific URLs")

    if _contains_duplicate_raw_url(raw_answer):
        return AnswerQuality(False, "answer contains duplicate URLs")

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
