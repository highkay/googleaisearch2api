from __future__ import annotations

import json
import re
import threading

from patchright.sync_api import Error as PatchrightError
from patchright.sync_api import TimeoutError as PatchrightTimeoutError
from patchright.sync_api import sync_playwright

from .browser import DEFAULT_BROWSER_CHANNEL, resolve_browser_proxy, resolve_browser_user_agent
from .config import ServiceConfig
from .proxy_bridge import (
    LocalSocksProxyBridge,
    build_socks_proxy_target,
    is_socks_proxy_server,
)
from .schemas import GoogleAiResult

DUCK_AI_URL = "https://duck.ai/"
DUCK_CHAT_ENDPOINT_MARKER = "/duckchat/v1/chat"
DUCK_RATE_LIMIT_MARKERS = [
    "too many requests",
    "take a short break",
    "try again later",
    "please try again later",
]
DOMAIN_ONLY_LINE_RE = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$",
    re.IGNORECASE,
)
SEARCH_RESULTS_TAIL_MARKERS = {
    "Search Results",
    "搜索结果",
}
FOLLOW_UP_TAIL_MARKERS = [
    "\u5982\u679c\u60a8\u60f3\u8fdb\u4e00\u6b65\u4e86\u89e3",
    "\u5982\u679c\u4f60\u60f3\u8fdb\u4e00\u6b65\u4e86\u89e3",
    "\u5982\u679c\u60a8\u60f3\u6df1\u5165\u4e86\u89e3",
    "\u5982\u679c\u4f60\u60f3\u6df1\u5165\u4e86\u89e3",
    "\u82e5\u9700\u6211",
    "\u5982\u9700\u6211",
    "\u5982\u679c\u9700\u8981\uff0c\u6211\u53ef\u4ee5",
    "Would you like me to",
    "I can also help",
]
DUCK_SEARCH_PROMPT_TEMPLATE = """You are being used as a search answer engine.
Answer the user's request directly with concrete, useful findings.
Do not return only search suggestions.
Do not ask whether to search.
If the user asks for a fixed output format, follow it exactly.
For stock answers, only include ticker symbols you are confident are real.
For A-share answers, use only real six-digit A-share codes; omit uncertain codes.
Do not output placeholder codes such as 0x, N/A, or unknown.
For source URLs, use only real public web pages from named sources.
Use one concrete publisher name in each source field, matching the URL host.
Do not combine multiple publishers, reprints, or aggregate labels in source fields.
If the URL host is not the named publisher's host, use the URL host's publisher
name or skip that result.
Do not include the raw search query, standalone hostnames, raw search result listings,
or duplicate URLs.
Use one specific page URL for each result; do not use homepages, topic pages, or
date ranges as publication dates.
Each result must have a concrete publication date from the page. If a page has
no explicit date, skip that result instead of writing "see page", "unknown", or
similar placeholders.
If the user writes Chinese, answer in Chinese and use Chinese field labels:
来源, 日期, 链接, 为什么相关. Do not use Japanese labels such as リンク.
Do not output placeholder URLs such as example.com.
If you cannot verify a real source URL, return an empty results list instead of invented results.

User request:
{prompt}"""


class DuckAiRuntimeError(RuntimeError):
    pass


class DuckAiRateLimitedError(DuckAiRuntimeError):
    pass


class DuckAiTimeoutError(DuckAiRuntimeError):
    pass


def build_duck_search_prompt(prompt: str) -> str:
    return DUCK_SEARCH_PROMPT_TEMPLATE.format(prompt=prompt.strip())


class DuckAiRunner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._playwright = None
        self._browser = None
        self._proxy_bridge: LocalSocksProxyBridge | None = None
        self._session_signature: tuple | None = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def reset(self) -> None:
        self.close()

    def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
        prompt = prompt.strip()
        if not prompt:
            raise DuckAiRuntimeError("Prompt is empty.")
        submitted_prompt = build_duck_search_prompt(prompt)

        with self._lock:
            for attempt in range(2):
                context = None
                page = None
                try:
                    self._ensure_browser_locked(config)
                    context = self._new_context(config)
                    page = context.new_page()
                    page.set_default_timeout(config.browser_timeout_ms)
                    return self._run_prompt_once(page, config, submitted_prompt)
                except (DuckAiRateLimitedError, DuckAiTimeoutError, DuckAiRuntimeError):
                    raise
                except PatchrightError as exc:
                    self._close_locked()
                    if attempt == 0:
                        continue
                    raise DuckAiRuntimeError(
                        f"Patchright browser runtime failure: {exc}"
                    ) from exc
                finally:
                    if page is not None:
                        try:
                            page.close()
                        except Exception:
                            pass
                    if context is not None:
                        try:
                            context.close()
                        except Exception:
                            pass

        raise DuckAiRuntimeError("Duck.ai request failed after browser recovery.")

    def _run_prompt_once(self, page, config: ServiceConfig, prompt: str) -> GoogleAiResult:
        page.goto(DUCK_AI_URL, wait_until="domcontentloaded", timeout=config.browser_timeout_ms)
        self._ensure_not_rate_limited(page, stage="opening chat page")

        try:
            textarea = page.locator("textarea").first
            textarea.wait_for(state="visible", timeout=min(config.browser_timeout_ms, 20_000))
            textarea.fill(prompt)
            page.wait_for_function(
                "() => {"
                " const button = document.querySelector('button[type=\"submit\"]');"
                " return button && !button.disabled;"
                "}",
                timeout=min(config.browser_timeout_ms, 10_000),
            )
        except PatchrightTimeoutError as exc:
            self._ensure_not_rate_limited(page, stage="preparing chat page")
            raise DuckAiTimeoutError("Duck.ai chat input did not become ready.") from exc

        try:
            with page.expect_response(
                lambda response: DUCK_CHAT_ENDPOINT_MARKER in response.url
                and response.request.method == "POST",
                timeout=config.answer_timeout_ms,
            ) as chat_response_info:
                page.locator("button[type='submit']").first.click()
            chat_response = chat_response_info.value
        except PatchrightTimeoutError as exc:
            self._ensure_not_rate_limited(page, stage="submitting chat request")
            raise DuckAiTimeoutError("Duck.ai chat request did not start.") from exc

        if chat_response.status == 429:
            raise DuckAiRateLimitedError("Duck.ai returned HTTP 429 for the chat request.")
        if chat_response.status >= 400:
            raise DuckAiRuntimeError(
                f"Duck.ai chat request failed with HTTP {chat_response.status}."
            )

        try:
            page.wait_for_function(
                "() => {"
                " const text = document.body ? document.body.innerText : '';"
                " const generating = text.includes('Generating response')"
                "   || text.includes('\\u6b63\\u5728\\u751f\\u6210');"
                " const stop = document.querySelector("
                "   'button[aria-label=\"Stop generating\"],"
                "    button[aria-label=\"\\u505c\\u6b62\\u751f\\u6210\"]'"
                " );"
                " return !generating && (!stop || stop.disabled);"
                "}",
                timeout=config.answer_timeout_ms,
            )
        except PatchrightTimeoutError:
            pass

        body_text = _normalize_text(page.locator("body").inner_text(timeout=5_000))
        if _is_rate_limited_text(body_text):
            raise DuckAiRateLimitedError("Duck.ai page reported too many requests.")

        answer_text = extract_duck_answer_text(body_text, prompt)
        if not answer_text:
            raise DuckAiRuntimeError("Duck.ai returned no extractable answer text.")

        return GoogleAiResult(
            answer_text=answer_text,
            citations=[],
            final_url=page.url,
            page_title=page.title(),
            body_excerpt=body_text[:800],
        )

    def _ensure_browser_locked(self, config: ServiceConfig):
        signature = self._build_session_signature(config)
        if (
            self._browser is not None
            and self._session_signature == signature
            and self._browser.is_connected()
        ):
            return self._browser

        self._close_locked()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(**self._build_launch_kwargs(config))
        self._session_signature = signature
        return self._browser

    def _new_context(self, config: ServiceConfig):
        context_kwargs = {"locale": config.browser_locale}
        browser_user_agent = resolve_browser_user_agent(config)
        if browser_user_agent:
            context_kwargs["user_agent"] = browser_user_agent
        return self._browser.new_context(**context_kwargs)

    def _build_session_signature(self, config: ServiceConfig) -> tuple:
        return (
            config.browser_headless,
            config.browser_user_agent,
            config.browser_locale,
            config.browser_proxy_server,
            config.browser_proxy_username,
            config.browser_proxy_password,
            config.browser_proxy_bypass,
        )

    def _close_locked(self) -> None:
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

        if self._proxy_bridge is not None:
            self._proxy_bridge.stop()
            self._proxy_bridge = None

        self._session_signature = None

    def _build_launch_kwargs(self, config: ServiceConfig) -> dict:
        kwargs: dict = {
            "headless": config.browser_headless,
            "channel": DEFAULT_BROWSER_CHANNEL,
        }

        browser_user_agent = resolve_browser_user_agent(config)
        if browser_user_agent:
            kwargs["args"] = [f"--user-agent={browser_user_agent}"]

        browser_proxy = self._resolve_launch_proxy_locked(config)
        if browser_proxy:
            kwargs["proxy"] = browser_proxy
        return kwargs

    def _resolve_launch_proxy_locked(self, config: ServiceConfig) -> dict | None:
        browser_proxy = resolve_browser_proxy(config)
        if not browser_proxy:
            return None

        proxy_server = browser_proxy["server"]
        if not is_socks_proxy_server(proxy_server):
            return browser_proxy

        target = build_socks_proxy_target(
            proxy_server,
            username=browser_proxy.get("username"),
            password=browser_proxy.get("password"),
        )
        if self._proxy_bridge is not None:
            self._proxy_bridge.stop()
        self._proxy_bridge = LocalSocksProxyBridge(target)
        self._proxy_bridge.start()
        return {
            "server": self._proxy_bridge.server_url,
            "bypass": browser_proxy.get("bypass"),
        }

    def _ensure_not_rate_limited(self, page, stage: str) -> None:
        try:
            body_text = page.locator("body").inner_text(timeout=3_000)
        except Exception:
            body_text = ""
        if _is_rate_limited_text(body_text):
            raise DuckAiRateLimitedError(f"Duck.ai rate limited the session while {stage}.")


def extract_duck_answer_text(body_text: str, prompt: str, *, limit: int = 3000) -> str:
    body_text = _normalize_text(body_text)
    prompt = _normalize_text(prompt)
    if not body_text:
        return ""

    if prompt and prompt in body_text:
        body_text = body_text.rsplit(prompt, 1)[1]

    skip_patterns = [
        re.compile(r"^GPT[-\w .]+$", re.IGNORECASE),
        re.compile(
            r"^(Duck\.ai|DuckDuckGo|Generating response|Tools|Fast|Private|Free|\u00b7)$",
            re.IGNORECASE,
        ),
        re.compile(r"^All chats are private", re.IGNORECASE),
        re.compile(r"^\u6240\u6709\u804a\u5929\u8bb0\u5f55\u5747\u4e3a"),
        re.compile(r"^DuckDuckGo \u5df2\u533f\u540d\u5904\u7406"),
        re.compile(r"^(\u5de5\u5177|\u5feb\u901f|\u804a\u5929)$"),
        re.compile(r"^(Searching the web|Search the web|Hide Reasoning)", re.IGNORECASE),
        re.compile(r"^(\u6b63\u5728\u641c\u7d22|\u9690\u85cf\u63a8\u7406)"),
    ]

    lines: list[str] = []
    for raw_line in body_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in SEARCH_RESULTS_TAIL_MARKERS:
            break
        if any(pattern.search(line) for pattern in skip_patterns):
            continue
        if DOMAIN_ONLY_LINE_RE.match(line):
            continue
        lines.append(line)

    lines = _strip_leading_search_query_line(lines)
    answer = _normalize_text("\n".join(lines))
    if _prompt_requests_json_results(prompt):
        json_answer = _extract_json_results_object(answer)
        if json_answer:
            return json_answer[:limit]

    return _strip_follow_up_tail(answer)[:limit]


def _strip_leading_search_query_line(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines
    first_line = lines[0]
    following_lines = lines[1:4]
    if _looks_like_search_query_line(first_line) and any(
        _looks_like_answer_item_start(line) for line in following_lines
    ):
        return lines[1:]
    return lines


def _looks_like_search_query_line(line: str) -> bool:
    return (
        len(line) <= 140
        and " " in line
        and not re.search(r"https?://|[：:。！？!?—–-]", line)
    )


def _looks_like_answer_item_start(line: str) -> bool:
    if re.search(r"https?://", line):
        return True
    return bool(re.search(r"^(标题|Title|来源|Source|[-*\d]+[.)、])", line))


def _prompt_requests_json_results(prompt: str) -> bool:
    normalized = (prompt or "").lower()
    return "json" in normalized and "results" in normalized


def _extract_json_results_object(text: str) -> str:
    for start_index, char in enumerate(text):
        if char != "{":
            continue

        depth = 0
        in_string = False
        escaped = False
        for index in range(start_index, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_index : index + 1]
                    try:
                        payload = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                        return json.dumps(payload, ensure_ascii=False)
                    break

    return ""


def _strip_follow_up_tail(answer: str) -> str:
    end_index: int | None = None
    for marker in FOLLOW_UP_TAIL_MARKERS:
        marker_index = answer.find(marker)
        if marker_index >= 0 and (end_index is None or marker_index < end_index):
            end_index = marker_index
    if end_index is None:
        return answer
    return answer[:end_index].strip()


def _normalize_text(value: str) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _is_rate_limited_text(value: str) -> bool:
    normalized = " ".join((value or "").lower().split())
    return any(marker in normalized for marker in DUCK_RATE_LIMIT_MARKERS)
