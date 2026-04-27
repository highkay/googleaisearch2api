from __future__ import annotations

import threading
import time
from textwrap import shorten
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlsplit, urlunparse, urlunsplit

from patchright.sync_api import Error as PatchrightError
from patchright.sync_api import TimeoutError as PatchrightTimeoutError
from patchright.sync_api import sync_playwright

from .config import ServiceConfig
from .schemas import Citation, GoogleAiResult

ANSWER_SELECTORS = [
    "div.mZJni.Dn7Fzd",
    "div.pWvJNd",
    "div.Zkbeff",
    "div.CKgc1d",
    "div.WzWwpc.vve6Ce.CZntF",
    "section",
]

DISCLAIMER_MARKERS = [
    "AI 的回答未必正确无误，请注意核查",
    "AI's response may contain mistakes",
    "AI responses may contain mistakes",
]

BLOCKED_MARKERS = [
    "unusual traffic",
    "not a robot",
    "captcha",
    "抱歉",
    "验证您不是机器人",
]

DEFAULT_BROWSER_CHANNEL = "chrome"
DEFAULT_CHROME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)

EXTRACT_SCRIPT = """
({ query, selectors }) => {
  const clean = (value) => (value || "").trim();
  const root = selectors
    .map((selector) => document.querySelector(selector))
    .find((element) => element && clean(element.innerText).length > 80);
  const section =
    (root && root.closest("section")) || document.querySelector("section") || document.body;
  let answerText = root ? clean(root.innerText) : "";
  if (query && answerText.startsWith(query)) {
    answerText = clean(answerText.slice(query.length));
  }
  const citations = [];
  const seen = new Set();
  section.querySelectorAll("a[href]").forEach((anchor) => {
    const href = anchor.href;
    const title = clean(anchor.innerText || anchor.getAttribute("aria-label") || "");
    if (!href || seen.has(href) || !/^https?:/i.test(href)) {
      return;
    }
    if (href.includes("policies.google.com") || href.includes("support.google.com/legal")) {
      return;
    }
    seen.add(href);
    citations.push({
      title: title || new URL(href).hostname,
      url: href,
    });
  });
  return {
    answerText,
    citations: citations.slice(0, 10),
    finalUrl: location.href,
    pageTitle: document.title,
    bodyExcerpt: clean(document.body.innerText).slice(0, 800),
  };
}
"""


class GoogleAiRuntimeError(RuntimeError):
    pass


class GoogleAiBlockedError(GoogleAiRuntimeError):
    pass


class GoogleAiTimeoutError(GoogleAiRuntimeError):
    pass


def clean_answer_text(answer_text: str, query: str) -> str:
    cleaned = (answer_text or "").strip()
    if query and cleaned.startswith(query):
        cleaned = cleaned[len(query) :].strip()
    for marker in DISCLAIMER_MARKERS:
        marker_index = cleaned.find(marker)
        if marker_index >= 0:
            cleaned = cleaned[:marker_index].strip()
    return cleaned.strip()


def filter_citations(citations: list[dict]) -> list[Citation]:
    filtered: list[Citation] = []
    seen: set[str] = set()
    for item in citations:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        filtered.append(Citation(title=title or url, url=url))
    return filtered


def resolve_browser_user_agent(config: ServiceConfig) -> str | None:
    if config.browser_user_agent:
        return config.browser_user_agent
    if not config.browser_headless:
        return None
    return DEFAULT_CHROME_USER_AGENT


def _split_proxy_server_auth(proxy_server: str) -> tuple[str, str | None, str | None]:
    parsed = urlsplit(proxy_server)
    if not parsed.scheme or not parsed.netloc or "@" not in parsed.netloc:
        return proxy_server, None, None

    server_netloc = parsed.netloc.rsplit("@", 1)[1]
    server = urlunsplit((parsed.scheme, server_netloc, "", "", ""))
    username = unquote(parsed.username) if parsed.username is not None else None
    password = unquote(parsed.password) if parsed.password is not None else None
    return server, username, password


def resolve_browser_proxy(config: ServiceConfig) -> dict | None:
    if not config.proxy_enabled or not config.browser_proxy_server:
        return None

    server, embedded_username, embedded_password = _split_proxy_server_auth(
        config.browser_proxy_server
    )
    return {
        "server": server,
        "username": config.browser_proxy_username or embedded_username,
        "password": config.browser_proxy_password or embedded_password,
        "bypass": config.browser_proxy_bypass,
    }


class GoogleAiRunner:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._playwright = None
        self._browser = None
        self._context = None
        self._session_signature: tuple | None = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def reset(self) -> None:
        self.close()

    def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
        prompt = prompt.strip()
        if not prompt:
            raise GoogleAiRuntimeError("Prompt is empty.")

        with self._lock:
            for attempt in range(2):
                page = None
                try:
                    context = self._ensure_context_locked(config)
                    page = context.new_page()
                    page.set_default_timeout(config.browser_timeout_ms)
                    return self._run_prompt_once(page, config, prompt)
                except (GoogleAiBlockedError, GoogleAiTimeoutError, GoogleAiRuntimeError):
                    raise
                except PatchrightError as exc:
                    self._close_locked()
                    if attempt == 0:
                        continue
                    raise GoogleAiRuntimeError(
                        f"Patchright browser runtime failure: {exc}"
                    ) from exc
                finally:
                    if page is not None:
                        try:
                            page.close()
                        except Exception:
                            pass

        raise GoogleAiRuntimeError("Google AI request failed after browser recovery.")

    def _run_prompt_once(self, page, config: ServiceConfig, prompt: str) -> GoogleAiResult:
        query_url = self._build_query_url(config.browser_base_url, prompt)
        submitted = False
        page.goto(
            query_url,
            wait_until="domcontentloaded",
            timeout=config.browser_timeout_ms,
        )
        self._ensure_not_blocked(page, stage="opening query page")

        try:
            return self._wait_for_answer(
                page,
                prompt,
                min(config.answer_timeout_ms, 15_000),
            )
        except GoogleAiTimeoutError:
            page.goto(
                config.browser_base_url,
                wait_until="domcontentloaded",
                timeout=config.browser_timeout_ms,
            )
            self._ensure_not_blocked(page, stage="opening base page")
            submitted = self._submit_query(page, prompt)
            if not submitted:
                page.goto(
                    query_url,
                    wait_until="domcontentloaded",
                    timeout=config.browser_timeout_ms,
                )

        if not submitted:
            page.goto(
                query_url,
                wait_until="domcontentloaded",
                timeout=config.browser_timeout_ms,
            )
        return self._wait_for_answer(page, prompt, config.answer_timeout_ms)

    def _ensure_context_locked(self, config: ServiceConfig):
        signature = self._build_session_signature(config)
        if (
            self._context is not None
            and self._browser is not None
            and self._session_signature == signature
            and self._browser.is_connected()
        ):
            return self._context

        self._close_locked()
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(**self._build_launch_kwargs(config))
        context_kwargs = {"locale": config.browser_locale}
        browser_user_agent = resolve_browser_user_agent(config)
        if browser_user_agent:
            context_kwargs["user_agent"] = browser_user_agent
        self._context = self._browser.new_context(**context_kwargs)
        self._session_signature = signature
        return self._context

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
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None

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

        self._session_signature = None

    def _build_launch_kwargs(self, config: ServiceConfig) -> dict:
        kwargs: dict = {
            "headless": config.browser_headless,
            "channel": DEFAULT_BROWSER_CHANNEL,
        }

        browser_user_agent = resolve_browser_user_agent(config)
        if browser_user_agent:
            kwargs["args"] = [f"--user-agent={browser_user_agent}"]

        browser_proxy = resolve_browser_proxy(config)
        if browser_proxy:
            kwargs["proxy"] = browser_proxy
        return kwargs

    def _submit_query(self, page, prompt: str) -> bool:
        try:
            textarea = page.locator("textarea").first
            textarea.wait_for(state="visible", timeout=15_000)
            textarea.click()
            textarea.fill(prompt)
            textarea.press("Enter")
            return True
        except PatchrightTimeoutError:
            return False

    def _build_query_url(self, base_url: str, prompt: str) -> str:
        parsed = urlparse(base_url)
        query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query_params["q"] = prompt
        new_query = urlencode(query_params)
        return urlunparse(parsed._replace(query=new_query))

    def _ensure_not_blocked(self, page, stage: str) -> None:
        url = (page.url or "").lower()
        body_text = ""
        try:
            body_text = page.locator("body").inner_text(timeout=3_000).lower()
        except PatchrightTimeoutError:
            body_text = ""
        if "/sorry/" in url or any(marker in body_text for marker in BLOCKED_MARKERS):
            excerpt = shorten(body_text, width=240, placeholder="...")
            raise GoogleAiBlockedError(f"Google blocked the session while {stage}: {excerpt}")

    def _extract_payload(self, page, prompt: str) -> dict:
        payload = page.evaluate(
            EXTRACT_SCRIPT,
            {
                "query": prompt,
                "selectors": ANSWER_SELECTORS,
            },
        )
        payload["answerText"] = clean_answer_text(payload.get("answerText", ""), prompt)
        return payload

    def _wait_for_answer(self, page, prompt: str, timeout_ms: int) -> GoogleAiResult:
        deadline = time.monotonic() + max(timeout_ms, 1_000) / 1000
        last_payload: dict | None = None
        while time.monotonic() < deadline:
            self._ensure_not_blocked(page, stage="waiting for the answer")
            payload = self._extract_payload(page, prompt)
            answer_text = payload.get("answerText", "").strip()
            if len(answer_text) >= 60:
                return GoogleAiResult(
                    answer_text=answer_text,
                    citations=filter_citations(payload.get("citations", [])),
                    final_url=payload.get("finalUrl", page.url),
                    page_title=payload.get("pageTitle", page.title()),
                    body_excerpt=payload.get("bodyExcerpt", ""),
                )
            last_payload = payload
            page.wait_for_timeout(1_000)

        last_excerpt = ""
        if last_payload:
            last_excerpt = last_payload.get("bodyExcerpt", "")
        raise GoogleAiTimeoutError(
            "Timed out waiting for Google AI answer. "
            "Last URL: "
            f"{page.url}. Last excerpt: "
            f"{shorten(last_excerpt, width=260, placeholder='...')}"
        )
