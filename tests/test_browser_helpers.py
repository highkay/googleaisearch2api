from googleaisearch2api.browser import (
    GoogleAiRunner,
    clean_answer_text,
    filter_citations,
    resolve_browser_proxy,
    resolve_browser_user_agent,
)
from googleaisearch2api.config import ServiceConfig


def test_clean_answer_text_removes_prompt_and_disclaimer() -> None:
    query = "What changed?"
    raw = "What changed?\nThe answer is here.\nAI's response may contain mistakes"
    assert clean_answer_text(raw, query) == "The answer is here."


def test_clean_answer_text_removes_you_said_prompt_echo() -> None:
    query = "What is 2+2? Reply with only the number."
    raw = f"You said:\n{query}\n4\nAI can make mistakes, so double-check responses"
    assert clean_answer_text(raw, query) == "4"


def test_clean_answer_text_rejects_exact_you_said_echo() -> None:
    query = "Return a JSON object with current search results."
    assert clean_answer_text(f"You said:\n{query}", query) == ""


class _FakePage:
    url = "https://www.google.com/search?udm=50"

    def __init__(self) -> None:
        self.waits: list[int] = []

    def title(self) -> str:
        return "Google Search"

    def wait_for_timeout(self, timeout_ms: int) -> None:
        self.waits.append(timeout_ms)


class _PayloadRunner(GoogleAiRunner):
    def __init__(self, payloads: list[dict]) -> None:
        super().__init__()
        self.payloads = payloads

    def _ensure_not_blocked(self, page, stage: str) -> None:
        return None

    def _extract_payload(self, page, prompt: str) -> dict:
        payload = self.payloads.pop(0)
        payload["answerText"] = clean_answer_text(payload.get("answerText", ""), prompt)
        return payload


def test_wait_for_answer_skips_prompt_echo_and_accepts_ready_short_answer() -> None:
    query = "This is a long prompt that used to trip the eighty character length gate."
    runner = _PayloadRunner(
        [
            {
                "answerText": f"You said:\n{query}",
                "answerReady": False,
                "citations": [],
                "finalUrl": "https://www.google.com/search?udm=50",
                "pageTitle": "Google Search",
                "bodyExcerpt": f"You said:\n{query}",
            },
            {
                "answerText": "READY\nAI can make mistakes, so double-check responses",
                "answerReady": True,
                "citations": [],
                "finalUrl": "https://www.google.com/search?udm=50",
                "pageTitle": "Google Search",
                "bodyExcerpt": f"You said:\n{query}\nREADY",
            },
        ]
    )

    result = runner._wait_for_answer(_FakePage(), query, 5_000)

    assert result.answer_text == "READY"


def test_filter_citations_deduplicates_empty_entries() -> None:
    citations = filter_citations(
        [
            {"title": "OpenAI", "url": "https://openai.com"},
            {"title": "", "url": "https://openai.com"},
            {"title": "Google", "url": ""},
        ]
    )

    assert len(citations) == 1
    assert citations[0].title == "OpenAI"


def test_filter_citations_removes_google_utility_links() -> None:
    citations = filter_citations(
        [
            {
                "title": "See my AI Mode history",
                "url": "https://myactivity.google.com/search-services/history/search?product=83",
            },
            {"title": "Google support", "url": "https://support.google.com/websearch"},
            {"title": "Report", "url": "https://example.com/report"},
        ]
    )

    assert [citation.url for citation in citations] == ["https://example.com/report"]


def test_resolve_browser_user_agent_normalizes_headless_chrome() -> None:
    config = ServiceConfig(browser_headless=True)
    user_agent = resolve_browser_user_agent(config)
    assert user_agent is not None
    assert "HeadlessChrome" not in user_agent
    assert "Chrome/147.0.0.0" in user_agent
    assert "Edg/" not in user_agent


def test_resolve_browser_user_agent_preserves_manual_override() -> None:
    config = ServiceConfig(
        browser_headless=True,
        browser_user_agent="Mozilla/5.0 Custom",
    )
    assert resolve_browser_user_agent(config) == "Mozilla/5.0 Custom"


def test_resolve_browser_user_agent_disabled_for_headful_without_override() -> None:
    config = ServiceConfig(browser_headless=False)
    assert resolve_browser_user_agent(config) is None


def test_resolve_browser_proxy_returns_none_without_proxy_server() -> None:
    assert resolve_browser_proxy(ServiceConfig()) is None


def test_resolve_browser_proxy_preserves_split_credentials() -> None:
    config = ServiceConfig(
        browser_proxy_server="http://192.168.1.18:2260",
        browser_proxy_username="US",
        browser_proxy_password="secret",
        browser_proxy_bypass="localhost,127.0.0.1",
    )

    assert resolve_browser_proxy(config) == {
        "server": "http://192.168.1.18:2260",
        "username": "US",
        "password": "secret",
        "bypass": "localhost,127.0.0.1",
    }


def test_resolve_browser_proxy_splits_embedded_credentials() -> None:
    config = ServiceConfig(
        browser_proxy_server="http://US:secret@192.168.1.18:2260",
    )

    assert resolve_browser_proxy(config) == {
        "server": "http://192.168.1.18:2260",
        "username": "US",
        "password": "secret",
        "bypass": None,
    }


def test_resolve_browser_proxy_explicit_credentials_override_embedded_credentials() -> None:
    config = ServiceConfig(
        browser_proxy_server="http://embedded:embedded-pass@192.168.1.18:2260",
        browser_proxy_username="explicit",
        browser_proxy_password="explicit-pass",
    )

    assert resolve_browser_proxy(config) == {
        "server": "http://192.168.1.18:2260",
        "username": "explicit",
        "password": "explicit-pass",
        "bypass": None,
    }


def test_resolve_browser_proxy_decodes_embedded_credentials() -> None:
    config = ServiceConfig(
        browser_proxy_server="http://user%40name:p%3Ass@proxy.example:8080",
    )

    assert resolve_browser_proxy(config) == {
        "server": "http://proxy.example:8080",
        "username": "user@name",
        "password": "p:ss",
        "bypass": None,
    }


def test_resolve_browser_proxy_splits_socks5h_embedded_credentials() -> None:
    config = ServiceConfig(
        browser_proxy_server="socks5h://openai:secret@192.168.1.18:2260",
    )

    assert resolve_browser_proxy(config) == {
        "server": "socks5h://192.168.1.18:2260",
        "username": "openai",
        "password": "secret",
        "bypass": None,
    }


def test_runner_wraps_authenticated_socks_proxy_with_local_http_bridge() -> None:
    runner = GoogleAiRunner()
    try:
        config = ServiceConfig(
            browser_proxy_server="socks5h://openai:secret@192.168.1.18:2260",
            browser_proxy_bypass="localhost,127.0.0.1",
        )

        proxy = runner._build_launch_kwargs(config)["proxy"]
    finally:
        runner.close()

    assert proxy["server"].startswith("http://127.0.0.1:")
    assert proxy["bypass"] == "localhost,127.0.0.1"
    assert "username" not in proxy
    assert "password" not in proxy
