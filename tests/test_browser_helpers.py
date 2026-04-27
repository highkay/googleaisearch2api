from googleaisearch2api.browser import (
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
