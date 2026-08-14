"""Tests for the config-gated hybrid runner (browser token mint + folif HTTP fast path)."""

from __future__ import annotations

import pytest

from googleaisearch2api.ai_mode_http import AiModeTokens, FolifResult
from googleaisearch2api.browser import GoogleAiRunner
from googleaisearch2api.config import ServiceConfig
from googleaisearch2api.hybrid_runner import HybridGoogleAiRunner
from googleaisearch2api.schemas import GoogleAiResult

BROWSER_URL = "https://www.google.com/search?udm=50"
FOLIF_URL = "https://www.google.com/search/async/folif"


def _complete_tokens() -> AiModeTokens:
    return AiModeTokens(
        srtst="srtst-v",
        garc="garc-v",
        xsrf_folif_token="xsrf-v",
        ei="ei-v",
        stkp="stkp-v",
        sca_esv="sca-v",
        mstk="mstk-v",
        elrc="elrc-v",
        cookies={"NID": "nid-v"},
    )


class _CannedTokenRunner(HybridGoogleAiRunner):
    def __init__(self, tokens: AiModeTokens | None) -> None:
        super().__init__()
        self.tokens = tokens
        self.mint_calls = 0

    def _mint_tokens(self, config: ServiceConfig) -> AiModeTokens | None:
        self.mint_calls += 1
        return self.tokens


def _install_fake_folif(
    monkeypatch: pytest.MonkeyPatch, results: list[FolifResult]
) -> list[tuple[AiModeTokens, str, ServiceConfig]]:
    calls: list[tuple[AiModeTokens, str, ServiceConfig]] = []

    def fake_fetch(
        tokens: AiModeTokens,
        prompt: str,
        *,
        config: ServiceConfig,
        timeout_s: float = 15.0,
        session: object | None = None,
    ) -> FolifResult:
        calls.append((tokens, prompt, config))
        return results.pop(0)

    monkeypatch.setattr("googleaisearch2api.hybrid_runner.fetch_folif", fake_fetch)
    return calls


def _install_fake_browser(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    state: dict[str, int] = {"calls": 0}

    def fake_run(self: object, config: ServiceConfig, prompt: str) -> GoogleAiResult:
        state["calls"] += 1
        return GoogleAiResult(
            answer_text="browser answer",
            final_url=BROWSER_URL,
            page_title="Google Search",
        )

    monkeypatch.setattr(GoogleAiRunner, "run_prompt", fake_run)
    return state


def test_hybrid_uses_folif_answer_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    body = "<div data-mstk>Folif answer body.</div>"
    folif_calls = _install_fake_folif(
        monkeypatch,
        [FolifResult(kind="answer", status_code=200, body=body, final_url=FOLIF_URL)],
    )
    browser_state = _install_fake_browser(monkeypatch)
    runner = _CannedTokenRunner(_complete_tokens())
    config = ServiceConfig(ai_mode_http_enabled=True)

    result = runner.run_prompt(config, "What is 2+2?")

    assert result.answer_text == body
    assert result.citations == []
    assert result.final_url == FOLIF_URL
    assert result.page_title == ""
    assert result.body_excerpt == body
    assert len(folif_calls) == 1
    assert folif_calls[0][1] == "What is 2+2?"
    assert browser_state["calls"] == 0


def test_hybrid_falls_back_on_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    folif_calls = _install_fake_folif(
        monkeypatch,
        [
            FolifResult(
                kind="shell",
                status_code=200,
                body="enablejs shell",
                final_url=BROWSER_URL,
            )
        ],
    )
    browser_state = _install_fake_browser(monkeypatch)
    runner = _CannedTokenRunner(_complete_tokens())
    config = ServiceConfig(ai_mode_http_enabled=True)

    result = runner.run_prompt(config, "shell prompt")

    assert len(folif_calls) == 1
    assert browser_state["calls"] == 1
    assert result.answer_text == "browser answer"


def test_hybrid_falls_back_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    folif_calls = _install_fake_folif(
        monkeypatch,
        [FolifResult(kind="empty", status_code=200, body="", final_url=FOLIF_URL)],
    )
    browser_state = _install_fake_browser(monkeypatch)
    runner = _CannedTokenRunner(_complete_tokens())
    config = ServiceConfig(ai_mode_http_enabled=True)

    result = runner.run_prompt(config, "empty prompt")

    assert len(folif_calls) == 1
    assert browser_state["calls"] == 1
    assert result.answer_text == "browser answer"


def test_hybrid_falls_back_when_tokens_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    folif_calls = _install_fake_folif(monkeypatch, [])
    browser_state = _install_fake_browser(monkeypatch)
    runner = _CannedTokenRunner(AiModeTokens(xsrf_folif_token=None))
    config = ServiceConfig(ai_mode_http_enabled=True)

    result = runner.run_prompt(config, "incomplete tokens")

    assert folif_calls == []
    assert browser_state["calls"] == 1
    assert runner.mint_calls == 1
    assert result.answer_text == "browser answer"


def test_hybrid_falls_back_when_token_mint_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    folif_calls = _install_fake_folif(monkeypatch, [])
    browser_state = _install_fake_browser(monkeypatch)
    runner = _CannedTokenRunner(None)
    config = ServiceConfig(ai_mode_http_enabled=True)

    result = runner.run_prompt(config, "mint failed")

    assert folif_calls == []
    assert browser_state["calls"] == 1
    assert result.answer_text == "browser answer"


def test_hybrid_uses_browser_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    folif_calls = _install_fake_folif(monkeypatch, [])
    browser_state = _install_fake_browser(monkeypatch)
    runner = _CannedTokenRunner(_complete_tokens())
    config = ServiceConfig(ai_mode_http_enabled=False)

    result = runner.run_prompt(config, "disabled path")

    assert browser_state["calls"] == 1
    assert runner.mint_calls == 0
    assert folif_calls == []
    assert result.answer_text == "browser answer"


class _FakeMintPage:
    def __init__(self) -> None:
        self.closed = False
        self.goto_args: list[tuple[str, str, int]] = []

    def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        self.goto_args.append((url, wait_until, timeout))

    def close(self) -> None:
        self.closed = True


class _FakeMintContext:
    def __init__(self) -> None:
        self.page = _FakeMintPage()

    def new_page(self) -> _FakeMintPage:
        return self.page


class _MintHarnessRunner(HybridGoogleAiRunner):
    def __init__(self) -> None:
        super().__init__()
        self.context = _FakeMintContext()
        self.blocked_checks = 0

    def _ensure_context_locked(self, config: ServiceConfig) -> _FakeMintContext:
        return self.context

    def _ensure_not_blocked(self, page: object, stage: str) -> None:
        self.blocked_checks += 1


def test_mint_tokens_maps_harvest_and_ignores_location_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harvested = {
        "tokens": {
            "srtst": "srtst-v",
            "garc": "garc-v",
            "xsrf_folif_token": "xsrf-v",
            "ei": "ei-v",
            "stkp": "stkp-v",
            "sca_esv": "sca-v",
            "mstk": "mstk-v",
            "elrc": "elrc-v",
            "location_search": "?udm=50",
        },
        "cookies": {"NID": "nid-v", "SID": "sid-v"},
    }
    monkeypatch.setattr(
        "googleaisearch2api.hybrid_runner.harvest_ai_mode_tokens",
        lambda page, context: harvested,
    )
    runner = _MintHarnessRunner()
    config = ServiceConfig(browser_base_url=BROWSER_URL)

    tokens = runner._mint_tokens(config)

    assert tokens is not None
    assert tokens.xsrf_folif_token == "xsrf-v"
    assert tokens.stkp == "stkp-v"
    assert tokens.elrc == "elrc-v"
    assert tokens.mstk == "mstk-v"
    assert tokens.cookies == {"NID": "nid-v", "SID": "sid-v"}
    assert tokens.is_complete()
    assert runner.context.page.goto_args == [(BROWSER_URL, "domcontentloaded", 90_000)]
    assert runner.blocked_checks == 1
    assert runner.context.page.closed


class _ExplodingContextRunner(HybridGoogleAiRunner):
    def _ensure_context_locked(self, config: ServiceConfig):
        raise RuntimeError("browser boom")


def test_mint_tokens_returns_none_on_exception() -> None:
    runner = _ExplodingContextRunner()

    assert runner._mint_tokens(ServiceConfig()) is None


def test_result_from_folif_uses_extracted_answer() -> None:
    runner = HybridGoogleAiRunner()
    result = runner._result_from_folif(
        FolifResult(
            kind="answer",
            status_code=200,
            body="  <div data-mstk>Folif body</div>  ",
            final_url=FOLIF_URL,
        )
    )

    assert result.answer_text == "<div data-mstk>Folif body</div>"
    assert result.citations == []
    assert result.final_url == FOLIF_URL
    assert result.page_title == ""
    assert result.body_excerpt == "<div data-mstk>Folif body</div>"


def test_result_from_folif_falls_back_to_raw_body() -> None:
    runner = HybridGoogleAiRunner()
    result = runner._result_from_folif(
        FolifResult(
            kind="answer",
            status_code=200,
            body="  no marker body  ",
            final_url=FOLIF_URL,
        )
    )

    assert result.answer_text == "no marker body"
