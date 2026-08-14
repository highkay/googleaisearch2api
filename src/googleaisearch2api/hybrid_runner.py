"""Config-gated hybrid runner: browser mints AI Mode folif tokens, HTTP fetches the answer.

The browser path remains the verified stable entry (see AGENTS.md); the folif
HTTP path is an OFF-by-default fast path gated by
``ServiceConfig.ai_mode_http_enabled``.  Token minting reuses all of
``GoogleAiRunner``'s browser machinery, and every non-answer folif response
falls back to the full browser flow.  folif answer/citation parsing is deferred
to TTL probe calibration, so ``_result_from_folif`` is intentionally minimal.
"""

from __future__ import annotations

import dataclasses

from loguru import logger

from .ai_mode_http import AiModeTokens, FolifResult, extract_answer_from_folif, fetch_folif
from .browser import GoogleAiRunner, harvest_ai_mode_tokens
from .config import ServiceConfig
from .schemas import GoogleAiResult


class HybridGoogleAiRunner(GoogleAiRunner):
    def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
        if not config.ai_mode_http_enabled:
            return super().run_prompt(config, prompt)

        tokens = self._mint_tokens(config)
        if tokens is None or not tokens.is_complete():
            logger.warning("hybrid: token mint incomplete; using browser path")
            return super().run_prompt(config, prompt)

        result = fetch_folif(tokens, prompt, config=config)
        if result.kind == "answer":
            return self._result_from_folif(result)

        logger.warning(
            "hybrid: folif fast path returned kind={}; using browser path",
            result.kind,
        )
        return super().run_prompt(config, prompt)

    def _mint_tokens(self, config: ServiceConfig) -> AiModeTokens | None:
        page = None
        try:
            context = self._ensure_context_locked(config)
            page = context.new_page()
            page.goto(
                config.browser_base_url,
                wait_until="domcontentloaded",
                timeout=config.browser_timeout_ms,
            )
            self._ensure_not_blocked(page, stage="minting AI Mode tokens")
            harvested = harvest_ai_mode_tokens(page, context)

            harvested_tokens = harvested.get("tokens")
            token_values: dict[str, str | None] = {}
            if isinstance(harvested_tokens, dict):
                for field in dataclasses.fields(AiModeTokens):
                    if field.name == "cookies":
                        continue
                    value = harvested_tokens.get(field.name)
                    token_values[field.name] = str(value) if value else None

            raw_cookies = harvested.get("cookies")
            cookies: dict[str, str] = {}
            if isinstance(raw_cookies, dict):
                cookies = {str(key): str(value) for key, value in raw_cookies.items()}

            return AiModeTokens(**token_values, cookies=cookies)
        except Exception as exc:
            logger.warning("hybrid: token mint failed: {}: {}", type(exc).__name__, exc)
            return None
        finally:
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

    def _result_from_folif(self, result: FolifResult) -> GoogleAiResult:
        answer = extract_answer_from_folif(result.body) or result.body.strip()
        return GoogleAiResult(
            answer_text=answer,
            citations=[],
            # Strip the folif query string: it carries the prompt and minted
            # session tokens that must not leak into request logs or responses.
            final_url=result.final_url.split("?", 1)[0],
            page_title="",
            body_excerpt=answer[:800],
        )
