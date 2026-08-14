"""Mint Google AI Mode folif tokens via the browser, then measure their TTL.

The folif protocol is UNVERIFIED internally (see AGENTS.md).  This probe is
the instrument that validates it in production: it mints tokens through the
real patchright + chrome flow, then replays GET /async/folif over curl_cffi
until the tokens go stale, reporting how many consecutive answers they
survived.

Expected failure mode on machines without live Google access: the browser
mint step fails, a clear error is printed to stderr, and the script exits 2.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, NoReturn

from googleaisearch2api.ai_mode_http import AiModeTokens, fetch_folif
from googleaisearch2api.browser import GoogleAiRunner, harvest_ai_mode_tokens
from googleaisearch2api.config import ServiceConfig, get_settings

EXIT_MINT_FAILED = 2
REPORT_NOTE = (
    "tokens minted by browser, replayed via curl_cffi folif; "
    "TTL is the number of consecutive answers before the first non-answer"
)


def _fail(message: str, code: int) -> NoReturn:
    print(f"[probe] {message}", file=sys.stderr)
    raise SystemExit(code)


def _mask(value: str | None) -> str | None:
    if not value:
        return None
    return f"{value[:8]}...({len(value)} chars)"


def _masked_mint(raw_tokens: dict[str, Any], cookies: dict[str, str]) -> dict[str, Any]:
    return {
        "tokens": {
            key: _mask(str(value)) if value else None
            for key, value in raw_tokens.items()
            if key != "location_search"
        },
        "cookies": {name: _mask(value) for name, value in cookies.items()},
    }


def _build_tokens(raw_tokens: dict[str, Any], cookies: dict[str, str]) -> AiModeTokens:
    kwargs = {}
    for field in dataclasses.fields(AiModeTokens):
        if field.name == "cookies":
            continue
        value = raw_tokens.get(field.name)
        kwargs[field.name] = str(value) if value else None
    return AiModeTokens(**kwargs, cookies=dict(cookies))


def _new_curl_session() -> Any:
    try:
        from curl_cffi import requests as curl_requests
    except ImportError as exc:
        _fail(f"curl_cffi is required for folif replay: {exc}", 1)
    return curl_requests.Session(impersonate="chrome131")


def _mint_tokens(
    runner: GoogleAiRunner, config: ServiceConfig
) -> tuple[dict[str, Any], AiModeTokens]:
    try:
        context = runner._ensure_context_locked(config)  # noqa: SLF001
        page = context.new_page()
        try:
            page.goto(
                config.browser_base_url,
                wait_until="domcontentloaded",
                timeout=config.browser_timeout_ms,
            )
            runner._ensure_not_blocked(page, stage="minting tokens")  # noqa: SLF001
            harvested = harvest_ai_mode_tokens(page, context)
            raw_tokens = dict(harvested.get("tokens") or {})
            cookies = dict(harvested.get("cookies") or {})
            masked = _masked_mint(raw_tokens, cookies)
            print(f"[probe] minted token summary: {json.dumps(masked, ensure_ascii=False)}")
            return masked, _build_tokens(raw_tokens, cookies)
        finally:
            try:
                page.close()
            except Exception:
                pass
    except Exception as exc:
        _fail(
            f"cannot mint tokens: {type(exc).__name__}: {exc} "
            "(this environment may have no live Google access or the proxy "
            "may be unreachable)",
            EXIT_MINT_FAILED,
        )


def _emit_report(
    minted: dict[str, Any],
    attempts: list[dict[str, Any]],
    stale_reason: str | None,
    out_path: str | None,
) -> None:
    ttl_queries = sum(1 for attempt in attempts if attempt["kind"] == "answer")
    ttl_seconds = sum(
        attempt["elapsed_s"] for attempt in attempts if attempt["kind"] == "answer"
    )
    summary = f"[probe] ttl: {ttl_queries} consecutive answer(s), {ttl_seconds:.2f}s"
    if stale_reason:
        summary += f"; first non-answer kind: {stale_reason}"
    print(summary)

    report = {
        "minted": minted,
        "attempts": attempts,
        "ttl_queries": ttl_queries,
        "ttl_seconds": round(ttl_seconds, 2),
        "stale_reason": stale_reason,
        "note": REPORT_NOTE,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if out_path:
        Path(out_path).write_text(f"{payload}\n", encoding="utf-8")
        print(f"[probe] report written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mint Google AI Mode folif tokens via the browser, then replay "
            "GET /async/folif over curl_cffi to measure token TTL."
        )
    )
    parser.add_argument(
        "--prompt", required=True, help="Prompt replayed against the folif endpoint."
    )
    parser.add_argument("--queries", type=int, default=5, help="Maximum number of folif attempts.")
    parser.add_argument(
        "--interval-s",
        type=float,
        default=1.0,
        help="Sleep in seconds between folif attempts.",
    )
    parser.add_argument("--out", default=None, help="Optional path to also write the JSON report.")
    args = parser.parse_args()

    if args.queries < 1:
        raise SystemExit("--queries must be at least 1")
    if args.interval_s < 0:
        raise SystemExit("--interval-s must be >= 0")

    config = ServiceConfig.from_settings(get_settings())

    attempts: list[dict[str, Any]] = []
    stale_reason: str | None = None
    minted: dict[str, Any] = {}
    session = None
    runner = GoogleAiRunner()
    try:
        minted, tokens = _mint_tokens(runner, config)
        if not tokens.is_complete():
            missing = ", ".join(
                name
                for name in ("xsrf_folif_token", "stkp")
                if not getattr(tokens, name)
            )
            _fail(f"harvested tokens are incomplete: missing required {missing}", EXIT_MINT_FAILED)

        session = _new_curl_session()
        for attempt_index in range(args.queries):
            started_at = time.monotonic()
            result = fetch_folif(tokens, args.prompt, config=config, session=session)
            elapsed_s = time.monotonic() - started_at
            record = {
                "attempt": attempt_index,
                "elapsed_s": round(elapsed_s, 2),
                "kind": result.kind,
                "status_code": result.status_code,
                "answer_len": len(result.answer_text),
            }
            attempts.append(record)
            print(
                f"[probe] attempt {attempt_index}: kind={result.kind} "
                f"status={result.status_code} elapsed={elapsed_s:.2f}s "
                f"answer_len={len(result.answer_text)}"
            )
            if result.kind != "answer":
                stale_reason = result.kind
                break
            if args.interval_s and attempt_index < args.queries - 1:
                time.sleep(args.interval_s)
    except KeyboardInterrupt:
        _emit_report(minted, attempts, stale_reason, args.out)
        raise SystemExit(130) from None
    finally:
        try:
            runner.close()
        except Exception:
            pass
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    _emit_report(minted, attempts, stale_reason, args.out)


if __name__ == "__main__":
    main()
