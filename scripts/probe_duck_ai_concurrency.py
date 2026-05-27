from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patchright.sync_api import Error as PatchrightError  # noqa: E402
from patchright.sync_api import TimeoutError as PatchrightTimeoutError  # noqa: E402
from patchright.sync_api import sync_playwright  # noqa: E402

from googleaisearch2api.browser import (  # noqa: E402
    DEFAULT_BROWSER_CHANNEL,
    GoogleAiRunner,
    resolve_browser_user_agent,
)
from googleaisearch2api.config import ServiceConfig, get_settings  # noqa: E402

DUCK_AI_URL = "https://duck.ai/"
CHAT_ENDPOINT_MARKER = "/duckchat/v1/chat"
DUCK_API_MARKER = "/duckchat/v1/"


@dataclass(slots=True)
class NetworkEvent:
    method: str
    url: str
    status: int


@dataclass(slots=True)
class ProbeResult:
    worker: int
    ok: bool
    duration_ms: int
    chat_status: int | None = None
    title: str = ""
    final_url: str = ""
    answer_excerpt: str = ""
    body_excerpt: str = ""
    error_type: str = ""
    error_message: str = ""
    network_events: list[NetworkEvent] = field(default_factory=list)


def _extract_answer_excerpt(body_text: str, prompt: str, *, limit: int = 500) -> str:
    body_text = _normalize_text(body_text)
    prompt = _normalize_text(prompt)
    if not body_text:
        return ""

    if prompt and prompt in body_text:
        body_text = body_text.rsplit(prompt, 1)[1]

    lines = []
    skip_patterns = [
        re.compile(r"^GPT-[\w .-]+$", re.IGNORECASE),
        re.compile(
            r"^(Generating response|Tools|Fast|Private|工具|快速|聊天|Free|·)$",
            re.IGNORECASE,
        ),
        re.compile(r"^所有聊天记录均为"),
        re.compile(r"^All chats are private"),
        re.compile(r"^DuckDuckGo 已匿名处理"),
    ]
    for raw_line in body_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern.search(line) for pattern in skip_patterns):
            continue
        lines.append(line)

    return _normalize_text("\n".join(lines))[:limit]


def _normalize_text(value: str) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _run_one(worker: int, prompt: str, config: ServiceConfig) -> ProbeResult:
    started_at = time.perf_counter()
    network_events: list[NetworkEvent] = []
    chat_status: int | None = None
    browser = None
    context = None
    runner = GoogleAiRunner()

    try:
        with sync_playwright() as playwright:
            launch_kwargs = {
                "headless": config.browser_headless,
                "channel": DEFAULT_BROWSER_CHANNEL,
            }
            browser_user_agent = resolve_browser_user_agent(config)
            if browser_user_agent:
                launch_kwargs["args"] = [f"--user-agent={browser_user_agent}"]

            browser_proxy = runner._resolve_launch_proxy_locked(config)  # noqa: SLF001
            if browser_proxy:
                launch_kwargs["proxy"] = browser_proxy

            browser = playwright.chromium.launch(**launch_kwargs)
            context_kwargs = {"locale": config.browser_locale}
            if browser_user_agent:
                context_kwargs["user_agent"] = browser_user_agent
            context = browser.new_context(**context_kwargs)
            page = context.new_page()
            page.set_default_timeout(config.browser_timeout_ms)

            def record_response(response) -> None:
                nonlocal chat_status
                if DUCK_API_MARKER not in response.url:
                    return
                event = NetworkEvent(
                    method=response.request.method,
                    url=response.url,
                    status=response.status,
                )
                network_events.append(event)
                if CHAT_ENDPOINT_MARKER in response.url and response.request.method == "POST":
                    chat_status = response.status

            page.on("response", record_response)
            page.goto(DUCK_AI_URL, wait_until="domcontentloaded", timeout=config.browser_timeout_ms)
            textarea = page.locator("textarea").first
            textarea.wait_for(state="visible", timeout=20_000)
            textarea.fill(prompt)
            page.wait_for_function(
                "() => { const b = document.querySelector('button[type=\"submit\"]');"
                " return b && !b.disabled; }",
                timeout=10_000,
            )

            with page.expect_response(
                lambda response: CHAT_ENDPOINT_MARKER in response.url
                and response.request.method == "POST",
                timeout=config.answer_timeout_ms,
            ) as chat_response_info:
                page.locator("button[type='submit']").first.click()
            chat_response = chat_response_info.value
            chat_status = chat_response.status

            try:
                page.wait_for_function(
                    "() => {"
                    " const text = document.body ? document.body.innerText : '';"
                    " const generating = text.includes('Generating response')"
                    "   || text.includes('正在生成');"
                    " const stop = document.querySelector("
                    "   'button[aria-label=\"停止生成\"], button[aria-label=\"Stop generating\"]'"
                    " );"
                    " return !generating && (!stop || stop.disabled);"
                    "}",
                    timeout=config.answer_timeout_ms,
                )
            except PatchrightTimeoutError:
                pass

            body_text = _normalize_text(page.locator("body").inner_text(timeout=5_000))
            answer_excerpt = _extract_answer_excerpt(body_text, prompt)
            ok = chat_status == 200 and bool(answer_excerpt)
            return ProbeResult(
                worker=worker,
                ok=ok,
                duration_ms=int((time.perf_counter() - started_at) * 1000),
                chat_status=chat_status,
                title=page.title(),
                final_url=page.url,
                answer_excerpt=answer_excerpt,
                body_excerpt=body_text[:500],
                error_message="" if ok else "Chat returned no extractable answer text.",
                network_events=network_events,
            )
    except Exception as exc:
        error_message = str(exc)
        if isinstance(exc, PatchrightError):
            error_message = re.sub(r"\s+", " ", error_message).strip()
        return ProbeResult(
            worker=worker,
            ok=False,
            duration_ms=int((time.perf_counter() - started_at) * 1000),
            chat_status=chat_status,
            error_type=type(exc).__name__,
            error_message=error_message[:800],
            network_events=network_events,
        )
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        runner.close()


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a live concurrent Patchright probe against Duck.ai."
    )
    parser.add_argument(
        "--prompt",
        default="Return one short sentence explaining what DuckDuckGo is.",
        help="Prompt to submit to Duck.ai.",
    )
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent browser workers.")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds to run.")
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run with a visible browser instead of the configured headless mode.",
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")

    settings = get_settings()
    config = ServiceConfig.from_settings(settings)
    if args.show_browser:
        config.browser_headless = False

    all_results: list[ProbeResult] = []
    for round_index in range(args.rounds):
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    (round_index * args.concurrency) + worker_index + 1,
                    args.prompt,
                    config,
                )
                for worker_index in range(args.concurrency)
            ]
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                status = "ok" if result.ok else "fail"
                print(
                    f"[{status}] worker={result.worker} status={result.chat_status} "
                    f"duration_ms={result.duration_ms} "
                    f"error={result.error_type or result.error_message}"
                )

    successes = sum(1 for result in all_results if result.ok)
    summary = {
        "ok": successes == len(all_results),
        "successes": successes,
        "total": len(all_results),
        "success_rate": successes / len(all_results),
        "results": [
            {
                **asdict(result),
                "network_events": [asdict(event) for event in result.network_events],
            }
            for result in sorted(all_results, key=lambda item: item.worker)
        ],
    }
    _print_json(summary)


if __name__ == "__main__":
    main()
