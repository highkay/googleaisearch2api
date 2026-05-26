from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from patchright.sync_api import Error as PatchrightError
from patchright.sync_api import TimeoutError as PatchrightTimeoutError
from patchright.sync_api import sync_playwright

from .browser import DEFAULT_BROWSER_CHANNEL, resolve_browser_user_agent
from .config import ServiceConfig

IPLARK_BASE_URL = "https://iplark.com"


@dataclass(frozen=True, slots=True)
class IplarkProbeResult:
    ip: str
    quality_score: int | None
    usage_type: str | None = None
    category: str | None = None
    public_proxy: bool = False
    threat: bool = False
    tag: str | None = None
    score_json: dict[str, Any] = field(default_factory=dict)
    intelligence_json: dict[str, Any] = field(default_factory=dict)


def _normalize_key(value: str) -> str:
    return value.replace("_", "").replace("-", "").lower()


def _find_key(data: Any, keys: set[str]) -> Any:
    normalized_keys = {_normalize_key(key) for key in keys}
    if isinstance(data, dict):
        for key, value in data.items():
            if _normalize_key(str(key)) in normalized_keys:
                return value
        for value in data.values():
            found = _find_key(value, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_key(item, keys)
            if found is not None:
                return found
    return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "risk", "risky"}
    return False


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts) if parts else None
    return str(value).strip() or None


def parse_iplark_payloads(
    ip: str,
    *,
    score_json: dict[str, Any] | None,
    intelligence_json: dict[str, Any] | None,
) -> IplarkProbeResult:
    score_json = score_json or {}
    intelligence_json = intelligence_json or {}
    return IplarkProbeResult(
        ip=ip,
        quality_score=_to_int(
            _find_key(score_json, {"quality_score", "qualityScore", "score", "risk_score"})
        ),
        usage_type=_to_text(
            _find_key(intelligence_json, {"usage_type", "usageType", "ip_type", "ipType"})
        ),
        category=_to_text(_find_key(intelligence_json, {"category", "risk_category"})),
        public_proxy=_to_bool(
            _find_key(intelligence_json, {"public_proxy", "publicProxy", "proxy"})
        ),
        threat=_to_bool(_find_key(intelligence_json, {"threat", "is_threat", "isThreat"})),
        tag=_to_text(_find_key(intelligence_json, {"tag", "tags"})),
        score_json=score_json,
        intelligence_json=intelligence_json,
    )


def _build_launch_kwargs(config: ServiceConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "headless": config.browser_headless,
        "channel": DEFAULT_BROWSER_CHANNEL,
    }
    browser_user_agent = resolve_browser_user_agent(config)
    if browser_user_agent:
        kwargs["args"] = [f"--user-agent={browser_user_agent}"]
    return kwargs


def probe_iplark_ip(
    ip: str,
    config: ServiceConfig,
    *,
    timeout_ms: int = 30_000,
    settle_ms: int = 3_000,
) -> IplarkProbeResult:
    score_json: dict[str, Any] = {}
    intelligence_json: dict[str, Any] = {}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_build_launch_kwargs(config))
        context = browser.new_context(locale=config.browser_locale)
        page = context.new_page()

        def capture_response(response) -> None:
            nonlocal score_json, intelligence_json
            url = response.url
            if "iplark.com" not in url:
                return
            if "/ipscore" not in url and "/ipintelligence" not in url:
                return
            try:
                payload = response.json()
            except (PatchrightError, ValueError):
                return
            if not isinstance(payload, dict):
                return
            if "/ipscore" in url:
                score_json = payload
            elif "/ipintelligence" in url:
                intelligence_json = payload

        page.on("response", capture_response)
        try:
            try:
                page.goto(
                    f"{IPLARK_BASE_URL}/{ip}",
                    wait_until="domcontentloaded",
                    timeout=timeout_ms,
                )
            except (PatchrightError, PatchrightTimeoutError):
                pass
            page.wait_for_timeout(settle_ms)
            if not score_json or not intelligence_json:
                try:
                    body = page.locator("body").inner_text(timeout=5_000)
                except PatchrightError:
                    body = ""
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {}
                if isinstance(payload, dict) and not score_json:
                    score_json = payload
        finally:
            context.close()
            browser.close()

    return parse_iplark_payloads(
        ip,
        score_json=score_json,
        intelligence_json=intelligence_json,
    )
