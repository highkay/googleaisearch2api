from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from patchright.sync_api import Error as PatchrightError
from patchright.sync_api import TimeoutError as PatchrightTimeoutError
from patchright.sync_api import sync_playwright

from .browser import DEFAULT_BROWSER_CHANNEL, resolve_browser_user_agent
from .config import ServiceConfig

IPLARK_BASE_URL = "https://iplark.com"
IPAPI_IS_BASE_URL = "https://api.ipapi.is"
QUALITY_SCORE_KEYS = {"quality_score", "qualityScore", "score", "risk_score"}
RISK_LABELS = {
    "very low": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "very high": 4,
    "critical": 5,
}


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
    source: str = "iplark"


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
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "risk",
            "risky",
            "是",
            "有",
        }
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


def _has_quality_score(payload: dict[str, Any]) -> bool:
    return _to_int(_find_key(payload, QUALITY_SCORE_KEYS)) is not None


def _clamp_quality_score(value: int) -> int:
    return max(0, min(100, value))


def _parse_abuser_score(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None

    score = None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        try:
            score = float(match.group(1))
        except ValueError:
            score = None

    label = None
    label_match = re.search(r"\(([^)]+)\)", text)
    if label_match:
        candidate = label_match.group(1).strip().lower()
        label = candidate or None
    return score, label


def _highest_risk_label(labels: list[str | None]) -> str | None:
    best = None
    best_rank = -1
    for label in labels:
        if not label:
            continue
        rank = RISK_LABELS.get(label.lower(), -1)
        if rank > best_rank:
            best = label
            best_rank = rank
    return best


def _payloads_from_ipapi_payload(
    ip: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    company = payload.get("company") if isinstance(payload.get("company"), dict) else {}
    asn = payload.get("asn") if isinstance(payload.get("asn"), dict) else {}

    risk_values = [
        _parse_abuser_score(company.get("abuser_score")),
        _parse_abuser_score(asn.get("abuser_score")),
        _parse_abuser_score(payload.get("abuser_score")),
    ]
    numeric_risks = [score for score, _ in risk_values if score is not None]
    risk_score = max(numeric_risks) if numeric_risks else None
    risk_label = _highest_risk_label([label for _, label in risk_values])

    is_datacenter = _to_bool(payload.get("is_datacenter"))
    is_proxy = _to_bool(payload.get("is_proxy"))
    is_vpn = _to_bool(payload.get("is_vpn"))
    is_tor = _to_bool(payload.get("is_tor"))
    is_abuser = _to_bool(payload.get("is_abuser"))

    public_proxy = is_proxy or is_tor
    high_label = RISK_LABELS.get((risk_label or "").lower(), -1) >= RISK_LABELS["high"]
    threat = is_tor or high_label or (risk_score is not None and risk_score >= 0.5)

    quality_score = 100
    if is_datacenter:
        quality_score -= 25
    if is_vpn:
        quality_score -= 25
    if is_proxy:
        quality_score -= 40
    if is_tor:
        quality_score -= 60
    if is_abuser and risk_score is None:
        quality_score -= 20
    if risk_score is not None:
        quality_score -= min(60, round(risk_score * 100))
    if risk_label and RISK_LABELS.get(risk_label.lower(), 0) >= RISK_LABELS["medium"]:
        quality_score = min(quality_score, 70)
    if public_proxy:
        quality_score = min(quality_score, 60)
    if threat:
        quality_score = min(quality_score, 50)
    quality_score = _clamp_quality_score(quality_score)

    flags = []
    if is_datacenter:
        flags.append("datacenter")
    if is_vpn:
        flags.append("vpn")
    if is_proxy:
        flags.append("proxy")
    if is_tor:
        flags.append("tor")
    if is_abuser:
        flags.append("abuser")
    if risk_label:
        flags.append(f"risk:{risk_label}")

    usage_type = None
    if is_tor:
        usage_type = "tor"
    elif is_proxy:
        usage_type = "proxy"
    elif is_vpn:
        usage_type = "vpn"
    elif is_datacenter:
        usage_type = "datacenter"
    elif _to_bool(payload.get("is_mobile")):
        usage_type = "mobile"

    category = _to_text(company.get("type")) or ("datacenter" if is_datacenter else None)
    intelligence = {
        "usageType": usage_type,
        "category": category,
        "publicProxy": public_proxy,
        "threat": threat,
        "tag": ", ".join(flags) if flags else None,
        "riskScore": risk_score,
        "riskLabel": risk_label,
        "source": "ipapi.is",
    }
    intelligence = {key: value for key, value in intelligence.items() if value is not None}

    score_json = {
        "provider": "ipapi.is",
        "ip": ip,
        "quality_score": quality_score,
        "risk_score": risk_score,
        "risk_label": risk_label,
    }
    intelligence_json = {
        "provider": "ipapi.is",
        "intelligence": intelligence,
        "raw": payload,
    }
    return score_json, intelligence_json


def _parse_quality_score_from_text(body_text: str) -> int | None:
    patterns = (
        r"(?:IP评分|IP\s*Score|IP\s*Quality\s*Score)\s+(\d{1,3})(?:\s+\1\s*/\s*100)?",
        r"(?:IP评分|IP\s*Score|IP\s*Quality\s*Score).*?(\d{1,3})\s*/\s*100",
    )
    for pattern in patterns:
        match = re.search(pattern, body_text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        score = _to_int(match.group(1))
        if score is not None and 0 <= score <= 100:
            return score
    return None


def _next_text_value(lines: list[str], labels: set[str]) -> str | None:
    normalized_labels = {label.strip().rstrip(":：").lower() for label in labels}
    for index, line in enumerate(lines):
        normalized_line = line.strip().rstrip(":：").lower()
        if normalized_line not in normalized_labels:
            continue
        for value in lines[index + 1 :]:
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _payloads_from_body_text(
    ip: str,
    body_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    score = _parse_quality_score_from_text(body_text)
    score_json = {"ip": ip, "quality_score": score} if score is not None else {}

    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    intelligence = {
        "usageType": _next_text_value(lines, {"使用类型", "Usage Type"}),
        "category": _next_text_value(lines, {"IP类型", "Category", "IP Type"}),
        "publicProxy": _next_text_value(lines, {"代理", "Proxy"}),
        "threat": _next_text_value(lines, {"威胁", "Threat"}),
        "tag": _next_text_value(lines, {"标签", "Tag", "Tags"}),
    }
    intelligence = {key: value for key, value in intelligence.items() if value is not None}
    intelligence_json = {"intelligence": intelligence} if intelligence else {}
    return score_json, intelligence_json


def _wait_for_payloads(page: Any, is_complete, timeout_ms: int) -> None:
    deadline = time.monotonic() + (timeout_ms / 1000)
    while not is_complete():
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            return
        page.wait_for_timeout(min(500, remaining_ms))


def parse_iplark_payloads(
    ip: str,
    *,
    score_json: dict[str, Any] | None,
    intelligence_json: dict[str, Any] | None,
    source: str = "iplark",
) -> IplarkProbeResult:
    score_json = score_json or {}
    intelligence_json = intelligence_json or {}
    return IplarkProbeResult(
        ip=ip,
        quality_score=_to_int(_find_key(score_json, QUALITY_SCORE_KEYS)),
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
        source=source,
    )


def parse_ipapi_is_payload(ip: str, payload: dict[str, Any]) -> IplarkProbeResult:
    score_json, intelligence_json = _payloads_from_ipapi_payload(ip, payload)
    return parse_iplark_payloads(
        ip,
        score_json=score_json,
        intelligence_json=intelligence_json,
        source="ipapi.is",
    )


def probe_ipapi_is_ip(ip: str, *, timeout_s: int = 20) -> IplarkProbeResult:
    url = f"{IPAPI_IS_BASE_URL}?{urlencode({'q': ip})}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/124 Safari/537.36",
        },
    )
    with urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", "replace")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        payload = {}
    return parse_ipapi_is_payload(ip, payload)


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
    settle_ms: int = 12_000,
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
            _wait_for_payloads(
                page,
                lambda: _has_quality_score(score_json) and bool(intelligence_json),
                settle_ms,
            )
            if not score_json or not intelligence_json:
                try:
                    body = page.locator("body").inner_text(timeout=5_000)
                except (PatchrightError, PatchrightTimeoutError):
                    body = ""
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {}
                if isinstance(payload, dict) and not score_json:
                    score_json = payload
                if body:
                    body_score_json, body_intelligence_json = _payloads_from_body_text(ip, body)
                    if not _has_quality_score(score_json):
                        score_json = body_score_json
                    if not intelligence_json:
                        intelligence_json = body_intelligence_json
        finally:
            context.close()
            browser.close()

    result = parse_iplark_payloads(
        ip,
        score_json=score_json,
        intelligence_json=intelligence_json,
    )
    if result.quality_score is not None:
        return result

    try:
        fallback = probe_ipapi_is_ip(ip)
    except Exception:
        return result
    if fallback.quality_score is not None:
        return fallback
    return result
