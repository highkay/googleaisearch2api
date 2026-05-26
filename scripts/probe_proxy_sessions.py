from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from googleaisearch2api.browser import GoogleAiBlockedError, GoogleAiRunner, resolve_browser_proxy
from googleaisearch2api.config import ServiceConfig, get_settings
from googleaisearch2api.db import create_db_engine, create_session_factory, create_tables, utc_now
from googleaisearch2api.egress import probe_egress
from googleaisearch2api.iplark import (
    IPAPI_IS_BASE_URL,
    IplarkProbeResult,
    parse_ipapi_is_payload,
    probe_ipapi_is_ip,
    probe_iplark_ip,
)
from googleaisearch2api.logging import configure_logging
from googleaisearch2api.proxy_sessions import (
    DEFAULT_STICKY_SUFFIX_TEMPLATE,
    STATUS_ACTIVE,
    STATUS_COOLDOWN,
    STATUS_RETIRED,
    ProxySessionSnapshot,
    ProxySessionStore,
    build_proxy_config_for_session,
    format_sticky_username,
    google_block_has_ip_mismatch,
    is_risk_metadata_retire_reason,
    parse_google_block_ips,
    resolve_proxy_base_username,
)
from googleaisearch2api.store import ConfigStore

DEFAULT_CANARY_PROMPT = "What is nineteen plus twenty-three? Reply with only the number."
DEFAULT_CANARY_EXPECTED_ANSWER = "42"
DEFAULT_CANARY_REPEATS = 2


def _snapshot_json(snapshot: ProxySessionSnapshot) -> dict:
    payload = asdict(snapshot)
    for key, value in list(payload.items()):
        if hasattr(value, "isoformat"):
            payload[key] = value.isoformat()
    return payload


def _datetime_for_compare(value: datetime, reference: datetime) -> datetime:
    if value.tzinfo is None and reference.tzinfo is not None:
        return value.replace(tzinfo=reference.tzinfo)
    if value.tzinfo is not None and reference.tzinfo is None:
        return value.replace(tzinfo=None)
    return value


def _skip_candidate_reason(
    snapshot: ProxySessionSnapshot,
    *,
    now: datetime,
    refresh_active: bool,
    retry_cooldown: bool,
    retry_retired: bool,
    only_risk_retired: bool,
) -> str | None:
    risk_retired = (
        snapshot.status == STATUS_RETIRED
        and is_risk_metadata_retire_reason(snapshot.retire_reason)
    )
    if only_risk_retired and not risk_retired:
        return "not retired by legacy risk metadata gate"
    if snapshot.status == STATUS_ACTIVE and not refresh_active:
        return "active session preserved"
    if snapshot.status == STATUS_COOLDOWN and not retry_cooldown:
        if snapshot.cooldown_until is None:
            return "session is in cooldown"
        cooldown_until = _datetime_for_compare(snapshot.cooldown_until, now)
        compare_now = _datetime_for_compare(now, snapshot.cooldown_until)
        if cooldown_until > compare_now:
            return "session is in cooldown"
    if snapshot.status == STATUS_RETIRED and not risk_retired and not retry_retired:
        return "retired session skipped"
    return None


def _candidate_indices(start: int, end: int, *, shuffle: bool, seed: int | None) -> list[int]:
    indices = list(range(start, end + 1))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return indices


def _normalize_canary_answer(text: str) -> str:
    normalized = " ".join(text.strip().split()).strip("`'\" ")
    while len(normalized) > 1 and normalized.endswith("."):
        normalized = normalized[:-1].strip().strip("`'\" ")
    return normalized.casefold()


def _canary_answer_matches(actual: str, expected: str) -> bool:
    expected = expected.strip()
    if not expected:
        return True
    return _normalize_canary_answer(actual) == _normalize_canary_answer(expected)


def _run_canary(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    config: ServiceConfig,
    prompt: str,
    expected_answer: str,
    repeats: int,
) -> ProxySessionSnapshot:
    expected = expected_answer.strip()
    attempts = max(repeats, 1)
    runner = GoogleAiRunner()
    try:
        for attempt in range(1, attempts + 1):
            try:
                result = runner.run_prompt(config, prompt)
            except GoogleAiBlockedError as exc:
                block_ips = parse_google_block_ips(str(exc))
                store.record_event(
                    proxy_session_id=snapshot.id,
                    event_type="google_canary_blocked",
                    message=str(exc),
                    raw_json={
                        "attempt": attempt,
                        "attempts": attempts,
                        "block_ips": block_ips,
                        "ip_mismatch": google_block_has_ip_mismatch(block_ips),
                    },
                )
                return store.mark_canary_blocked(
                    snapshot.id,
                    error_message=str(exc),
                    block_ips=block_ips,
                )
            except Exception as exc:
                return store.mark_session_cooldown(
                    snapshot.id,
                    reason=f"google canary failed on attempt {attempt}/{attempts}: {exc!r}",
                )

            actual = result.answer_text.strip()
            if not _canary_answer_matches(actual, expected):
                store.record_event(
                    proxy_session_id=snapshot.id,
                    event_type="google_canary_unexpected_answer",
                    message=f"expected={expected!r} actual={actual!r}",
                    raw_json={
                        "attempt": attempt,
                        "attempts": attempts,
                        "expected": expected,
                        "actual": actual,
                        "final_url": result.final_url,
                        "page_title": result.page_title,
                        "body_excerpt": result.body_excerpt,
                    },
                )
                return store.mark_session_cooldown(
                    snapshot.id,
                    reason=(
                        "google canary returned unexpected answer: "
                        f"expected={expected!r} actual={actual!r}"
                    ),
                )

        store.record_event(
            proxy_session_id=snapshot.id,
            event_type="google_canary_success",
            message=f"passed {attempts} canary attempt(s)",
            raw_json={
                "attempts": attempts,
                "expected": expected,
            },
        )
        return store.mark_canary_success(snapshot.id)
    finally:
        runner.close()


def _run_iplark(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    direct_config: ServiceConfig,
    *,
    min_quality_score: int,
    risk_source: str,
) -> tuple[ProxySessionSnapshot, IplarkProbeResult | None]:
    if not snapshot.primary_ip:
        snapshot = store.mark_session_cooldown(
            snapshot.id,
            reason="egress probe did not find an IP",
        )
        return snapshot, None
    try:
        if risk_source == "ipapi":
            result = probe_ipapi_is_ip(snapshot.primary_ip)
        else:
            result = probe_iplark_ip(snapshot.primary_ip, direct_config)
    except Exception as exc:
        store.record_event(
            proxy_session_id=snapshot.id,
            event_type="iplark_error",
            message=repr(exc),
        )
        return snapshot, None
    snapshot = store.update_iplark_result(
        proxy_session_id=snapshot.id,
        quality_score=result.quality_score,
        usage_type=result.usage_type,
        category=result.category,
        public_proxy=result.public_proxy,
        threat=result.threat,
        tag=result.tag,
        min_quality_score=min_quality_score,
    )
    store.record_event(
        proxy_session_id=snapshot.id,
        event_type="iplark",
        message=f"source={result.source} score={result.quality_score}",
        raw_json={
            "source": result.source,
            "score": result.score_json,
            "intelligence": result.intelligence_json,
        },
    )
    return snapshot, result


def _proxy_url_for_urllib(config: ServiceConfig) -> str:
    browser_proxy = resolve_browser_proxy(config)
    if not browser_proxy:
        raise ValueError("proxy is not configured")
    server = str(browser_proxy["server"])
    if "://" not in server:
        server = f"http://{server}"
    scheme, rest = server.split("://", 1)
    if scheme.lower().startswith("socks"):
        raise ValueError("fast IPAPI egress only supports HTTP proxies")

    username = browser_proxy.get("username")
    if not username:
        return server
    password = browser_proxy.get("password") or ""
    return f"{scheme}://{quote(username, safe='')}:{quote(password, safe='')}@{rest}"


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ipapi_asn_and_org(payload: dict) -> tuple[str | None, str | None]:
    asn_payload = payload.get("asn") if isinstance(payload.get("asn"), dict) else {}
    company_payload = payload.get("company") if isinstance(payload.get("company"), dict) else {}
    asn_value = _text_or_none(asn_payload.get("asn"))
    if asn_value and not asn_value.upper().startswith("AS"):
        asn_value = f"AS{asn_value}"
    organization = (
        _text_or_none(company_payload.get("name"))
        or _text_or_none(asn_payload.get("org"))
        or _text_or_none(asn_payload.get("name"))
    )
    return asn_value, organization


def _run_fast_ipapi_egress(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    config: ServiceConfig,
    *,
    min_quality_score: int,
    timeout_s: int,
) -> tuple[ProxySessionSnapshot, IplarkProbeResult | None]:
    try:
        proxy_url = _proxy_url_for_urllib(config)
        opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
        request = Request(
            IPAPI_IS_BASE_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/124 Safari/537.36",
            },
        )
        with opener.open(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", "replace")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            payload = {}
        ip = payload.get("ip")
        if not isinstance(ip, str) or not ip.strip():
            raise ValueError("api.ipapi.is response did not include ip")
    except Exception as exc:
        store.record_event(
            proxy_session_id=snapshot.id,
            event_type="fast_ipapi_egress_error",
            message=repr(exc),
        )
        return (
            store.mark_session_cooldown(
                snapshot.id,
                reason=f"fast IPAPI egress failed: {exc!r}",
            ),
            None,
        )

    asn, organization = _ipapi_asn_and_org(payload)
    snapshot = store.update_egress(
        proxy_session_id=snapshot.id,
        ips=[ip],
        source="ipapi.is",
        asn=asn,
        organization=organization,
        raw_json={"ipapi": payload},
    )
    if snapshot.status in {STATUS_COOLDOWN, STATUS_RETIRED}:
        return snapshot, None

    result = parse_ipapi_is_payload(ip, payload)
    snapshot = store.update_iplark_result(
        proxy_session_id=snapshot.id,
        quality_score=result.quality_score,
        usage_type=result.usage_type,
        category=result.category,
        public_proxy=result.public_proxy,
        threat=result.threat,
        tag=result.tag,
        min_quality_score=min_quality_score,
    )
    store.record_event(
        proxy_session_id=snapshot.id,
        event_type="ipapi_egress_risk",
        message=f"score={result.quality_score}",
        raw_json={
            "source": result.source,
            "score": result.score_json,
            "intelligence": result.intelligence_json,
        },
    )
    return snapshot, result


def _run_egress(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    config: ServiceConfig,
    *,
    checks: int,
) -> ProxySessionSnapshot:
    observed_vectors: list[list[str]] = []
    raw_payloads = {}
    asn = None
    organization = None
    for round_index in range(1, checks + 1):
        try:
            result = probe_egress(config)
        except Exception as exc:
            store.record_event(
                proxy_session_id=snapshot.id,
                event_type="egress_error",
                message=repr(exc),
            )
            return store.mark_session_cooldown(
                snapshot.id,
                reason=f"egress probe failed: {exc!r}",
            )
        observed_vectors.append(result.ips)
        raw_payloads[f"round_{round_index}"] = result.raw
        asn = result.asn or asn
        organization = result.organization or organization

    if not observed_vectors or not observed_vectors[0]:
        return store.mark_session_cooldown(snapshot.id, reason="egress probe did not find an IP")

    first_vector = observed_vectors[0]
    unstable = any(vector != first_vector for vector in observed_vectors[1:])
    snapshot = store.update_egress(
        proxy_session_id=snapshot.id,
        ips=first_vector,
        source="egress",
        asn=asn,
        organization=organization,
        raw_json=raw_payloads,
    )
    if unstable:
        return store.mark_session_cooldown(
            snapshot.id,
            reason=f"egress probe returned unstable IP vectors: {observed_vectors}",
        )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or refresh the Resin sticky proxy session allowlist."
    )
    parser.add_argument("--base-username", default="", help="Proxy username prefix, e.g. openai.")
    parser.add_argument("--start", type=int, default=1, help="First sticky session index.")
    parser.add_argument("--end", type=int, default=10, help="Last sticky session index.")
    parser.add_argument(
        "--suffix-template",
        default=DEFAULT_STICKY_SUFFIX_TEMPLATE,
        help="Suffix template. Default creates <base>.user{n}.",
    )
    parser.add_argument("--egress-checks", type=int, default=2, help="Egress checks per session.")
    parser.add_argument(
        "--min-quality-score",
        type=int,
        default=70,
        help="Deprecated compatibility option; third-party risk scores are recorded only.",
    )
    parser.add_argument(
        "--risk-source",
        choices=["auto", "ipapi"],
        default="auto",
        help="IP risk metadata source. auto uses IPLark; ipapi skips IPLark browser probing.",
    )
    parser.add_argument("--skip-egress", action="store_true", help="Skip egress probing.")
    parser.add_argument("--skip-iplark", action="store_true", help="Skip risk metadata probing.")
    parser.add_argument(
        "--fast-ipapi-egress",
        action="store_true",
        help="Use one proxied api.ipapi.is request for both egress IP discovery and risk metadata.",
    )
    parser.add_argument(
        "--fast-egress-timeout",
        type=int,
        default=12,
        help="Timeout in seconds for --fast-ipapi-egress.",
    )
    parser.add_argument(
        "--refresh-active",
        action="store_true",
        help="Re-probe active sessions instead of preserving known-good sessions.",
    )
    parser.add_argument(
        "--retry-cooldown",
        action="store_true",
        help="Re-probe sessions before their cooldown expires.",
    )
    parser.add_argument(
        "--retry-retired",
        action="store_true",
        help="Re-probe all retired sessions, including duplicate exits.",
    )
    parser.add_argument(
        "--only-risk-retired",
        action="store_true",
        help="Only probe sessions retired by the legacy third-party risk metadata gate.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Probe candidate indexes in random order to sample a wider exit pool.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for --shuffle.")
    parser.add_argument(
        "--stop-after-active",
        type=int,
        default=0,
        help="Stop once this many active sessions exist for the base username.",
    )
    parser.add_argument("--skip-google-canary", action="store_true", help="Skip Google canary.")
    parser.add_argument(
        "--canary-prompt",
        default=DEFAULT_CANARY_PROMPT,
        help="Google canary prompt. Default asks for a deterministic short answer.",
    )
    parser.add_argument(
        "--canary-expected-answer",
        default=DEFAULT_CANARY_EXPECTED_ANSWER,
        help=(
            "Only mark canary successful if the normalized answer matches this value. "
            "Set to an empty string to accept any non-blocked answer."
        ),
    )
    parser.add_argument(
        "--canary-repeats",
        type=int,
        default=DEFAULT_CANARY_REPEATS,
        help="Require this many consecutive successful Google canary answers before activation.",
    )
    args = parser.parse_args()

    if args.end < args.start:
        raise SystemExit("--end must be >= --start")
    if args.canary_repeats < 1:
        raise SystemExit("--canary-repeats must be >= 1")

    settings = get_settings()
    configure_logging(settings.app_log_level)
    settings.ensure_directories()
    engine = create_db_engine(str(settings.db_path))
    create_tables(engine)
    session_factory = create_session_factory(engine)
    store = ProxySessionStore(session_factory)
    config_store = ConfigStore(
        session_factory,
        ServiceConfig.from_settings(settings),
        request_log_max_rows=settings.request_log_max_rows,
    )

    base_config = config_store.get_config()
    base_username = args.base_username.strip() or resolve_proxy_base_username(base_config)
    direct_config = base_config.model_copy(
        update={
            "browser_proxy_server": None,
            "browser_proxy_username": None,
            "browser_proxy_password": None,
            "browser_proxy_bypass": None,
        },
        deep=True,
    )

    records = []
    for index in _candidate_indices(args.start, args.end, shuffle=args.shuffle, seed=args.seed):
        if (
            args.stop_after_active > 0
            and store.count_active_sessions(base_username) >= args.stop_after_active
        ):
            break
        proxy_username = format_sticky_username(base_username, index, args.suffix_template)
        session_name = f"user{index}"
        snapshot = store.upsert_proxy_session(
            proxy_base_username=base_username,
            session_name=session_name,
            proxy_username=proxy_username,
        )
        skip_reason = _skip_candidate_reason(
            snapshot,
            now=utc_now(),
            refresh_active=args.refresh_active,
            retry_cooldown=args.retry_cooldown,
            retry_retired=args.retry_retired,
            only_risk_retired=args.only_risk_retired,
        )
        if skip_reason:
            records.append(
                {
                    "session": _snapshot_json(snapshot),
                    "iplark": None,
                    "skipped": skip_reason,
                }
            )
            continue
        candidate_config = build_proxy_config_for_session(base_config, proxy_username)

        iplark_result = None
        if args.fast_ipapi_egress:
            snapshot, iplark_result = _run_fast_ipapi_egress(
                store,
                snapshot,
                candidate_config,
                min_quality_score=args.min_quality_score,
                timeout_s=args.fast_egress_timeout,
            )
        elif not args.skip_egress:
            snapshot = _run_egress(
                store,
                snapshot,
                candidate_config,
                checks=max(args.egress_checks, 1),
            )
        if snapshot.status in {STATUS_COOLDOWN, STATUS_RETIRED}:
            records.append(
                {
                    "session": _snapshot_json(snapshot),
                    "iplark": asdict(iplark_result) if iplark_result else None,
                }
            )
            continue

        if not args.skip_iplark and not args.fast_ipapi_egress:
            snapshot, iplark_result = _run_iplark(
                store,
                snapshot,
                direct_config,
                min_quality_score=args.min_quality_score,
                risk_source=args.risk_source,
            )
        if snapshot.status in {STATUS_COOLDOWN, STATUS_RETIRED}:
            records.append(
                {
                    "session": _snapshot_json(snapshot),
                    "iplark": asdict(iplark_result) if iplark_result else None,
                }
            )
            continue

        if not args.skip_google_canary:
            snapshot = _run_canary(
                store,
                snapshot,
                candidate_config,
                args.canary_prompt,
                args.canary_expected_answer,
                args.canary_repeats,
            )

        records.append(
            {
                "session": _snapshot_json(snapshot),
                "iplark": asdict(iplark_result) if iplark_result else None,
            }
        )

    print(
        json.dumps(
            {
                "base_username": base_username,
                "active_sessions": store.count_active_sessions(base_username),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
