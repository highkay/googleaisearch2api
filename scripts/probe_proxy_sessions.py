from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
from itertools import groupby
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from googleaisearch2api.browser import GoogleAiBlockedError, GoogleAiRunner, resolve_browser_proxy
from googleaisearch2api.config import ServiceConfig, get_settings
from googleaisearch2api.db import create_db_engine, create_session_factory, create_tables, utc_now
from googleaisearch2api.duck_ai import DuckAiRateLimitedError, DuckAiRunner
from googleaisearch2api.egress import probe_egress
from googleaisearch2api.fast_proxy_probe import probe_proxy_http_fast
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
    STATUS_EGRESS_CHECKED,
    STATUS_NEW,
    STATUS_RETIRED,
    STATUS_RISK_CHECKED,
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
DEFAULT_DUCK_CANARY_REPEATS = 1
EXISTING_SESSION_RANKING_FETCH_LIMIT = 5_000


def _session_success_score(snapshot: ProxySessionSnapshot) -> int:
    return int(snapshot.request_success_count) + int(snapshot.canary_success_count)


def _session_failure_score(snapshot: ProxySessionSnapshot) -> int:
    return (
        int(snapshot.request_block_count)
        + int(snapshot.canary_block_count)
        + int(snapshot.request_error_count)
    )


def _browser_canary_rank(snapshot: ProxySessionSnapshot) -> tuple[int, int, int, int, int]:
    """Rank L0 survivors for expensive browser canaries.

    Prefer never-blocked egress IPs, then lower historical failure, then higher
    success history. Sessions without a known primary IP sort last (need egress).
    """
    google_canary = (snapshot.google_canary_status or "").strip().lower()
    if google_canary == "ok":
        canary_rank = 0
    elif google_canary in {"", "unknown"}:
        canary_rank = 1
    elif google_canary == "blocked":
        canary_rank = 3
    else:
        canary_rank = 2
    has_ip_rank = 0 if snapshot.primary_ip else 1
    return (
        has_ip_rank,
        canary_rank,
        _session_failure_score(snapshot),
        -_session_success_score(snapshot),
        snapshot.id,
    )


def _dedupe_browser_candidates_by_ip(
    passed: list[tuple[ProxySessionSnapshot, dict[str, object] | None]],
) -> tuple[
    list[tuple[ProxySessionSnapshot, dict[str, object] | None]],
    list[tuple[ProxySessionSnapshot, dict[str, object] | None, str]],
]:
    """Keep one best session per primary_ip before browser canary.

    Returns (selected, skipped_with_reason). Sessions without primary_ip are kept
    (they still need egress discovery) but only once each by session id.
    """
    selected: list[tuple[ProxySessionSnapshot, dict[str, object] | None]] = []
    skipped: list[tuple[ProxySessionSnapshot, dict[str, object] | None, str]] = []
    seen_ips: set[str] = set()
    ordered = sorted(passed, key=lambda item: _browser_canary_rank(item[0]))
    for snapshot, payload in ordered:
        ip = (snapshot.primary_ip or "").strip()
        if not ip:
            selected.append((snapshot, payload))
            continue
        if ip in seen_ips:
            skipped.append(
                (
                    snapshot,
                    payload,
                    f"duplicate egress IP {ip}; browser canary budget uses one session per IP",
                )
            )
            continue
        # Soft-skip: same IP already proven blocked on this session — do not re-burn budget.
        if (snapshot.google_canary_status or "").strip().lower() == "blocked":
            skipped.append(
                (
                    snapshot,
                    payload,
                    f"primary IP {ip} already google_canary_status=blocked on this session",
                )
            )
            continue
        seen_ips.add(ip)
        selected.append((snapshot, payload))
    return selected, skipped


def _cooldown_is_ready(snapshot: ProxySessionSnapshot, now: datetime) -> bool:
    if snapshot.cooldown_until is None:
        return True
    cooldown_until = _datetime_for_compare(snapshot.cooldown_until, now)
    compare_now = _datetime_for_compare(now, cooldown_until)
    return cooldown_until <= compare_now


def _existing_session_recovery_group(
    snapshot: ProxySessionSnapshot,
    *,
    now: datetime | None = None,
) -> tuple[int, int, int, int]:
    """Rank existing sessions for recovery probes.

    Real production evidence (2026-07-16): recovery always burned max_probes=3 on the
    same high-history sessions that were still in cooldown / recently Google-blocked
    (user169/347/5700), while ready selectable sessions with prior canary ok
    (user122/user2055) were never probed. Prefer currently ready candidates first,
    then lower failure scores, then higher success history.
    """
    compare_now = now or utc_now()
    success_score = _session_success_score(snapshot)
    failure_score = _session_failure_score(snapshot)
    duplicate_rank = 1 if snapshot.duplicate_of_session_id is not None else 0
    google_canary = (snapshot.google_canary_status or "").strip().lower()
    if snapshot.status == STATUS_ACTIVE:
        status_rank = 0
    elif snapshot.status not in {STATUS_COOLDOWN, STATUS_RETIRED} and success_score > 0:
        status_rank = 1
    elif snapshot.status in {STATUS_RISK_CHECKED, STATUS_EGRESS_CHECKED, STATUS_NEW}:
        status_rank = 2
    elif snapshot.status == STATUS_COOLDOWN and success_score > 0:
        if _cooldown_is_ready(snapshot, compare_now):
            # Ready again: prefer sessions not last marked blocked by canary.
            status_rank = 3 if google_canary != "blocked" else 4
        else:
            # Still cooling — only try after ready candidates are exhausted.
            status_rank = 6
    elif snapshot.status == STATUS_COOLDOWN:
        status_rank = 5 if _cooldown_is_ready(snapshot, compare_now) else 7
    elif snapshot.status == STATUS_RETIRED and is_risk_metadata_retire_reason(
        snapshot.retire_reason
    ):
        status_rank = 8
    else:
        status_rank = 9
    # Prefer fewer failures before raw historical success so chronically blocked
    # high-success sessions stop monopolizing the probe budget.
    return (duplicate_rank, status_rank, failure_score, -success_score)


def _existing_session_recovery_rank(
    snapshot: ProxySessionSnapshot,
    *,
    now: datetime | None = None,
) -> tuple[int, int, int, int, int]:
    return (*_existing_session_recovery_group(snapshot, now=now), snapshot.id)


def _shuffle_within_rank_groups(
    snapshots: list[ProxySessionSnapshot],
    *,
    seed: int | None,
    group_key: Callable[[ProxySessionSnapshot], tuple[int, int, int, int]],
) -> list[ProxySessionSnapshot]:
    rng = random.Random(seed)
    shuffled: list[ProxySessionSnapshot] = []
    for _key, items_iter in groupby(snapshots, key=group_key):
        items = list(items_iter)
        rng.shuffle(items)
        shuffled.extend(items)
    return shuffled


def _rank_group_key_for_now(
    now: datetime,
) -> Callable[[ProxySessionSnapshot], tuple[int, int, int, int]]:
    def _key(snapshot: ProxySessionSnapshot) -> tuple[int, int, int, int]:
        return _existing_session_recovery_group(snapshot, now=now)

    return _key


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
    if value.tzinfo is not None and reference.tzinfo is not None:
        return value.astimezone(reference.tzinfo)
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
        compare_now = _datetime_for_compare(now, cooldown_until)
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


def _candidate_existing_sessions(
    store: ProxySessionStore,
    base_username: str,
    *,
    limit: int,
    shuffle: bool,
    seed: int | None,
    now: datetime | None = None,
) -> list[ProxySessionSnapshot]:
    compare_now = now or utc_now()
    # limit <= 0 means the whole dynamic inventory for this sticky base.
    fetch_limit = (
        max(EXISTING_SESSION_RANKING_FETCH_LIMIT, 50_000)
        if limit <= 0
        else max(limit, EXISTING_SESSION_RANKING_FETCH_LIMIT)
    )
    snapshots = store.list_proxy_sessions(limit=fetch_limit, proxy_base_username=base_username)
    snapshots = sorted(
        snapshots,
        key=lambda snapshot: _existing_session_recovery_rank(snapshot, now=compare_now),
    )
    if shuffle:
        snapshots = _shuffle_within_rank_groups(
            snapshots,
            seed=seed,
            group_key=_rank_group_key_for_now(compare_now),
        )
    if limit <= 0:
        return snapshots
    return snapshots[:limit]


def _discover_missing_index_sessions(
    store: ProxySessionStore,
    base_username: str,
    *,
    start: int,
    end: int,
    suffix_template: str,
    known_proxy_usernames: set[str],
    shuffle: bool,
    seed: int | None,
) -> list[ProxySessionSnapshot]:
    """Create inventory rows for sticky indices in [start, end] not yet known."""
    missing_indices = [
        index
        for index in _candidate_indices(start, end, shuffle=shuffle, seed=seed)
        if format_sticky_username(base_username, index, suffix_template)
        not in known_proxy_usernames
    ]
    discovered: list[ProxySessionSnapshot] = []
    for index in missing_indices:
        proxy_username = format_sticky_username(base_username, index, suffix_template)
        discovered.append(
            store.upsert_proxy_session(
                proxy_base_username=base_username,
                session_name=f"user{index}",
                proxy_username=proxy_username,
            )
        )
    return discovered


def _merge_existing_with_index_discovery(
    store: ProxySessionStore,
    base_username: str,
    existing: list[ProxySessionSnapshot],
    *,
    start: int,
    end: int,
    suffix_template: str,
    shuffle: bool,
    seed: int | None,
) -> list[ProxySessionSnapshot]:
    """Full-pool candidates = ranked existing inventory ∪ missing user{start}..user{end}."""
    known = {snapshot.proxy_username for snapshot in existing}
    discovered = _discover_missing_index_sessions(
        store,
        base_username,
        start=start,
        end=end,
        suffix_template=suffix_template,
        known_proxy_usernames=known,
        shuffle=shuffle,
        seed=seed,
    )
    # Existing first (recovery-ranked / history-aware); newly discovered indexes after.
    return [*existing, *discovered]


def _should_stop_after_terminal_stage(
    snapshot: ProxySessionSnapshot,
    *,
    stage_ran: bool,
) -> bool:
    return stage_ran and snapshot.status in {STATUS_COOLDOWN, STATUS_RETIRED}


def _fresh_active_checked_after(now: datetime, active_freshness_seconds: int) -> datetime | None:
    if active_freshness_seconds <= 0:
        return None
    return now - timedelta(seconds=active_freshness_seconds)


def _normalize_canary_answer(text: str) -> str:
    normalized = " ".join(text.strip().split()).strip("`'\" ")
    while len(normalized) > 1 and normalized.endswith("."):
        normalized = normalized[:-1].strip().strip("`'\" ")
    return normalized.casefold()


def _canary_answer_matches(actual: str, expected: str) -> bool:
    expected = expected.strip()
    if not expected:
        return True
    actual_normalized = _normalize_canary_answer(actual)
    expected_normalized = _normalize_canary_answer(expected)
    if actual_normalized == expected_normalized:
        return True
    prefix_pattern = rf"^{re.escape(expected_normalized)}(?:\s|[.,;:!?])"
    return re.match(prefix_pattern, actual_normalized) is not None


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


def _run_duck_canary(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    config: ServiceConfig,
    prompt: str,
    expected_answer: str,
    repeats: int,
) -> ProxySessionSnapshot:
    expected = expected_answer.strip()
    attempts = max(repeats, 1)
    runner = DuckAiRunner()
    try:
        for attempt in range(1, attempts + 1):
            try:
                result = runner.run_prompt(config, prompt)
            except DuckAiRateLimitedError as exc:
                store.record_event(
                    proxy_session_id=snapshot.id,
                    event_type="duck_canary_rate_limited",
                    message=str(exc),
                    raw_json={
                        "attempt": attempt,
                        "attempts": attempts,
                    },
                )
                return store.mark_duck_canary_rate_limited(
                    snapshot.id,
                    error_message=str(exc),
                )
            except Exception as exc:
                store.record_event(
                    proxy_session_id=snapshot.id,
                    event_type="duck_canary_error",
                    message=repr(exc),
                    raw_json={
                        "attempt": attempt,
                        "attempts": attempts,
                    },
                )
                return store.mark_duck_canary_error(
                    snapshot.id,
                    error_message=repr(exc),
                )

            actual = result.answer_text.strip()
            if not _canary_answer_matches(actual, expected):
                store.record_event(
                    proxy_session_id=snapshot.id,
                    event_type="duck_canary_unexpected_answer",
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
                return store.mark_duck_canary_error(
                    snapshot.id,
                    error_message=(
                        "duck canary returned unexpected answer: "
                        f"expected={expected!r} actual={actual!r}"
                    ),
                )

        store.record_event(
            proxy_session_id=snapshot.id,
            event_type="duck_canary_success",
            message=f"passed {attempts} canary attempt(s)",
            raw_json={
                "attempts": attempts,
                "expected": expected,
            },
        )
        return store.mark_duck_canary_success(snapshot.id)
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


def _snapshot_ip_candidates(snapshot: ProxySessionSnapshot) -> list[str]:
    from googleaisearch2api.proxy_sessions import normalize_ip_vector

    candidates = list(getattr(snapshot, "ip_vector", None) or [])
    if snapshot.primary_ip:
        candidates.append(snapshot.primary_ip)
    return normalize_ip_vector(candidates)


def _skip_known_google_blocked_ip(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    *,
    base_username: str,
    enabled: bool,
) -> ProxySessionSnapshot:
    candidates = _snapshot_ip_candidates(snapshot)
    if not enabled or not candidates:
        return snapshot

    blocked = store.find_google_blocked_session_for_ips(
        base_username,
        candidates,
        exclude_session_id=snapshot.id,
    )
    if blocked is None:
        return snapshot

    message = (
        f"egress IP(s) {', '.join(candidates)} matched Google-blocked session "
        f"{blocked.proxy_username}"
    )
    store.record_event(
        proxy_session_id=snapshot.id,
        event_type="known_google_blocked_ip_skipped",
        message=message,
        raw_json={
            "candidate_ips": candidates,
            "matched_session_id": blocked.id,
            "matched_proxy_username": blocked.proxy_username,
            "matched_primary_ip": blocked.primary_ip,
            "matched_canary_block_count": blocked.canary_block_count,
            "matched_request_block_count": blocked.request_block_count,
            "matched_last_blocked_at": (
                blocked.last_blocked_at.isoformat() if blocked.last_blocked_at else None
            ),
        },
    )
    return store.mark_session_cooldown(snapshot.id, reason=message)


def _skip_known_google_blocked_prefix(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    *,
    base_username: str,
    enabled: bool,
    min_blocked_count: int,
) -> ProxySessionSnapshot:
    if not enabled or not snapshot.primary_ip:
        return snapshot

    blocked_prefix = store.find_google_blocked_prefix_for_ip(
        base_username,
        snapshot.primary_ip,
        min_blocked_count=min_blocked_count,
        exclude_session_id=snapshot.id,
    )
    if blocked_prefix is None:
        return snapshot

    message = (
        f"egress IP {snapshot.primary_ip} matched Google-blocked prefix "
        f"{blocked_prefix.prefix} ({blocked_prefix.blocked_count} blocked, "
        f"{blocked_prefix.success_count} successful)"
    )
    store.record_event(
        proxy_session_id=snapshot.id,
        event_type="known_google_blocked_prefix_skipped",
        message=message,
        raw_json={
            "prefix": blocked_prefix.prefix,
            "blocked_count": blocked_prefix.blocked_count,
            "success_count": blocked_prefix.success_count,
            "matched_session_id": blocked_prefix.matched_session.id,
            "matched_proxy_username": blocked_prefix.matched_session.proxy_username,
            "matched_primary_ip": blocked_prefix.matched_session.primary_ip,
            "matched_canary_block_count": (
                blocked_prefix.matched_session.canary_block_count
            ),
            "matched_request_block_count": (
                blocked_prefix.matched_session.request_block_count
            ),
            "matched_last_blocked_at": (
                blocked_prefix.matched_session.last_blocked_at.isoformat()
                if blocked_prefix.matched_session.last_blocked_at
                else None
            ),
        },
    )
    return store.mark_session_cooldown(snapshot.id, reason=message)


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


def _run_fast_http_prefilter(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    config: ServiceConfig,
    *,
    timeout_s: float,
    check_google: bool,
) -> tuple[ProxySessionSnapshot, dict[str, object]]:
    """L0 curl_cffi screen. Rejects dead/blocked exits without launching Chrome."""
    result = probe_proxy_http_fast(
        config,
        timeout_s=timeout_s,
        check_google=check_google,
    )
    payload = result.as_dict()
    store.record_event(
        proxy_session_id=snapshot.id,
        event_type="fast_http_prefilter",
        message=("ok" if result.ok else (result.reason or "failed")),
        raw_json=payload,
    )
    # Always persist observed egress IPs when available (including L0 rejects)
    # so the inventory can learn bad exits without a second probe.
    if result.ips:
        snapshot = store.update_egress(
            proxy_session_id=snapshot.id,
            ips=result.ips,
            source="fast_http",
            raw_json=result.raw,
        )
        if snapshot.status in {STATUS_COOLDOWN, STATUS_RETIRED}:
            return snapshot, payload

    if not result.ok:
        reason = result.reason or "fast http prefilter failed"
        return store.mark_session_cooldown(snapshot.id, reason=reason), payload

    return snapshot, payload


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
    parser.add_argument(
        "--existing-sessions",
        action="store_true",
        help=(
            "Probe existing proxy_sessions rows (optionally merged with missing "
            "--start/--end indices via --discover-missing-indices)."
        ),
    )
    parser.add_argument(
        "--existing-session-limit",
        type=int,
        default=0,
        help=(
            "Maximum existing proxy_sessions rows to probe with --existing-sessions. "
            "0 means the whole dynamic inventory for the sticky base."
        ),
    )
    parser.add_argument(
        "--discover-missing-indices",
        action="store_true",
        help=(
            "With --existing-sessions, also include sticky indices in --start..--end "
            "that are not yet in the inventory (full-pool = existing ∪ missing)."
        ),
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=0,
        help=(
            "Maximum sessions that may run expensive browser egress/canary probes. "
            "Fast HTTP prefilter rejections do not count. 0 means unlimited."
        ),
    )
    parser.add_argument(
        "--fast-http-prefilter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use curl_cffi to cheaply screen proxy tunnel/egress/Google shell before "
            "browser canary (default: enabled)."
        ),
    )
    parser.add_argument(
        "--fast-http-timeout",
        type=float,
        default=8.0,
        help="Timeout seconds for each curl_cffi request in the fast prefilter.",
    )
    parser.add_argument(
        "--fast-http-scan-limit",
        type=int,
        default=0,
        help=(
            "Maximum candidates to run through the fast HTTP prefilter. "
            "0 means unlimited (scan whole dynamic pool). Browser max-probes is separate."
        ),
    )
    parser.add_argument(
        "--fast-http-workers",
        type=int,
        default=16,
        help="Concurrent curl_cffi workers for the fast HTTP prefilter.",
    )
    parser.add_argument(
        "--full-fast-http-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, continue fast-HTTP screening the whole candidate set even after "
            "stop-after-active is reached; only browser canaries stop early (default: on)."
        ),
    )
    parser.add_argument(
        "--fast-http-skip-google",
        action="store_true",
        help="Only check egress IPs in the fast prefilter; skip Google AI URL check.",
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
        "--allow-known-google-blocked-ip",
        action="store_true",
        help="Do not skip exits whose IP already has Google blocked evidence.",
    )
    parser.add_argument(
        "--allow-known-google-blocked-prefix",
        action="store_true",
        help=(
            "Do not skip exits from a /24 IPv4 or /48 IPv6 prefix with repeated "
            "Google blocked evidence and no successful session."
        ),
    )
    parser.add_argument(
        "--known-google-blocked-prefix-min-count",
        type=int,
        default=1,
        help="Minimum Google-blocked sessions in a prefix before prefix-level skipping.",
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
    parser.add_argument(
        "--active-freshness-seconds",
        type=int,
        default=0,
        help=(
            "When used with --stop-after-active, count only active sessions whose "
            "Google canary was checked within this many seconds. Default counts "
            "all active sessions."
        ),
    )
    parser.add_argument("--skip-google-canary", action="store_true", help="Skip Google canary.")
    parser.add_argument("--skip-duck-canary", action="store_true", help="Skip Duck.ai canary.")
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
    parser.add_argument(
        "--duck-canary-prompt",
        default=DEFAULT_CANARY_PROMPT,
        help="Duck.ai canary prompt. Default asks for a deterministic short answer.",
    )
    parser.add_argument(
        "--duck-canary-expected-answer",
        default=DEFAULT_CANARY_EXPECTED_ANSWER,
        help=(
            "Only mark Duck.ai canary successful if the normalized answer matches this value. "
            "Set to an empty string to accept any non-rate-limited answer."
        ),
    )
    parser.add_argument(
        "--duck-canary-repeats",
        type=int,
        default=DEFAULT_DUCK_CANARY_REPEATS,
        help="Require this many consecutive successful Duck.ai canary answers.",
    )
    args = parser.parse_args()

    if args.end < args.start:
        raise SystemExit("--end must be >= --start")
    if args.canary_repeats < 1:
        raise SystemExit("--canary-repeats must be >= 1")
    if args.duck_canary_repeats < 1:
        raise SystemExit("--duck-canary-repeats must be >= 1")
    if args.known_google_blocked_prefix_min_count < 1:
        raise SystemExit("--known-google-blocked-prefix-min-count must be >= 1")
    if args.existing_session_limit < 0:
        raise SystemExit("--existing-session-limit must be >= 0 (0 = whole inventory)")
    if args.max_probes < 0:
        raise SystemExit("--max-probes must be >= 0")
    if args.fast_http_scan_limit < 0:
        raise SystemExit("--fast-http-scan-limit must be >= 0")
    if args.fast_http_workers < 1:
        raise SystemExit("--fast-http-workers must be >= 1")
    if args.fast_http_timeout <= 0:
        raise SystemExit("--fast-http-timeout must be > 0")
    if args.active_freshness_seconds < 0:
        raise SystemExit("--active-freshness-seconds must be >= 0")

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

    def stop_target_reached() -> bool:
        if args.stop_after_active <= 0:
            return False
        checked_after = _fresh_active_checked_after(utc_now(), args.active_freshness_seconds)
        return (
            store.count_active_sessions(base_username, checked_after=checked_after)
            >= args.stop_after_active
        )

    records: list[dict[str, object]] = []
    probed_count = 0
    fast_http_screened = 0
    fast_http_rejected = 0
    browser_ip_deduped = 0
    known_block_prefiltered = 0
    # IPs already spent on a browser canary in this process (avoid same-IP re-burn).
    canary_probed_ips: set[str] = set()

    def append_record(
        snapshot: ProxySessionSnapshot,
        *,
        skipped: str | None = None,
        iplark_result: object | None = None,
        fast_http_payload: dict[str, object] | None = None,
    ) -> None:
        records.append(
            {
                "session": _snapshot_json(snapshot),
                "iplark": asdict(iplark_result) if iplark_result is not None else None,
                "fast_http": fast_http_payload,
                **({"skipped": skipped} if skipped is not None else {}),
            }
        )

    def apply_known_block_filters(
        snapshot: ProxySessionSnapshot,
    ) -> ProxySessionSnapshot:
        checked = _skip_known_google_blocked_ip(
            store,
            snapshot,
            base_username=base_username,
            enabled=not args.allow_known_google_blocked_ip,
        )
        if checked is snapshot:
            checked = _skip_known_google_blocked_prefix(
                store,
                snapshot,
                base_username=base_username,
                enabled=not args.allow_known_google_blocked_prefix,
                min_blocked_count=args.known_google_blocked_prefix_min_count,
            )
        return checked

    def run_browser_stages(
        snapshot: ProxySessionSnapshot,
        *,
        fast_http_payload: dict[str, object] | None,
    ) -> None:
        nonlocal probed_count, known_block_prefiltered
        # Full-pool fast sweeps keep screening everyone; only browser canaries stop early.
        if stop_target_reached() or (args.max_probes > 0 and probed_count >= args.max_probes):
            append_record(
                snapshot,
                skipped=(
                    "hot target already met"
                    if stop_target_reached()
                    else "browser probe budget exhausted"
                ),
                fast_http_payload=fast_http_payload,
            )
            return

        candidate_config = build_proxy_config_for_session(base_config, snapshot.proxy_username)
        iplark_result = None

        # Cheap known-bad IP/prefix filter BEFORE spending browser max-probes budget.
        if snapshot.primary_ip:
            pre = apply_known_block_filters(snapshot)
            if pre is not snapshot or pre.status in {STATUS_COOLDOWN, STATUS_RETIRED}:
                known_block_prefiltered += 1
                append_record(
                    pre,
                    iplark_result=iplark_result,
                    fast_http_payload=fast_http_payload,
                )
                return
            snapshot = pre
            ip_key = snapshot.primary_ip.strip()
            if ip_key in canary_probed_ips:
                append_record(
                    snapshot,
                    skipped=f"egress IP {ip_key} already browser-probed in this run",
                    fast_http_payload=fast_http_payload,
                )
                return

        egress_stage_ran = False
        if not (args.fast_http_prefilter and snapshot.primary_ip):
            if args.fast_ipapi_egress:
                egress_stage_ran = True
                snapshot, iplark_result = _run_fast_ipapi_egress(
                    store,
                    snapshot,
                    candidate_config,
                    min_quality_score=args.min_quality_score,
                    timeout_s=args.fast_egress_timeout,
                )
            elif not args.skip_egress:
                egress_stage_ran = True
                snapshot = _run_egress(
                    store,
                    snapshot,
                    candidate_config,
                    checks=max(args.egress_checks, 1),
                )
        if _should_stop_after_terminal_stage(snapshot, stage_ran=egress_stage_ran):
            append_record(
                snapshot,
                iplark_result=iplark_result,
                fast_http_payload=fast_http_payload,
            )
            return

        iplark_stage_ran = False
        if not args.skip_iplark and not args.fast_ipapi_egress:
            iplark_stage_ran = True
            snapshot, iplark_result = _run_iplark(
                store,
                snapshot,
                direct_config,
                min_quality_score=args.min_quality_score,
                risk_source=args.risk_source,
            )
        if _should_stop_after_terminal_stage(snapshot, stage_ran=iplark_stage_ran):
            append_record(
                snapshot,
                iplark_result=iplark_result,
                fast_http_payload=fast_http_payload,
            )
            return

        # If IP was only discovered during egress, apply known-block filters now
        # (still before the expensive canary when possible).
        if snapshot.primary_ip:
            checked = apply_known_block_filters(snapshot)
            if checked is not snapshot or checked.status in {STATUS_COOLDOWN, STATUS_RETIRED}:
                known_block_prefiltered += 1
                append_record(
                    checked,
                    iplark_result=iplark_result,
                    fast_http_payload=fast_http_payload,
                )
                return
            snapshot = checked
            ip_key = snapshot.primary_ip.strip()
            if ip_key in canary_probed_ips:
                append_record(
                    snapshot,
                    skipped=f"egress IP {ip_key} already browser-probed in this run",
                    iplark_result=iplark_result,
                    fast_http_payload=fast_http_payload,
                )
                return

        # max-probes counts real browser canaries only (not known-block prefilters).
        if stop_target_reached() or (args.max_probes > 0 and probed_count >= args.max_probes):
            append_record(
                snapshot,
                skipped=(
                    "hot target already met"
                    if stop_target_reached()
                    else "browser probe budget exhausted"
                ),
                iplark_result=iplark_result,
                fast_http_payload=fast_http_payload,
            )
            return

        probed_count += 1
        if snapshot.primary_ip:
            canary_probed_ips.add(snapshot.primary_ip.strip())

        if not args.skip_duck_canary:
            snapshot = _run_duck_canary(
                store,
                snapshot,
                candidate_config,
                args.duck_canary_prompt,
                args.duck_canary_expected_answer,
                args.duck_canary_repeats,
            )
        if not args.skip_google_canary:
            snapshot = _run_canary(
                store,
                snapshot,
                candidate_config,
                args.canary_prompt,
                args.canary_expected_answer,
                args.canary_repeats,
            )
        append_record(
            snapshot,
            iplark_result=iplark_result,
            fast_http_payload=fast_http_payload,
        )

    existing_candidates: list[ProxySessionSnapshot] = []
    if args.existing_sessions:
        existing_candidates = _candidate_existing_sessions(
            store,
            base_username,
            limit=args.existing_session_limit,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    if existing_candidates and args.discover_missing_indices:
        # Interval full-pool: every known row plus any missing user{start}..user{end}.
        candidate_snapshots = _merge_existing_with_index_discovery(
            store,
            base_username,
            existing_candidates,
            start=args.start,
            end=args.end,
            suffix_template=args.suffix_template,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        candidate_mode = "existing+index"
    elif existing_candidates:
        candidate_snapshots = existing_candidates
        candidate_mode = "existing"
    else:
        # Dynamic pool empty (or not using --existing-sessions): discover indexes once.
        candidate_snapshots = _discover_missing_index_sessions(
            store,
            base_username,
            start=args.start,
            end=args.end,
            suffix_template=args.suffix_template,
            known_proxy_usernames=set(),
            shuffle=args.shuffle,
            seed=args.seed,
        )
        candidate_mode = "index"

    # Phase 1: local skip filters (no network).
    to_screen: list[ProxySessionSnapshot] = []
    for snapshot in candidate_snapshots:
        skip_reason = _skip_candidate_reason(
            snapshot,
            now=utc_now(),
            refresh_active=args.refresh_active,
            retry_cooldown=args.retry_cooldown,
            retry_retired=args.retry_retired,
            only_risk_retired=args.only_risk_retired,
        )
        if skip_reason:
            append_record(snapshot, skipped=skip_reason)
            continue
        to_screen.append(snapshot)

    if args.fast_http_scan_limit > 0:
        to_screen = to_screen[: args.fast_http_scan_limit]

    # Phase 2: concurrent curl_cffi prefilter over the dynamic pool.
    passed: list[tuple[ProxySessionSnapshot, dict[str, object] | None]] = []
    if args.fast_http_prefilter and to_screen:

        def _screen_one(
            snapshot: ProxySessionSnapshot,
        ) -> tuple[ProxySessionSnapshot, dict[str, object]]:
            candidate_config = build_proxy_config_for_session(
                base_config,
                snapshot.proxy_username,
            )
            return _run_fast_http_prefilter(
                store,
                snapshot,
                candidate_config,
                timeout_s=args.fast_http_timeout,
                check_google=not args.fast_http_skip_google,
            )

        with ThreadPoolExecutor(max_workers=args.fast_http_workers) as executor:
            futures = [executor.submit(_screen_one, snapshot) for snapshot in to_screen]
            for future in as_completed(futures):
                snapshot, fast_http_payload = future.result()
                fast_http_screened += 1
                if snapshot.status in {STATUS_COOLDOWN, STATUS_RETIRED}:
                    fast_http_rejected += 1
                    append_record(snapshot, fast_http_payload=fast_http_payload)
                else:
                    passed.append((snapshot, fast_http_payload))
    else:
        passed = [(snapshot, None) for snapshot in to_screen]

    # Phase 2b: rank + one-session-per-IP before burning browser canary budget.
    browser_candidates, ip_dupes = _dedupe_browser_candidates_by_ip(passed)
    browser_ip_deduped = len(ip_dupes)
    for snapshot, fast_http_payload, reason in ip_dupes:
        append_record(snapshot, skipped=reason, fast_http_payload=fast_http_payload)

    # Phase 3: expensive browser canaries only for L0 survivors (IP-deduped).
    for snapshot, fast_http_payload in browser_candidates:
        if not args.full_fast_http_sweep and stop_target_reached():
            break
        run_browser_stages(snapshot, fast_http_payload=fast_http_payload)

    active_sessions = store.count_active_sessions(base_username)
    payload = {
        "base_username": base_username,
        "active_sessions": active_sessions,
        "probed_sessions": probed_count,
        "fast_http_screened": fast_http_screened,
        "fast_http_rejected": fast_http_rejected,
        "fast_http_passed": len(passed),
        "browser_candidates": len(browser_candidates),
        "browser_ip_deduped": browser_ip_deduped,
        "known_block_prefiltered": known_block_prefiltered,
        "browser_unique_ips_probed": len(canary_probed_ips),
        "candidate_mode": candidate_mode,
        "candidate_total": len(candidate_snapshots),
        "full_fast_http_sweep": args.full_fast_http_sweep,
        "stop_after_active": args.stop_after_active,
        "target_met": (
            args.stop_after_active <= 0 or active_sessions >= args.stop_after_active
        ),
        "records": records,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    # Auto-recovery used to treat any probe process exit 0 as success even when
    # sticky_active_sessions stayed at 0 after repeatedly canary-blocking the same
    # sessions. Fail the process when a target was requested but not met.
    if args.stop_after_active > 0 and active_sessions < args.stop_after_active:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
