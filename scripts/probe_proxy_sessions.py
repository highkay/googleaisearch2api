from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from googleaisearch2api.browser import GoogleAiBlockedError, GoogleAiRunner
from googleaisearch2api.config import ServiceConfig, get_settings
from googleaisearch2api.db import create_db_engine, create_session_factory, create_tables
from googleaisearch2api.egress import probe_egress
from googleaisearch2api.iplark import IplarkProbeResult, probe_iplark_ip
from googleaisearch2api.logging import configure_logging
from googleaisearch2api.proxy_sessions import (
    DEFAULT_STICKY_SUFFIX_TEMPLATE,
    STATUS_COOLDOWN,
    STATUS_RETIRED,
    ProxySessionSnapshot,
    ProxySessionStore,
    build_proxy_config_for_session,
    format_sticky_username,
    google_block_has_ip_mismatch,
    parse_google_block_ips,
    resolve_proxy_base_username,
)


def _snapshot_json(snapshot: ProxySessionSnapshot) -> dict:
    payload = asdict(snapshot)
    for key, value in list(payload.items()):
        if hasattr(value, "isoformat"):
            payload[key] = value.isoformat()
    return payload


def _run_canary(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    config: ServiceConfig,
    prompt: str,
) -> ProxySessionSnapshot:
    runner = GoogleAiRunner()
    try:
        runner.run_prompt(config, prompt)
    except GoogleAiBlockedError as exc:
        block_ips = parse_google_block_ips(str(exc))
        store.record_event(
            proxy_session_id=snapshot.id,
            event_type="google_canary_blocked",
            message=str(exc),
            raw_json={
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
        return store.mark_session_cooldown(snapshot.id, reason=f"google canary failed: {exc!r}")
    finally:
        runner.close()
    return store.mark_canary_success(snapshot.id)


def _run_iplark(
    store: ProxySessionStore,
    snapshot: ProxySessionSnapshot,
    direct_config: ServiceConfig,
    *,
    min_quality_score: int,
) -> tuple[ProxySessionSnapshot, IplarkProbeResult | None]:
    if not snapshot.primary_ip:
        snapshot = store.mark_session_cooldown(
            snapshot.id,
            reason="egress probe did not find an IP",
        )
        return snapshot, None
    try:
        result = probe_iplark_ip(snapshot.primary_ip, direct_config)
    except Exception as exc:
        store.record_event(
            proxy_session_id=snapshot.id,
            event_type="iplark_error",
            message=repr(exc),
        )
        snapshot = store.mark_session_cooldown(
            snapshot.id,
            reason=f"iplark probe failed: {exc!r}",
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
    parser.add_argument("--min-quality-score", type=int, default=70, help="IPLark score threshold.")
    parser.add_argument("--skip-egress", action="store_true", help="Skip egress probing.")
    parser.add_argument("--skip-iplark", action="store_true", help="Skip IPLark probing.")
    parser.add_argument("--skip-google-canary", action="store_true", help="Skip Google canary.")
    parser.add_argument(
        "--canary-prompt",
        default="What is OpenAI Responses API? Answer in one short sentence.",
    )
    args = parser.parse_args()

    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    settings = get_settings()
    configure_logging(settings.app_log_level)
    settings.ensure_directories()
    engine = create_db_engine(str(settings.db_path))
    create_tables(engine)
    store = ProxySessionStore(create_session_factory(engine))

    base_config = ServiceConfig.from_settings(settings)
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
    for index in range(args.start, args.end + 1):
        proxy_username = format_sticky_username(base_username, index, args.suffix_template)
        session_name = f"user{index}"
        snapshot = store.upsert_proxy_session(
            proxy_base_username=base_username,
            session_name=session_name,
            proxy_username=proxy_username,
        )
        candidate_config = build_proxy_config_for_session(base_config, proxy_username)

        iplark_result = None
        if not args.skip_egress:
            snapshot = _run_egress(
                store,
                snapshot,
                candidate_config,
                checks=max(args.egress_checks, 1),
            )
        if snapshot.status in {STATUS_COOLDOWN, STATUS_RETIRED}:
            records.append({"session": _snapshot_json(snapshot), "iplark": None})
            continue

        if not args.skip_iplark:
            snapshot, iplark_result = _run_iplark(
                store,
                snapshot,
                direct_config,
                min_quality_score=args.min_quality_score,
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
            snapshot = _run_canary(store, snapshot, candidate_config, args.canary_prompt)

        records.append(
            {
                "session": _snapshot_json(snapshot),
                "iplark": asdict(iplark_result) if iplark_result else None,
            }
        )

    print(
        json.dumps(
            {"base_username": base_username, "records": records},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
