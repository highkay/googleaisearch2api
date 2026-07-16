from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from googleaisearch2api.browser import resolve_browser_proxy
from googleaisearch2api.config import ServiceConfig
from googleaisearch2api.db import create_db_engine, create_session_factory, create_tables, utc_now
from googleaisearch2api.proxy_sessions import (
    STATUS_ACTIVE,
    STATUS_COOLDOWN,
    STATUS_RETIRED,
    STATUS_RISK_CHECKED,
    ProxySessionConfigError,
    ProxySessionSelector,
    ProxySessionStore,
    ProxySessionUnavailableError,
    build_proxy_config_for_session,
    format_sticky_username,
    google_block_has_ip_mismatch,
    hash_ip_vector,
    is_risk_metadata_retire_reason,
    normalize_ip_vector,
    parse_google_block_ips,
    resolve_proxy_base_username,
)


def _make_store(tmp_path: Path) -> ProxySessionStore:
    engine = create_db_engine(str(tmp_path / "proxy.sqlite3"))
    create_tables(engine)
    return ProxySessionStore(create_session_factory(engine))


def test_format_sticky_username_accepts_any_base_prefix() -> None:
    assert format_sticky_username("US", 1) == "US.user1"
    assert format_sticky_username("JP", 2) == "JP.user2"
    assert format_sticky_username("openai", 3) == "openai.user3"
    assert format_sticky_username("openai", 4, "{base}-session-{n}") == "openai-session-4"

    with pytest.raises(ProxySessionConfigError):
        format_sticky_username("", 1)


def test_resolve_proxy_base_username_supports_explicit_and_embedded_credentials() -> None:
    explicit = ServiceConfig(
        browser_proxy_server="http://192.0.2.1:2260",
        browser_proxy_username="openai",
        browser_proxy_password="pass",
    )
    embedded = ServiceConfig(browser_proxy_server="http://JP:pass@192.0.2.1:2260")

    assert resolve_proxy_base_username(explicit) == "openai"
    assert resolve_proxy_base_username(embedded) == "JP"


def test_build_proxy_config_for_session_overrides_embedded_proxy_username() -> None:
    config = ServiceConfig(browser_proxy_server="http://US:pass@192.0.2.1:2260")

    session_config = build_proxy_config_for_session(config, "US.user1")
    proxy = resolve_browser_proxy(session_config)

    assert proxy == {
        "server": "http://192.0.2.1:2260",
        "username": "US.user1",
        "password": "pass",
        "bypass": None,
    }


def test_parse_google_block_ips_detects_ipv4_ipv6_mismatch() -> None:
    message = (
        "Google blocked the session while opening query page: our systems have detected "
        "unusual traffic from your computer network. ip address: 66.187.6.127 "
        "≠ 2a09:bac5:624d:2da5::48c:59 time: 2026-05-26t04:15:43z"
    )

    ips = parse_google_block_ips(message)

    assert ips == ["66.187.6.127", "2a09:bac5:624d:2da5::48c:59"]
    assert google_block_has_ip_mismatch(ips) is True


def test_ip_vector_hash_is_stable_and_deduplicated() -> None:
    first = normalize_ip_vector(["203.0.113.2", "203.0.113.1", "bad", "203.0.113.2"])
    second = normalize_ip_vector(["203.0.113.1", "203.0.113.2"])

    assert first == second
    assert hash_ip_vector(first) == hash_ip_vector(second)


def test_risk_metadata_retire_reasons_are_retryable() -> None:
    assert is_risk_metadata_retire_reason("iplark flagged public proxy/threat")
    assert is_risk_metadata_retire_reason("iplark score 34 below threshold 70")
    assert is_risk_metadata_retire_reason("iplark score missing")
    assert not is_risk_metadata_retire_reason("duplicate egress with session 1")


def test_proxy_session_selector_uses_active_session_without_hardcoded_country(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
        status=STATUS_ACTIVE,
    )
    selector = ProxySessionSelector(store)
    config = ServiceConfig(
        browser_proxy_server="http://192.0.2.1:2260",
        browser_proxy_username="openai",
        browser_proxy_password="pass",
        resin_sticky_session_enabled=True,
    )

    selection = selector.select(config)

    assert selection is not None
    assert selection.session.proxy_username == "openai.user1"
    assert selection.config.browser_proxy_username == "openai.user1"


def test_duck_canary_success_does_not_promote_google_active_session(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )
    store.update_egress(
        proxy_session_id=snapshot.id,
        ips=["203.0.113.10"],
        source="test",
    )
    store.update_iplark_result(
        proxy_session_id=snapshot.id,
        quality_score=80,
        min_quality_score=0,
    )

    snapshot = store.mark_duck_canary_success(snapshot.id)

    assert snapshot.status == STATUS_RISK_CHECKED
    assert snapshot.duck_canary_status == "ok"
    assert snapshot.duck_canary_success_count == 1
    assert snapshot.request_success_count == 0
    assert store.count_active_sessions("openai") == 0


def test_count_active_sessions_can_require_recent_google_canary(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
        status=STATUS_ACTIVE,
    )
    store.mark_canary_success(snapshot.id)

    assert store.count_active_sessions("openai") == 1
    assert store.count_active_sessions(
        "openai",
        checked_after=utc_now() - timedelta(minutes=1),
    ) == 1
    assert store.count_active_sessions(
        "openai",
        checked_after=utc_now() + timedelta(minutes=1),
    ) == 0


def test_canary_success_clears_stale_cooldown_reason(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
        status=STATUS_ACTIVE,
    )
    snapshot = store.mark_session_cooldown(snapshot.id, reason="old block reason")

    assert snapshot.retire_reason == "old block reason"

    snapshot = store.mark_canary_success(snapshot.id)

    assert snapshot.status == STATUS_ACTIVE
    assert snapshot.cooldown_until is None
    assert snapshot.retire_reason is None


def test_request_success_clears_stale_cooldown_reason(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
        status=STATUS_ACTIVE,
    )
    snapshot = store.mark_session_cooldown(snapshot.id, reason="old block reason")

    assert snapshot.retire_reason == "old block reason"

    store.finish_request_success(snapshot.id)
    snapshot = store.list_proxy_sessions(limit=1, proxy_base_username="openai")[0]

    assert snapshot.status == STATUS_ACTIVE
    assert snapshot.cooldown_until is None
    assert snapshot.retire_reason is None


def test_expired_proven_cooldown_session_is_selectable(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
        status=STATUS_ACTIVE,
    )
    store.mark_canary_success(snapshot.id)
    store.mark_session_cooldown(snapshot.id, reason="temporary non-answer page", hours=1)

    assert store.count_active_sessions("openai") == 0
    assert store.count_selectable_sessions("openai") == 0

    store.mark_session_cooldown(snapshot.id, reason="cooldown elapsed", hours=-1)
    selector = ProxySessionSelector(store)
    config = ServiceConfig(
        browser_proxy_server="http://192.0.2.1:2260",
        browser_proxy_username="openai",
        browser_proxy_password="pass",
        resin_sticky_session_enabled=True,
    )
    selection = selector.select(config)

    assert store.count_active_sessions("openai") == 0
    assert store.count_selectable_sessions("openai") == 1
    assert selection is not None
    assert selection.session.proxy_username == "openai.user1"


def test_duck_selector_can_use_google_cooldown_session(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )
    store.update_egress(
        proxy_session_id=snapshot.id,
        ips=["203.0.113.10"],
        source="test",
    )
    store.update_iplark_result(
        proxy_session_id=snapshot.id,
        quality_score=80,
        min_quality_score=0,
    )
    store.mark_session_cooldown(snapshot.id, reason="google blocked")
    store.mark_duck_canary_success(snapshot.id)
    selector = ProxySessionSelector(store)
    config = ServiceConfig(
        browser_proxy_server="http://192.0.2.1:2260",
        browser_proxy_username="openai",
        browser_proxy_password="pass",
        resin_sticky_session_enabled=True,
    )

    duck_selection = selector.select(config, engine="duck")

    assert duck_selection is not None
    assert duck_selection.session.proxy_username == "openai.user1"
    with pytest.raises(ProxySessionUnavailableError, match="No active sticky"):
        selector.select(config, engine="google")


def test_duck_selector_can_use_historical_duck_success_after_error(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )
    store.mark_duck_canary_success(snapshot.id)
    store.mark_duck_canary_error(snapshot.id, error_message="temporary tunnel failure")
    selector = ProxySessionSelector(store)
    config = ServiceConfig(
        browser_proxy_server="http://192.0.2.1:2260",
        browser_proxy_username="openai",
        browser_proxy_password="pass",
        resin_sticky_session_enabled=True,
    )

    duck_selection = selector.select(config, engine="duck")

    assert duck_selection is not None
    assert duck_selection.session.proxy_username == "openai.user1"
    assert duck_selection.config.browser_proxy_username == "openai.user1"


def test_proxy_session_selector_prefers_real_success_before_unproven_active_session(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    first = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
        status=STATUS_ACTIVE,
    )
    store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user2",
        proxy_username="openai.user2",
        status=STATUS_ACTIVE,
    )
    selector = ProxySessionSelector(store)
    config = ServiceConfig(
        browser_proxy_server="http://192.0.2.1:2260",
        browser_proxy_username="openai",
        browser_proxy_password="pass",
        resin_sticky_session_enabled=True,
    )

    first_selection = selector.select(config)
    assert first_selection is not None
    assert first_selection.session.proxy_username == "openai.user1"
    store.finish_request_success(first.id)

    second_selection = selector.select(config)

    assert second_selection is not None
    assert second_selection.session.proxy_username == "openai.user1"


def test_proxy_session_selector_rotates_between_proven_success_sessions(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    first = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
        status=STATUS_ACTIVE,
    )
    second = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user2",
        proxy_username="openai.user2",
        status=STATUS_ACTIVE,
    )
    store.finish_request_success(first.id)
    store.finish_request_success(second.id)
    selector = ProxySessionSelector(store)
    config = ServiceConfig(
        browser_proxy_server="http://192.0.2.1:2260",
        browser_proxy_username="openai",
        browser_proxy_password="pass",
        resin_sticky_session_enabled=True,
    )

    first_selection = selector.select(config)
    assert first_selection is not None
    assert first_selection.session.proxy_username == "openai.user1"

    second_selection = selector.select(config)

    assert second_selection is not None
    assert second_selection.session.proxy_username == "openai.user2"


def test_update_egress_retires_duplicate_ip_vector(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    first = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )
    second = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user2",
        proxy_username="openai.user2",
    )

    first = store.update_egress(
        proxy_session_id=first.id,
        ips=["203.0.113.10"],
        source="test",
    )
    second = store.update_egress(
        proxy_session_id=second.id,
        ips=["203.0.113.10"],
        source="test",
    )

    assert first.ip_vector_hash == second.ip_vector_hash
    assert second.status == STATUS_RETIRED
    assert second.duplicate_of_session_id == first.id


def test_update_egress_clears_stale_duplicate_when_ip_changes(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    first = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )
    second = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user2",
        proxy_username="openai.user2",
    )
    store.update_egress(
        proxy_session_id=first.id,
        ips=["203.0.113.10"],
        source="test",
    )
    duplicate = store.update_egress(
        proxy_session_id=second.id,
        ips=["203.0.113.10"],
        source="test",
    )

    refreshed = store.update_egress(
        proxy_session_id=duplicate.id,
        ips=["203.0.113.11"],
        source="test",
    )

    assert refreshed.status != STATUS_RETIRED
    assert refreshed.duplicate_of_session_id is None


def test_find_google_blocked_session_for_ip_uses_exact_base_and_ip(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    blocked = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )
    other = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user2",
        proxy_username="openai.user2",
    )
    blocked = store.update_egress(
        proxy_session_id=blocked.id,
        ips=["203.0.113.10"],
        source="test",
    )
    store.mark_canary_blocked(
        blocked.id,
        error_message="Google unusual traffic",
        block_ips=["203.0.113.10"],
    )

    found = store.find_google_blocked_session_for_ip(
        "openai",
        "203.0.113.10",
        exclude_session_id=other.id,
    )

    assert found is not None
    assert found.id == blocked.id
    assert store.find_google_blocked_session_for_ip("JP", "203.0.113.10") is None
    assert (
        store.find_google_blocked_session_for_ip(
            "openai",
            "203.0.113.10",
            exclude_session_id=blocked.id,
        )
        is None
    )


def test_find_google_blocked_session_for_ip_uses_google_block_observation(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    blocked = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )
    other = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user2",
        proxy_username="openai.user2",
    )
    blocked = store.update_egress(
        proxy_session_id=blocked.id,
        ips=["66.187.6.127"],
        source="test",
    )
    store.mark_canary_blocked(
        blocked.id,
        error_message="Google unusual traffic",
        block_ips=["2a09:bac5:624d:2da5::48c:59"],
    )

    found = store.find_google_blocked_session_for_ip(
        "openai",
        "2a09:bac5:624d:2da5::48c:59",
        exclude_session_id=other.id,
    )

    assert found is not None
    assert found.id == blocked.id
    assert found.primary_ip == "66.187.6.127"
    assert store.find_google_blocked_session_for_ip(
        "JP",
        "2a09:bac5:624d:2da5::48c:59",
    ) is None


def test_find_google_blocked_prefix_for_ip_uses_blocked_prefix_without_success(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    for index, ip in enumerate(
        ["203.0.113.10", "203.0.113.11", "203.0.113.12"],
        start=1,
    ):
        session = store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name=f"user{index}",
            proxy_username=f"openai.user{index}",
        )
        store.update_egress(
            proxy_session_id=session.id,
            ips=[ip],
            source="test",
        )
        store.mark_canary_blocked(
            session.id,
            error_message="Google unusual traffic",
            block_ips=[ip],
        )
    candidate = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user42",
        proxy_username="openai.user42",
    )
    store.update_egress(
        proxy_session_id=candidate.id,
        ips=["203.0.113.42"],
        source="test",
    )

    found = store.find_google_blocked_prefix_for_ip(
        "openai",
        "203.0.113.42",
        exclude_session_id=candidate.id,
    )

    assert found is not None
    assert found.prefix == "203.0.113.0/24"
    assert found.blocked_count == 3
    assert found.success_count == 0
    assert found.matched_session.proxy_username.startswith("openai.user")
    assert store.find_google_blocked_prefix_for_ip("JP", "203.0.113.42") is None


def test_find_google_blocked_prefix_for_ip_uses_google_block_observations(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    for index, ip in enumerate(
        ["198.51.100.10", "198.51.100.11", "198.51.100.12"],
        start=1,
    ):
        session = store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name=f"user{index}",
            proxy_username=f"openai.user{index}",
        )
        store.update_egress(
            proxy_session_id=session.id,
            ips=[f"203.0.113.{index}"],
            source="test",
        )
        store.mark_canary_blocked(
            session.id,
            error_message=f"Google unusual traffic: ip address: {ip}",
            block_ips=[ip],
        )
    candidate = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user42",
        proxy_username="openai.user42",
    )
    store.update_egress(
        proxy_session_id=candidate.id,
        ips=["198.51.100.42"],
        source="test",
    )

    found = store.find_google_blocked_prefix_for_ip(
        "openai",
        "198.51.100.42",
        exclude_session_id=candidate.id,
    )

    assert found is not None
    assert found.prefix == "198.51.100.0/24"
    assert found.blocked_count == 3
    assert found.success_count == 0


def test_find_google_blocked_prefix_for_ip_ignores_prefix_with_success(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    for index, ip in enumerate(
        ["203.0.113.10", "203.0.113.11", "203.0.113.12"],
        start=1,
    ):
        session = store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name=f"user{index}",
            proxy_username=f"openai.user{index}",
        )
        store.update_egress(
            proxy_session_id=session.id,
            ips=[ip],
            source="test",
        )
        store.mark_canary_blocked(
            session.id,
            error_message="Google unusual traffic",
            block_ips=[ip],
        )
    success = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user20",
        proxy_username="openai.user20",
    )
    store.update_egress(
        proxy_session_id=success.id,
        ips=["203.0.113.20"],
        source="test",
    )
    store.mark_canary_success(success.id)

    found = store.find_google_blocked_prefix_for_ip(
        "openai",
        "203.0.113.42",
        min_blocked_count=3,
    )

    assert found is None


def test_google_blocked_prefix_does_not_treat_duck_success_as_google_success(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    for index, ip in enumerate(
        ["203.0.113.10", "203.0.113.11", "203.0.113.12"],
        start=1,
    ):
        session = store.upsert_proxy_session(
            proxy_base_username="openai",
            session_name=f"user{index}",
            proxy_username=f"openai.user{index}",
        )
        store.update_egress(
            proxy_session_id=session.id,
            ips=[ip],
            source="test",
        )
        store.mark_canary_blocked(
            session.id,
            error_message="Google unusual traffic",
            block_ips=[ip],
        )
    duck_ok = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user20",
        proxy_username="openai.user20",
    )
    store.update_egress(
        proxy_session_id=duck_ok.id,
        ips=["203.0.113.20"],
        source="test",
    )
    store.mark_duck_canary_success(duck_ok.id)
    store.finish_request_success(duck_ok.id, engine="duck")

    found = store.find_google_blocked_prefix_for_ip(
        "openai",
        "203.0.113.42",
        min_blocked_count=3,
    )

    assert found is not None
    assert found.blocked_count == 3
    assert found.success_count == 0


def test_risk_metadata_does_not_filter_before_google_canary(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )

    snapshot = store.update_iplark_result(
        proxy_session_id=snapshot.id,
        quality_score=None,
        public_proxy=True,
        threat=True,
        min_quality_score=90,
    )

    assert snapshot.status == STATUS_RISK_CHECKED
    assert snapshot.cooldown_until is None
    assert snapshot.retire_reason is None
    assert snapshot.iplark_min_quality_score is None

    snapshot = store.update_iplark_result(
        proxy_session_id=snapshot.id,
        quality_score=10,
        min_quality_score=90,
    )

    assert snapshot.status == STATUS_RISK_CHECKED
    assert snapshot.cooldown_until is None
    assert snapshot.retire_reason is None
    assert snapshot.iplark_min_quality_score == 10


def test_upsert_preserves_existing_status_unless_explicitly_changed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
        status=STATUS_COOLDOWN,
    )

    snapshot = store.upsert_proxy_session(
        proxy_base_username="openai",
        session_name="user1",
        proxy_username="openai.user1",
    )

    assert snapshot.status == STATUS_COOLDOWN
