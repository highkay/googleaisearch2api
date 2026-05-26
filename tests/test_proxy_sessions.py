from __future__ import annotations

from pathlib import Path

import pytest

from googleaisearch2api.browser import resolve_browser_proxy
from googleaisearch2api.config import ServiceConfig
from googleaisearch2api.db import create_db_engine, create_session_factory, create_tables
from googleaisearch2api.proxy_sessions import (
    STATUS_ACTIVE,
    STATUS_COOLDOWN,
    STATUS_RETIRED,
    STATUS_RISK_CHECKED,
    ProxySessionConfigError,
    ProxySessionSelector,
    ProxySessionStore,
    build_proxy_config_for_session,
    format_sticky_username,
    google_block_has_ip_mismatch,
    hash_ip_vector,
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


def test_proxy_session_selector_rotates_active_sessions_before_success_bias(
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
