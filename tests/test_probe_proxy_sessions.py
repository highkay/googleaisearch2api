import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from googleaisearch2api.proxy_sessions import (
    STATUS_ACTIVE,
    STATUS_COOLDOWN,
    STATUS_RETIRED,
    STATUS_RISK_CHECKED,
    ProxyBlockedPrefixSnapshot,
    ProxySessionSnapshot,
)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "probe_proxy_sessions.py"
_SPEC = importlib.util.spec_from_file_location("probe_proxy_sessions", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)

_candidate_indices = _SCRIPT._candidate_indices
_canary_answer_matches = _SCRIPT._canary_answer_matches
_candidate_existing_sessions = _SCRIPT._candidate_existing_sessions
_fresh_active_checked_after = _SCRIPT._fresh_active_checked_after
_run_canary = _SCRIPT._run_canary
_run_duck_canary = _SCRIPT._run_duck_canary
_skip_candidate_reason = _SCRIPT._skip_candidate_reason
_skip_known_google_blocked_ip = _SCRIPT._skip_known_google_blocked_ip
_skip_known_google_blocked_prefix = _SCRIPT._skip_known_google_blocked_prefix
_should_stop_after_terminal_stage = _SCRIPT._should_stop_after_terminal_stage


def _snapshot(
    *,
    status: str,
    id: int = 1,
    proxy_username: str = "openai.user1",
    primary_ip: str | None = "203.0.113.10",
    retire_reason: str | None = None,
    cooldown_until: datetime | None = None,
    request_success_count: int = 0,
    request_block_count: int = 0,
    request_error_count: int = 0,
    canary_success_count: int = 0,
    canary_block_count: int = 0,
    duplicate_of_session_id: int | None = None,
    google_canary_status: str = "",
) -> ProxySessionSnapshot:
    return ProxySessionSnapshot(
        id=id,
        proxy_base_username="openai",
        session_name="user1",
        proxy_username=proxy_username,
        status=status,
        epoch=0,
        primary_ip=primary_ip,
        ip_vector_hash="hash",
        iplark_min_quality_score=34,
        google_canary_status=google_canary_status,
        google_canary_error=None,
        google_canary_checked_at=None,
        duck_canary_status="",
        duck_canary_error=None,
        duck_canary_checked_at=None,
        request_success_count=request_success_count,
        request_block_count=request_block_count,
        request_error_count=request_error_count,
        canary_success_count=canary_success_count,
        canary_block_count=canary_block_count,
        duck_canary_success_count=0,
        duck_canary_rate_limit_count=0,
        duck_canary_error_count=0,
        duplicate_of_session_id=duplicate_of_session_id,
        last_selected_at=None,
        last_success_at=None,
        last_blocked_at=None,
        cooldown_until=cooldown_until,
        retire_reason=retire_reason,
    )


def test_skip_candidate_preserves_active_and_cooldown_sessions() -> None:
    now = datetime(2026, 5, 26, tzinfo=UTC)
    active = _snapshot(status=STATUS_ACTIVE)
    cooldown = _snapshot(status=STATUS_COOLDOWN, cooldown_until=now + timedelta(hours=1))

    assert (
        _skip_candidate_reason(
            active,
            now=now,
            refresh_active=False,
            retry_cooldown=False,
            retry_retired=False,
            only_risk_retired=False,
        )
        == "active session preserved"
    )
    assert (
        _skip_candidate_reason(
            cooldown,
            now=now,
            refresh_active=False,
            retry_cooldown=False,
            retry_retired=False,
            only_risk_retired=False,
        )
        == "session is in cooldown"
    )
    assert (
        _skip_candidate_reason(
            active,
            now=now,
            refresh_active=True,
            retry_cooldown=False,
            retry_retired=False,
            only_risk_retired=False,
        )
        is None
    )


def test_skip_candidate_handles_sqlite_naive_cooldown_datetime() -> None:
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)
    active_cooldown = _snapshot(
        status=STATUS_COOLDOWN,
        cooldown_until=datetime(2026, 5, 26, 13, 0),
    )
    expired_cooldown = _snapshot(
        status=STATUS_COOLDOWN,
        cooldown_until=datetime(2026, 5, 26, 11, 0),
    )

    assert (
        _skip_candidate_reason(
            active_cooldown,
            now=now,
            refresh_active=False,
            retry_cooldown=False,
            retry_retired=False,
            only_risk_retired=False,
        )
        == "session is in cooldown"
    )
    assert (
        _skip_candidate_reason(
            expired_cooldown,
            now=now,
            refresh_active=False,
            retry_cooldown=False,
            retry_retired=False,
            only_risk_retired=False,
        )
        is None
    )


def test_skip_candidate_retries_legacy_risk_retired_only() -> None:
    now = datetime(2026, 5, 26, tzinfo=UTC)
    risk_retired = _snapshot(
        status=STATUS_RETIRED,
        retire_reason="iplark flagged public proxy/threat",
    )
    duplicate_retired = _snapshot(
        status=STATUS_RETIRED,
        retire_reason="duplicate egress with session 1",
    )

    assert (
        _skip_candidate_reason(
            risk_retired,
            now=now,
            refresh_active=False,
            retry_cooldown=False,
            retry_retired=False,
            only_risk_retired=False,
        )
        is None
    )
    assert (
        _skip_candidate_reason(
            duplicate_retired,
            now=now,
            refresh_active=False,
            retry_cooldown=False,
            retry_retired=False,
            only_risk_retired=False,
        )
        == "retired session skipped"
    )
    assert (
        _skip_candidate_reason(
            duplicate_retired,
            now=now,
            refresh_active=False,
            retry_cooldown=False,
            retry_retired=False,
            only_risk_retired=True,
        )
        == "not retired by legacy risk metadata gate"
    )


def test_candidate_indices_can_shuffle_with_stable_seed() -> None:
    assert _candidate_indices(1, 5, shuffle=False, seed=None) == [1, 2, 3, 4, 5]
    assert _candidate_indices(1, 5, shuffle=True, seed=7) == [5, 1, 4, 2, 3]


def test_fresh_active_checked_after_uses_seconds_window() -> None:
    now = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)

    assert _fresh_active_checked_after(now, 0) is None
    assert _fresh_active_checked_after(now, 300) == now - timedelta(seconds=300)


class _FakeExistingSessionStore:
    def __init__(self) -> None:
        self.snapshots = [
            _snapshot(status=STATUS_ACTIVE, id=1, proxy_username="openai.user1"),
            _snapshot(status=STATUS_COOLDOWN, id=2, proxy_username="openai.user2"),
            _snapshot(status=STATUS_RETIRED, id=3, proxy_username="openai.user3"),
        ]
        self.calls: list[tuple[int, str]] = []

    def list_proxy_sessions(
        self,
        *,
        limit: int,
        proxy_base_username: str | None = None,
    ) -> list[ProxySessionSnapshot]:
        self.calls.append((limit, proxy_base_username or ""))
        return self.snapshots[:limit]


def test_candidate_existing_sessions_reads_wider_store_pool_and_ranks() -> None:
    store = _FakeExistingSessionStore()

    result = _candidate_existing_sessions(
        store,
        "openai",
        limit=3,
        shuffle=True,
        seed=7,
    )

    assert store.calls == [(_SCRIPT.EXISTING_SESSION_RANKING_FETCH_LIMIT, "openai")]
    assert [item.id for item in result] == [1, 2, 3]


def test_candidate_existing_sessions_prefers_recoverable_success_history() -> None:
    store = _FakeExistingSessionStore()
    store.snapshots = [
        _snapshot(status=STATUS_COOLDOWN, id=10, request_error_count=8),
        _snapshot(status=STATUS_RETIRED, id=11, request_success_count=200),
        _snapshot(
            status=STATUS_COOLDOWN,
            id=12,
            request_success_count=20,
            request_block_count=1,
        ),
        _snapshot(status=STATUS_RISK_CHECKED, id=13),
        _snapshot(
            status=STATUS_COOLDOWN,
            id=14,
            request_success_count=100,
            duplicate_of_session_id=12,
        ),
    ]

    result = _candidate_existing_sessions(
        store,
        "openai",
        limit=5,
        shuffle=False,
        seed=None,
    )

    assert [item.id for item in result] == [13, 12, 10, 11, 14]


def test_candidate_existing_sessions_prefers_ready_over_active_cooldown() -> None:
    # Mirrors production DB state where high-history sessions still in cooldown
    # (user169/347/5700) monopolized recovery probes while ready sessions
    # (user122/user2055) with prior canary ok were never tried.
    now = datetime(2026, 7, 16, 16, 24, tzinfo=UTC)
    store = _FakeExistingSessionStore()
    store.snapshots = [
        _snapshot(
            status=STATUS_COOLDOWN,
            id=169,
            proxy_username="openai.user169",
            request_success_count=536,
            canary_success_count=40,
            request_block_count=96,
            request_error_count=60,
            canary_block_count=40,
            cooldown_until=now + timedelta(hours=23),
            google_canary_status="blocked",
        ),
        _snapshot(
            status=STATUS_COOLDOWN,
            id=122,
            proxy_username="openai.user122",
            request_success_count=104,
            canary_success_count=7,
            request_block_count=13,
            request_error_count=6,
            cooldown_until=now - timedelta(hours=1),
            google_canary_status="ok",
        ),
        _snapshot(
            status=STATUS_COOLDOWN,
            id=726,
            proxy_username="openai.user2055",
            request_success_count=64,
            canary_success_count=3,
            request_block_count=13,
            request_error_count=9,
            cooldown_until=now - timedelta(hours=1),
            google_canary_status="ok",
        ),
    ]

    result = _candidate_existing_sessions(
        store,
        "openai",
        limit=3,
        shuffle=False,
        seed=None,
        now=now,
    )

    assert [item.id for item in result] == [122, 726, 169]


def test_terminal_stage_gate_allows_direct_cooldown_canary_retry() -> None:
    cooldown = _snapshot(status=STATUS_COOLDOWN)

    assert not _should_stop_after_terminal_stage(cooldown, stage_ran=False)
    assert _should_stop_after_terminal_stage(cooldown, stage_ran=True)


class _FakeKnownBlockedStore:
    def __init__(self, blocked: ProxySessionSnapshot | None) -> None:
        self.blocked = blocked
        self.events: list[dict] = []
        self.cooldown_reasons: list[str] = []

    def find_google_blocked_session_for_ip(
        self,
        proxy_base_username: str,
        primary_ip: str,
        *,
        exclude_session_id: int | None = None,
    ) -> ProxySessionSnapshot | None:
        assert proxy_base_username == "openai"
        assert primary_ip == "203.0.113.10"
        assert exclude_session_id == 2
        return self.blocked

    def record_event(self, **kwargs: object) -> None:
        self.events.append(kwargs)

    def mark_session_cooldown(
        self,
        _proxy_session_id: int,
        *,
        reason: str,
    ) -> ProxySessionSnapshot:
        self.cooldown_reasons.append(reason)
        return _snapshot(status=STATUS_COOLDOWN, id=2, proxy_username="openai.user2")


def test_skip_known_google_blocked_ip_uses_own_google_failure_history() -> None:
    blocked = _snapshot(
        status=STATUS_COOLDOWN,
        id=1,
        proxy_username="openai.user1",
        primary_ip="203.0.113.10",
    )
    store = _FakeKnownBlockedStore(blocked)

    result = _skip_known_google_blocked_ip(
        store,
        _snapshot(status=STATUS_ACTIVE, id=2, proxy_username="openai.user2"),
        base_username="openai",
        enabled=True,
    )

    assert result.status == STATUS_COOLDOWN
    assert "openai.user1" in store.cooldown_reasons[0]
    assert store.events[0]["event_type"] == "known_google_blocked_ip_skipped"


def test_skip_known_google_blocked_ip_handles_retired_duplicate_candidate() -> None:
    blocked = _snapshot(
        status=STATUS_COOLDOWN,
        id=1,
        proxy_username="openai.user1",
        primary_ip="203.0.113.10",
    )
    store = _FakeKnownBlockedStore(blocked)

    result = _skip_known_google_blocked_ip(
        store,
        _snapshot(status=STATUS_RETIRED, id=2, proxy_username="openai.user2"),
        base_username="openai",
        enabled=True,
    )

    assert result.status == STATUS_COOLDOWN
    assert store.events[0]["event_type"] == "known_google_blocked_ip_skipped"


def test_skip_known_google_blocked_ip_can_be_disabled() -> None:
    store = _FakeKnownBlockedStore(
        _snapshot(status=STATUS_COOLDOWN, id=1, proxy_username="openai.user1")
    )
    snapshot = _snapshot(status=STATUS_ACTIVE, id=2, proxy_username="openai.user2")

    result = _skip_known_google_blocked_ip(
        store,
        snapshot,
        base_username="openai",
        enabled=False,
    )

    assert result is snapshot
    assert store.events == []
    assert store.cooldown_reasons == []


class _FakeKnownBlockedPrefixStore:
    def __init__(self, blocked_prefix: ProxyBlockedPrefixSnapshot | None) -> None:
        self.blocked_prefix = blocked_prefix
        self.events: list[dict] = []
        self.cooldown_reasons: list[str] = []

    def find_google_blocked_prefix_for_ip(
        self,
        proxy_base_username: str,
        primary_ip: str,
        *,
        min_blocked_count: int = 1,
        exclude_session_id: int | None = None,
    ) -> ProxyBlockedPrefixSnapshot | None:
        assert proxy_base_username == "openai"
        assert primary_ip == "203.0.113.42"
        assert min_blocked_count == 1
        assert exclude_session_id == 4
        return self.blocked_prefix

    def record_event(self, **kwargs: object) -> None:
        self.events.append(kwargs)

    def mark_session_cooldown(
        self,
        _proxy_session_id: int,
        *,
        reason: str,
    ) -> ProxySessionSnapshot:
        self.cooldown_reasons.append(reason)
        return _snapshot(
            status=STATUS_COOLDOWN,
            id=4,
            proxy_username="openai.user4",
            primary_ip="203.0.113.42",
        )


def test_skip_known_google_blocked_prefix_uses_failed_prefix_without_success() -> None:
    blocked_prefix = ProxyBlockedPrefixSnapshot(
        prefix="203.0.113.0/24",
        blocked_count=3,
        success_count=0,
        matched_session=_snapshot(
            status=STATUS_COOLDOWN,
            id=1,
            proxy_username="openai.user1",
            primary_ip="203.0.113.10",
        ),
    )
    store = _FakeKnownBlockedPrefixStore(blocked_prefix)

    result = _skip_known_google_blocked_prefix(
        store,
        _snapshot(
            status=STATUS_ACTIVE,
            id=4,
            proxy_username="openai.user4",
            primary_ip="203.0.113.42",
        ),
        base_username="openai",
        enabled=True,
        min_blocked_count=1,
    )

    assert result.status == STATUS_COOLDOWN
    assert "203.0.113.0/24" in store.cooldown_reasons[0]
    assert store.events[0]["event_type"] == "known_google_blocked_prefix_skipped"
    assert store.events[0]["raw_json"]["blocked_count"] == 3


def test_skip_known_google_blocked_prefix_can_be_disabled() -> None:
    store = _FakeKnownBlockedPrefixStore(
        ProxyBlockedPrefixSnapshot(
            prefix="203.0.113.0/24",
            blocked_count=3,
            success_count=0,
            matched_session=_snapshot(status=STATUS_COOLDOWN),
        )
    )
    snapshot = _snapshot(
        status=STATUS_ACTIVE,
        id=4,
        proxy_username="openai.user4",
        primary_ip="203.0.113.42",
    )

    result = _skip_known_google_blocked_prefix(
        store,
        snapshot,
        base_username="openai",
        enabled=False,
        min_blocked_count=3,
    )

    assert result is snapshot
    assert store.events == []
    assert store.cooldown_reasons == []


def test_canary_answer_match_normalizes_trailing_period_and_whitespace() -> None:
    assert _canary_answer_matches("  42.\n", "42")
    assert _canary_answer_matches("`42`", "42")
    assert _canary_answer_matches(
        "42\nIf you have any other math problems, feel free to ask!",
        "42",
    )
    assert not _canary_answer_matches("420", "42")
    assert not _canary_answer_matches("42abc", "42")
    assert not _canary_answer_matches(
        "You said: What is nineteen plus twenty-three?",
        "42",
    )
    assert _canary_answer_matches("any non-blocked answer", "")


class _FakeCanaryRunner:
    def __init__(self, answers: list[object]) -> None:
        self._answers = answers
        self.prompts: list[str] = []
        self.closed = False

    def run_prompt(self, _config: object, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        answer_text = str(answer)
        return SimpleNamespace(
            answer_text=answer_text,
            final_url="https://www.google.com/search?udm=50",
            page_title="Google AI",
            body_excerpt=answer_text,
        )

    def close(self) -> None:
        self.closed = True


class _FakeCanaryStore:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.cooldown_reasons: list[str] = []
        self.success_count = 0
        self.duck_success_count = 0
        self.duck_rate_limit_errors: list[str] = []
        self.duck_errors: list[str] = []

    def record_event(self, **kwargs: object) -> None:
        self.events.append(kwargs)

    def mark_canary_success(self, _proxy_session_id: int) -> ProxySessionSnapshot:
        self.success_count += 1
        return _snapshot(status=STATUS_ACTIVE)

    def mark_session_cooldown(
        self,
        _proxy_session_id: int,
        *,
        reason: str,
    ) -> ProxySessionSnapshot:
        self.cooldown_reasons.append(reason)
        return _snapshot(status=STATUS_COOLDOWN)

    def mark_duck_canary_success(self, _proxy_session_id: int) -> ProxySessionSnapshot:
        self.duck_success_count += 1
        return _snapshot(status=STATUS_ACTIVE)

    def mark_duck_canary_rate_limited(
        self,
        _proxy_session_id: int,
        *,
        error_message: str,
    ) -> ProxySessionSnapshot:
        self.duck_rate_limit_errors.append(error_message)
        return _snapshot(status=STATUS_ACTIVE)

    def mark_duck_canary_error(
        self,
        _proxy_session_id: int,
        *,
        error_message: str,
    ) -> ProxySessionSnapshot:
        self.duck_errors.append(error_message)
        return _snapshot(status=STATUS_ACTIVE)


def test_run_canary_requires_all_repeated_expected_answers(monkeypatch) -> None:
    runner = _FakeCanaryRunner(["42", "42."])
    monkeypatch.setattr(_SCRIPT, "GoogleAiRunner", lambda: runner)
    store = _FakeCanaryStore()

    result = _run_canary(
        store,
        _snapshot(status=STATUS_ACTIVE),
        object(),
        "What is nineteen plus twenty-three? Reply with only the number.",
        "42",
        2,
    )

    assert result.status == STATUS_ACTIVE
    assert store.success_count == 1
    assert len(runner.prompts) == 2
    assert runner.closed


def test_run_canary_rejects_prompt_echo_before_activation(monkeypatch) -> None:
    runner = _FakeCanaryRunner(["You said: What is nineteen plus twenty-three?"])
    monkeypatch.setattr(_SCRIPT, "GoogleAiRunner", lambda: runner)
    store = _FakeCanaryStore()

    result = _run_canary(
        store,
        _snapshot(status=STATUS_ACTIVE),
        object(),
        "What is nineteen plus twenty-three? Reply with only the number.",
        "42",
        2,
    )

    assert result.status == STATUS_COOLDOWN
    assert store.success_count == 0
    assert len(runner.prompts) == 1
    assert "unexpected answer" in store.cooldown_reasons[0]
    assert runner.closed


def test_run_duck_canary_records_duck_success_only(monkeypatch) -> None:
    runner = _FakeCanaryRunner(["42"])
    monkeypatch.setattr(_SCRIPT, "DuckAiRunner", lambda: runner)
    store = _FakeCanaryStore()

    _run_duck_canary(
        store,
        _snapshot(status=STATUS_COOLDOWN),
        object(),
        "What is nineteen plus twenty-three? Reply with only the number.",
        "42",
        1,
    )

    assert store.duck_success_count == 1
    assert store.success_count == 0
    assert store.cooldown_reasons == []
    assert store.events[0]["event_type"] == "duck_canary_success"
    assert runner.closed


def test_run_duck_canary_records_rate_limit_without_google_cooldown(monkeypatch) -> None:
    runner = _FakeCanaryRunner([_SCRIPT.DuckAiRateLimitedError("duck limited")])
    monkeypatch.setattr(_SCRIPT, "DuckAiRunner", lambda: runner)
    store = _FakeCanaryStore()

    _run_duck_canary(
        store,
        _snapshot(status=STATUS_COOLDOWN),
        object(),
        "What is nineteen plus twenty-three? Reply with only the number.",
        "42",
        1,
    )

    assert store.duck_rate_limit_errors == ["duck limited"]
    assert store.success_count == 0
    assert store.cooldown_reasons == []
    assert store.events[0]["event_type"] == "duck_canary_rate_limited"
    assert runner.closed


def test_run_duck_canary_records_unexpected_answer_without_google_cooldown(
    monkeypatch,
) -> None:
    runner = _FakeCanaryRunner(["You said: What is nineteen plus twenty-three?"])
    monkeypatch.setattr(_SCRIPT, "DuckAiRunner", lambda: runner)
    store = _FakeCanaryStore()

    _run_duck_canary(
        store,
        _snapshot(status=STATUS_COOLDOWN),
        object(),
        "What is nineteen plus twenty-three? Reply with only the number.",
        "42",
        1,
    )

    assert len(store.duck_errors) == 1
    assert "unexpected answer" in store.duck_errors[0]
    assert store.success_count == 0
    assert store.cooldown_reasons == []
    assert store.events[0]["event_type"] == "duck_canary_unexpected_answer"
    assert runner.closed
