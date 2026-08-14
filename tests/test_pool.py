from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from googleaisearch2api.browser import GoogleAiBlockedError, GoogleAiUnavailableError
from googleaisearch2api.browser_gate import BrowserResourceGate
from googleaisearch2api.config import ServiceConfig
from googleaisearch2api.hybrid_runner import HybridGoogleAiRunner
from googleaisearch2api.pool import (
    BrowserPool,
    BrowserPoolSaturatedError,
    BrowserPoolTimeoutError,
)
from googleaisearch2api.schemas import GoogleAiResult


def _result(prompt: str) -> GoogleAiResult:
    return GoogleAiResult(
        answer_text=f"answer for {prompt}",
        final_url="https://www.google.com/search?udm=50",
        page_title="Google Search",
    )


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition not met within timeout")


def test_browser_pool_runs_one_runner_per_worker_concurrently() -> None:
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    class FakeRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            try:
                time.sleep(0.1)
                return _result(prompt)
            finally:
                with lock:
                    state["active"] -= 1

        def close(self) -> None:
            pass

    pool = BrowserPool(worker_count=2, queue_capacity=2, runner_factory=FakeRunner)
    try:
        outputs: list[GoogleAiResult] = []
        threads = [
            threading.Thread(
                target=lambda prompt=prompt: outputs.append(pool.execute(ServiceConfig(), prompt))
            )
            for prompt in ("one", "two")
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert len(outputs) == 2
        assert state["max_active"] == 2
        assert pool.get_summary().busy_workers == 0
    finally:
        pool.close()


def test_browser_pool_queues_until_a_worker_is_released() -> None:
    starts: list[tuple[str, float]] = []
    lock = threading.Lock()
    t0 = time.perf_counter()

    class FakeRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            with lock:
                starts.append((prompt, time.perf_counter() - t0))
            time.sleep(0.1)
            return _result(prompt)

        def close(self) -> None:
            pass

    pool = BrowserPool(worker_count=2, queue_capacity=2, runner_factory=FakeRunner)
    try:
        # Start the first two jobs first so both workers are occupied before the third
        # is enqueued. Launching all three at once can race put_nowait against an empty
        # queue and spuriously raise BrowserPoolSaturatedError.
        first = threading.Thread(target=lambda: pool.execute(ServiceConfig(), "one"))
        second = threading.Thread(target=lambda: pool.execute(ServiceConfig(), "two"))
        first.start()
        second.start()

        deadline = time.monotonic() + 5
        while len(starts) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(starts) == 2

        third = threading.Thread(target=lambda: pool.execute(ServiceConfig(), "three"))
        third.start()
        for thread in (first, second, third):
            thread.join(timeout=5)

        by_prompt = {prompt: started_at for prompt, started_at in starts}
        assert by_prompt["one"] < 0.08
        assert by_prompt["two"] < 0.08
        assert by_prompt["three"] >= 0.08
    finally:
        pool.close()


def test_browser_pool_returns_saturated_when_workers_and_queue_are_full() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            started.set()
            release.wait(timeout=5)
            return _result(prompt)

        def close(self) -> None:
            release.set()

    pool = BrowserPool(worker_count=1, queue_capacity=1, runner_factory=BlockingRunner)
    try:
        first = threading.Thread(target=lambda: pool.execute(ServiceConfig(), "first"))
        second = threading.Thread(target=lambda: pool.execute(ServiceConfig(), "second"))

        first.start()
        assert started.wait(timeout=5)
        second.start()

        deadline = time.monotonic() + 5
        while pool.get_summary().queued_requests < 1 and time.monotonic() < deadline:
            time.sleep(0.01)

        with pytest.raises(BrowserPoolSaturatedError):
            pool.execute(ServiceConfig(), "third")

        release.set()
        first.join(timeout=5)
        second.join(timeout=5)
    finally:
        pool.close()


def test_browser_pool_reset_marks_next_generation() -> None:
    class FakeRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            return _result(prompt)

        def close(self) -> None:
            pass

    pool = BrowserPool(worker_count=1, queue_capacity=1, runner_factory=FakeRunner)
    try:
        pool.reset()
        assert pool.get_summary().generation == 1
    finally:
        pool.close()


def test_browser_pool_times_out_and_recycles_stuck_work() -> None:
    release = threading.Event()

    class BlockingRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            release.wait(timeout=5)
            return _result(prompt)

        def close(self) -> None:
            release.set()

    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=BlockingRunner,
        request_timeout_buffer_ms=50,
        request_timeout_override_s=0.55,
    )
    config = ServiceConfig(browser_timeout_ms=100, answer_timeout_ms=100)
    try:
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(config, "stuck")
        assert pool.get_summary().generation == 0
        release.set()
        result = pool.execute(config, "again")
        assert result.answer_text == "answer for again"
    finally:
        release.set()
        pool.close()


def test_timeout_poisons_only_the_involved_worker() -> None:
    release = threading.Event()

    class TrackedRunner:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompt_calls += 1
            release.wait(timeout=5)
            return _result(prompt)

        def close(self) -> None:
            self.close_calls += 1

    fakes: list[TrackedRunner] = []

    def factory() -> TrackedRunner:
        fake = TrackedRunner()
        fakes.append(fake)
        return fake

    pool = BrowserPool(
        worker_count=2,
        queue_capacity=2,
        runner_factory=factory,
        worker_poll_interval_s=0.05,
        request_timeout_override_s=0.55,
    )
    try:
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(ServiceConfig(), "stuck")

        _wait_until(lambda: sum(fake.prompt_calls for fake in fakes) == 1)
        involved = next(fake for fake in fakes if fake.prompt_calls == 1)
        uninvolved = next(fake for fake in fakes if fake.prompt_calls == 0)

        assert involved.close_calls == 0
        assert uninvolved.close_calls == 0

        release.set()
        _wait_until(lambda: involved.close_calls >= 1)
        assert uninvolved.close_calls == 0
    finally:
        release.set()
        pool.close()


def test_timeout_worker_serves_again_after_release() -> None:
    release = threading.Event()

    class BlockingRunner:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompt_calls += 1
            release.wait(timeout=5)
            return _result(prompt)

        def close(self) -> None:
            self.close_calls += 1

    runner = BlockingRunner()
    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=lambda: runner,
        worker_poll_interval_s=0.05,
        request_timeout_override_s=0.55,
    )
    try:
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(ServiceConfig(), "stuck")

        release.set()
        result = pool.execute(ServiceConfig(), "recovered")

        assert result.answer_text == "answer for recovered"
        assert runner.prompt_calls == 2
        assert runner.close_calls == 1
    finally:
        release.set()
        pool.close()


def test_pool_recycles_runner_after_unavailable_error() -> None:
    class UnavailableRunner:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompt_calls += 1
            raise GoogleAiUnavailableError(
                "Google AI Mode is not available for this browser session."
            )

        def close(self) -> None:
            self.close_calls += 1

    runner = UnavailableRunner()
    pool = BrowserPool(worker_count=1, queue_capacity=1, runner_factory=lambda: runner)
    try:
        with pytest.raises(GoogleAiUnavailableError):
            pool.execute(ServiceConfig(), "nope")

        assert runner.prompt_calls == 1
        assert runner.close_calls == 1
    finally:
        pool.close()


def test_browser_pool_recycles_and_retries_blocked_sessions() -> None:
    class BlockedOnceRunner:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                raise GoogleAiBlockedError(
                    "Google blocked the session while opening query page: "
                    "this network is blocked due to unaddressed abuse complaints"
                )
            return _result(prompt)

        def close(self) -> None:
            self.close_calls += 1

    runner = BlockedOnceRunner()
    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=lambda: runner,
        blocked_retry_count=1,
    )
    try:
        result = pool.execute(ServiceConfig(), "retry me")

        assert result.answer_text == "answer for retry me"
        assert runner.prompts == ["retry me", "retry me"]
        assert runner.close_calls == 1
    finally:
        pool.close()


def test_browser_pool_can_disable_blocked_retry_for_one_request() -> None:
    class BlockedRunner:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompt_calls += 1
            raise GoogleAiBlockedError("Google blocked the session while opening query page")

        def close(self) -> None:
            self.close_calls += 1

    runner = BlockedRunner()
    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=lambda: runner,
        blocked_retry_count=1,
    )
    try:
        with pytest.raises(GoogleAiBlockedError):
            pool.execute(ServiceConfig(), "do not retry", blocked_retry_count=0)

        assert runner.prompt_calls == 1
        assert runner.close_calls == 1
    finally:
        pool.close()


def test_browser_pool_does_not_retry_blocked_sessions_by_default() -> None:
    class BlockedRunner:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompt_calls += 1
            raise GoogleAiBlockedError("Google blocked the session while opening query page")

        def close(self) -> None:
            self.close_calls += 1

    runner = BlockedRunner()
    pool = BrowserPool(worker_count=1, queue_capacity=1, runner_factory=lambda: runner)
    try:
        with pytest.raises(GoogleAiBlockedError):
            pool.execute(ServiceConfig(), "do not retry")

        assert runner.prompt_calls == 1
        assert runner.close_calls == 1
    finally:
        pool.close()


def test_hybrid_runner_slots_into_browser_pool() -> None:
    class FakeHybridRunner(HybridGoogleAiRunner):
        def __init__(self) -> None:
            super().__init__()
            self.prompts: list[str] = []

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompts.append(prompt)
            return _result(prompt)

        def close(self) -> None:
            pass

    runner = FakeHybridRunner()
    pool = BrowserPool(worker_count=1, queue_capacity=1, runner_factory=lambda: runner)
    try:
        result = pool.execute(ServiceConfig(ai_mode_http_enabled=True), "hybrid prompt")

        assert result.answer_text == "answer for hybrid prompt"
        assert runner.prompts == ["hybrid prompt"]
    finally:
        pool.close()


def test_watchdog_abandons_wedged_worker_and_respawns() -> None:
    wedge = threading.Event()  # set at teardown only, so runner 0 stays wedged
    runners: list[object] = []

    class WedgeOnceRunner:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompt_calls += 1
            if runners and runners[0] is self:
                wedge.wait(timeout=10)
            return _result(prompt)

        def close(self) -> None:
            self.close_calls += 1

    def factory() -> WedgeOnceRunner:
        runner = WedgeOnceRunner()
        runners.append(runner)
        return runner

    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=factory,
        worker_hard_timeout_s=0.5,
        watchdog_poll_interval_s=0.1,
        request_timeout_override_s=0.2,
    )
    try:
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(ServiceConfig(), "stuck")

        _wait_until(lambda: len(runners) == 2 and pool.get_summary().abandoned_workers == 1)

        result = pool.execute(ServiceConfig(), "fresh")
        assert result.answer_text == "answer for fresh"
        assert runners[1].prompt_calls == 1
    finally:
        wedge.set()
        pool.close()


def test_watchdog_releases_abandoned_workers_gate_slot() -> None:
    wedge = threading.Event()
    gate = BrowserResourceGate()
    runners: list[object] = []

    class WedgeOnceRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            if runners and runners[0] is self:
                wedge.wait(timeout=10)
            return _result(prompt)

        def close(self) -> None:
            pass

    def factory() -> WedgeOnceRunner:
        runner = WedgeOnceRunner()
        runners.append(runner)
        return runner

    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=factory,
        browser_gate=gate,
        worker_hard_timeout_s=0.5,
        watchdog_poll_interval_s=0.1,
        request_timeout_override_s=0.2,
    )
    try:
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(ServiceConfig(), "stuck")

        _wait_until(lambda: gate.status()["shared_holders"] == 0)
        assert gate.acquire_exclusive("recovery", timeout_s=0.3) is True
        gate.release_exclusive("recovery")
    finally:
        wedge.set()
        pool.close()


def test_watchdog_does_not_respawn_slow_but_alive_worker() -> None:
    class SlowOnceRunner:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompt_calls += 1
            if self.prompt_calls == 1:
                # Slow but alive: returns after the request timeout (0.2s) yet
                # well before the hard deadline (2s), so poison-recycle, not
                # respawn, must handle it.
                time.sleep(0.4)
            return _result(prompt)

        def close(self) -> None:
            self.close_calls += 1

    runners: list[SlowOnceRunner] = []

    def factory() -> SlowOnceRunner:
        runner = SlowOnceRunner()
        runners.append(runner)
        return runner

    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=factory,
        worker_hard_timeout_s=2.0,
        watchdog_poll_interval_s=0.05,
        request_timeout_override_s=0.2,
    )
    try:
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(ServiceConfig(), "slow")

        assert pool.get_summary().abandoned_workers == 0
        assert len(runners) == 1

        _wait_until(lambda: runners[0].close_calls == 1)
        assert pool.get_summary().abandoned_workers == 0
        assert len(runners) == 1

        result = pool.execute(ServiceConfig(), "again")
        assert result.answer_text == "answer for again"
        assert pool.get_summary().abandoned_workers == 0
        assert len(runners) == 1
        assert runners[0].close_calls == 1
    finally:
        pool.close()


def test_stale_slot_state_does_not_corrupt_replacement() -> None:
    release_old = threading.Event()
    release_new = threading.Event()
    runners: list[object] = []

    class StaleStateRunner:
        def __init__(self) -> None:
            self.prompt_calls = 0
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            self.prompt_calls += 1
            gate_event = release_old if runners and runners[0] is self else release_new
            gate_event.wait(timeout=10)
            return _result(prompt)

        def close(self) -> None:
            self.close_calls += 1

    def factory() -> StaleStateRunner:
        runner = StaleStateRunner()
        runners.append(runner)
        return runner

    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=factory,
        watchdog_poll_interval_s=0.05,
    )
    try:
        wedged_config = ServiceConfig(browser_worker_hard_timeout_seconds=0.5)
        fresh_config = ServiceConfig(browser_worker_hard_timeout_seconds=60.0)
        stuck_results: list[GoogleAiResult] = []
        fresh_results: list[GoogleAiResult] = []

        stuck_thread = threading.Thread(
            target=lambda: stuck_results.append(pool.execute(wedged_config, "stuck"))
        )
        stuck_thread.start()
        _wait_until(lambda: len(runners) == 2 and pool.get_summary().abandoned_workers == 1)

        fresh_thread = threading.Thread(
            target=lambda: fresh_results.append(pool.execute(fresh_config, "fresh"))
        )
        fresh_thread.start()
        _wait_until(lambda: runners[1].prompt_calls == 1)
        assert pool.get_summary().busy_workers == 1

        # The abandoned thread returns late while the replacement is busy; its
        # stale not-busy write must not clear the replacement's busy state.
        release_old.set()
        _wait_until(lambda: runners[0].close_calls == 1)
        assert pool.get_summary().busy_workers == 1
        assert pool.get_summary().abandoned_workers == 1

        release_new.set()
        fresh_thread.join(timeout=5)
        stuck_thread.join(timeout=5)
        assert [result.answer_text for result in fresh_results] == ["answer for fresh"]
        assert [result.answer_text for result in stuck_results] == ["answer for stuck"]

        summary = pool.get_summary()
        assert summary.busy_workers == 0
        assert summary.abandoned_workers == 1
    finally:
        release_old.set()
        release_new.set()
        pool.close()


def test_close_with_orphaned_worker_returns() -> None:
    wedge = threading.Event()

    class WedgedRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            wedge.wait(timeout=10)
            return _result(prompt)

        def close(self) -> None:
            pass

    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=WedgedRunner,
        worker_hard_timeout_s=0.5,
        watchdog_poll_interval_s=0.05,
        request_timeout_override_s=0.2,
    )
    with pytest.raises(BrowserPoolTimeoutError):
        pool.execute(ServiceConfig(), "stuck")
    _wait_until(lambda: pool.get_summary().abandoned_workers == 1)

    started_at = time.monotonic()
    pool.close()
    elapsed_s = time.monotonic() - started_at
    wedge.set()  # free the orphaned daemon thread for a clean interpreter exit

    assert elapsed_s < 5.0


class _SpyingGate(BrowserResourceGate):
    def __init__(self) -> None:
        super().__init__()
        self._spy_lock = threading.Lock()
        self.acquired_holders: list[str] = []

    def try_acquire_shared(self, holder: str = "worker") -> bool:
        with self._spy_lock:
            self.acquired_holders.append(holder)
        return super().try_acquire_shared(holder)


def test_respawned_worker_uses_new_generation_holder() -> None:
    wedge = threading.Event()
    gate = _SpyingGate()
    runners: list[object] = []

    class WedgeOnceRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            if runners and runners[0] is self:
                wedge.wait(timeout=10)
            return _result(prompt)

        def close(self) -> None:
            pass

    def factory() -> WedgeOnceRunner:
        runner = WedgeOnceRunner()
        runners.append(runner)
        return runner

    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=factory,
        browser_gate=gate,
        worker_hard_timeout_s=0.5,
        watchdog_poll_interval_s=0.05,
        request_timeout_override_s=0.2,
    )
    try:
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(ServiceConfig(), "stuck")
        _wait_until(lambda: pool.get_summary().abandoned_workers == 1)

        result = pool.execute(ServiceConfig(), "fresh")
        assert result.answer_text == "answer for fresh"
        _wait_until(lambda: len(set(gate.acquired_holders)) >= 2)

        assert "browser-worker-1-g0" in gate.acquired_holders
        assert "browser-worker-1-g1" in gate.acquired_holders
    finally:
        wedge.set()
        pool.close()


def test_summary_reports_poisoned_and_abandoned() -> None:
    wedge_long = threading.Event()
    wedge_short = threading.Event()
    runners: list[object] = []

    class PromptBlockingRunner:
        def __init__(self) -> None:
            self.close_calls = 0

        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            gate_event = wedge_long if prompt == "stuck-long" else wedge_short
            gate_event.wait(timeout=10)
            return _result(prompt)

        def close(self) -> None:
            self.close_calls += 1

    def factory() -> PromptBlockingRunner:
        runner = PromptBlockingRunner()
        runners.append(runner)
        return runner

    pool = BrowserPool(
        worker_count=2,
        queue_capacity=2,
        runner_factory=factory,
        watchdog_poll_interval_s=0.05,
        request_timeout_override_s=0.2,
    )
    try:
        long_config = ServiceConfig(browser_worker_hard_timeout_seconds=10.0)
        short_config = ServiceConfig(browser_worker_hard_timeout_seconds=0.5)

        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(long_config, "stuck-long")
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(short_config, "stuck-short")

        _wait_until(lambda: pool.get_summary().abandoned_workers == 1)

        summary = pool.get_summary()
        assert summary.poisoned_workers == 1
        assert summary.abandoned_workers == 1
        assert len(runners) == 3  # 2 initial + 1 replacement
    finally:
        wedge_long.set()
        wedge_short.set()
        pool.close()


def test_watchdog_respawns_crashed_worker() -> None:
    calls = {"count": 0}

    class HealthyRunner:
        def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
            return _result(prompt)

        def close(self) -> None:
            return None

    def factory() -> HealthyRunner:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("browser launch failed")
        return HealthyRunner()

    pool = BrowserPool(
        worker_count=1,
        queue_capacity=1,
        runner_factory=factory,
        watchdog_poll_interval_s=0.05,
    )
    try:
        # First factory call (during __init__) raises; the watchdog must detect
        # the dead worker and respawn a healthy replacement.
        _wait_until(lambda: calls["count"] >= 2)
        result = pool.execute(ServiceConfig(), "after-crash")
        assert result.answer_text == "answer for after-crash"
        assert pool.get_summary().abandoned_workers == 1
    finally:
        pool.close()
