from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest

from googleaisearch2api.browser import GoogleAiBlockedError, GoogleAiUnavailableError
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
