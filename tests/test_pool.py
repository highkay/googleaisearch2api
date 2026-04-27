from __future__ import annotations

import threading
import time

import pytest

from googleaisearch2api.config import ServiceConfig
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
        threads = [
            threading.Thread(target=lambda prompt=prompt: pool.execute(ServiceConfig(), prompt))
            for prompt in ("one", "two", "three")
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
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
    )
    config = ServiceConfig(browser_timeout_ms=100, answer_timeout_ms=100)
    try:
        with pytest.raises(BrowserPoolTimeoutError):
            pool.execute(config, "stuck")
        assert pool.get_summary().generation == 1
    finally:
        release.set()
        pool.close()
