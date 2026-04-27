from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from .browser import GoogleAiRunner
from .config import ServiceConfig
from .schemas import GoogleAiResult, RuntimePoolSummary


class RunnerProtocol(Protocol):
    def run_prompt(self, config: ServiceConfig, prompt: str) -> GoogleAiResult: ...

    def close(self) -> None: ...


class BrowserPoolError(RuntimeError):
    pass


class BrowserPoolClosedError(BrowserPoolError):
    pass


class BrowserPoolSaturatedError(BrowserPoolError):
    pass


class BrowserPoolTimeoutError(BrowserPoolError):
    pass


@dataclass(slots=True)
class _PoolJob:
    config: ServiceConfig
    prompt: str
    future: Future


_STOP = object()


class BrowserPool:
    def __init__(
        self,
        *,
        worker_count: int,
        queue_capacity: int,
        runner_factory: Callable[[], RunnerProtocol] = GoogleAiRunner,
        worker_poll_interval_s: float = 0.25,
        request_timeout_buffer_ms: int = 5_000,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1")
        if request_timeout_buffer_ms < 0:
            raise ValueError("request_timeout_buffer_ms must be at least 0")

        self._worker_count = worker_count
        self._runner_factory = runner_factory
        self._worker_poll_interval_s = worker_poll_interval_s
        self._request_timeout_buffer_ms = request_timeout_buffer_ms
        self._jobs: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._lock = threading.Lock()
        self._closed = False
        self._generation = 0
        self._busy_workers: set[int] = set()
        self._worker_errors: dict[int, str | None] = {index: None for index in range(worker_count)}
        self._workers: list[threading.Thread] = []

        for index in range(worker_count):
            worker = threading.Thread(
                target=self._worker_main,
                name=f"browser-worker-{index + 1}",
                args=(index,),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def execute(self, config: ServiceConfig, prompt: str) -> GoogleAiResult:
        future: Future = Future()
        job = _PoolJob(
            config=config.model_copy(deep=True),
            prompt=prompt,
            future=future,
        )

        with self._lock:
            if self._closed:
                raise BrowserPoolClosedError("Browser pool is closed.")

        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            raise BrowserPoolSaturatedError(
                f"Browser pool is saturated: {self._worker_count} workers and "
                f"{self._jobs.maxsize} queued requests are already occupied."
            ) from exc

        timeout_s = self._resolve_request_timeout_s(config)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            self.reset()
            raise BrowserPoolTimeoutError(
                "Browser worker exceeded the request wait timeout and was scheduled for recycle."
            ) from exc

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._generation += 1
            generation = self._generation
        logger.info("Scheduled browser worker reset for generation {}", generation)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        while True:
            try:
                item = self._jobs.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                self._jobs.task_done()
                continue
            job = item
            if not job.future.done():
                job.future.set_exception(BrowserPoolClosedError("Browser pool is shutting down."))
            self._jobs.task_done()

        for _ in self._workers:
            self._jobs.put(_STOP)

        for worker in self._workers:
            worker.join(timeout=30)

    def get_summary(self) -> RuntimePoolSummary:
        with self._lock:
            busy_workers = len(self._busy_workers)
            closed = self._closed
            generation = self._generation
            worker_errors = sum(1 for value in self._worker_errors.values() if value)

        return RuntimePoolSummary(
            worker_count=self._worker_count,
            busy_workers=busy_workers,
            idle_workers=self._worker_count - busy_workers,
            queued_requests=self._jobs.qsize(),
            queue_capacity=self._jobs.maxsize,
            accepting_requests=not closed,
            generation=generation,
            workers_with_errors=worker_errors,
        )

    def _worker_main(self, worker_index: int) -> None:
        runner = self._runner_factory()
        generation_seen = 0

        try:
            while True:
                try:
                    item = self._jobs.get(timeout=self._worker_poll_interval_s)
                except queue.Empty:
                    generation_seen = self._sync_generation(worker_index, runner, generation_seen)
                    continue

                if item is _STOP:
                    self._jobs.task_done()
                    break

                job = item
                generation_seen = self._sync_generation(worker_index, runner, generation_seen)
                if job.future.cancelled():
                    self._jobs.task_done()
                    continue
                self._mark_worker_busy(worker_index, busy=True)
                try:
                    result = runner.run_prompt(job.config, job.prompt)
                except Exception as exc:
                    self._set_worker_error(worker_index, repr(exc))
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    self._set_worker_error(worker_index, None)
                    if not job.future.done():
                        job.future.set_result(result)
                finally:
                    self._mark_worker_busy(worker_index, busy=False)
                    self._jobs.task_done()
        except Exception:
            logger.exception("Browser pool worker {} crashed", worker_index + 1)
            self._set_worker_error(worker_index, "worker crashed")
        finally:
            try:
                runner.close()
            except Exception:
                logger.exception("Failed to close browser runner for worker {}", worker_index + 1)

    def _sync_generation(
        self,
        worker_index: int,
        runner: RunnerProtocol,
        generation_seen: int,
    ) -> int:
        with self._lock:
            generation = self._generation

        if generation == generation_seen:
            return generation_seen

        try:
            runner.close()
        except Exception:
            logger.exception(
                "Failed to recycle browser runner for worker {} during generation sync",
                worker_index + 1,
            )
        return generation

    def _mark_worker_busy(self, worker_index: int, *, busy: bool) -> None:
        with self._lock:
            if busy:
                self._busy_workers.add(worker_index)
            else:
                self._busy_workers.discard(worker_index)

    def _set_worker_error(self, worker_index: int, error_message: str | None) -> None:
        with self._lock:
            self._worker_errors[worker_index] = error_message

    def _resolve_request_timeout_s(self, config: ServiceConfig) -> float:
        timeout_ms = config.pool_wait_timeout_ms(buffer_ms=self._request_timeout_buffer_ms)
        return timeout_ms / 1000
