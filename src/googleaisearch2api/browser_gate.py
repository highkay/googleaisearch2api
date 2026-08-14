from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class BrowserResourceGate:
    """Serialize expensive browser usage between API workers and recovery probes.

    Shared mode is used by request workers; holders are tracked by name so a
    wedged worker's slot can be force-released later without dropping a
    replacement worker's slot (stale late releases are per-name no-ops).
    Exclusive mode is used by proxy auto recovery so canary Chrome processes do
    not compete with live traffic for PID and memory (observed failure: fork:
    Resource temporarily unavailable).
    """

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.Lock())
        self._exclusive_holder: str | None = None
        self._holders: dict[str, float] = {}

    def status(self) -> dict[str, object]:
        with self._cv:
            return {
                "exclusive_holder": self._exclusive_holder,
                "shared_holders": len(self._holders),
                "busy": self._exclusive_holder is not None or bool(self._holders),
            }

    def is_exclusive(self) -> bool:
        with self._cv:
            return self._exclusive_holder is not None

    def try_acquire_shared(self, holder: str = "worker") -> bool:
        with self._cv:
            if self._exclusive_holder is not None:
                return False
            if holder not in self._holders:
                self._holders[holder] = time.monotonic()
            return True

    def release_shared(self, holder: str = "worker") -> None:
        with self._cv:
            self._holders.pop(holder, None)
            self._cv.notify_all()

    def release_shared_for(self, holder: str = "worker") -> None:
        self.release_shared(holder)

    def try_acquire_exclusive(self, holder: str = "recovery") -> bool:
        with self._cv:
            if self._exclusive_holder is not None or self._holders:
                return False
            self._exclusive_holder = holder
            return True

    def acquire_exclusive(
        self,
        holder: str = "recovery",
        *,
        timeout_s: float | None = 30.0,
    ) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + max(timeout_s, 0.0)
        with self._cv:
            while self._exclusive_holder is not None or self._holders:
                if deadline is None:
                    self._cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            self._exclusive_holder = holder
            return True

    def release_exclusive(self, holder: str | None = None) -> None:
        with self._cv:
            if holder is not None and self._exclusive_holder not in {None, holder}:
                return
            self._exclusive_holder = None
            self._cv.notify_all()

    @contextmanager
    def shared(self, holder: str = "worker") -> Iterator[bool]:
        acquired = self.try_acquire_shared(holder)
        try:
            yield acquired
        finally:
            if acquired:
                self.release_shared(holder)

    @contextmanager
    def exclusive(
        self,
        holder: str = "recovery",
        *,
        timeout_s: float | None = 30.0,
    ) -> Iterator[bool]:
        acquired = self.acquire_exclusive(holder, timeout_s=timeout_s)
        try:
            yield acquired
        finally:
            if acquired:
                self.release_exclusive(holder)
