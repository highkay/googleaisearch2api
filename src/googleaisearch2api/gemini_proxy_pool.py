"""Round-robin proxy exit pool with per-exit failure cooldown (pure stdlib).

The probe is injected so this module stays free of imports from
``config``/``fast_proxy_probe``/``gemini_web`` (no dependency cycles).
All cooldown clocks use ``time.monotonic()``; tests never sleep.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class _ExitState:
    url: str
    fails: int = 0
    cooldown_until: float = 0.0


class GeminiWarpPool:
    def __init__(self, proxies: list[str], *, cooldown_sec: float = 300.0) -> None:
        self._cooldown_sec = cooldown_sec
        self._order = list(dict.fromkeys(proxies))
        self._states = {url: _ExitState(url=url) for url in self._order}
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._order)

    def pick_healthy(
        self,
        probe: Callable[[str], bool],
        *,
        max_tries: int | None = None,
    ) -> str | None:
        with self._lock:
            size = len(self._order)
            if size == 0:
                return None
            cap = min(size if max_tries is None else max_tries, size)
            now = time.monotonic()
            for _ in range(cap):
                url = self._order[self._cursor]
                self._cursor = (self._cursor + 1) % size
                state = self._states[url]
                if state.cooldown_until > now:
                    continue
                if probe(url):
                    state.fails = 0
                    state.cooldown_until = 0.0
                    return url
                state.fails += 1
                state.cooldown_until = now + self._cooldown_sec
            return None

    def record_failure(self, url: str) -> None:
        with self._lock:
            state = self._states.get(url)
            if state is None:
                return
            state.cooldown_until = time.monotonic() + self._cooldown_sec
            state.fails += 1

    def is_cooling(self, url: str) -> bool:
        with self._lock:
            state = self._states.get(url)
            return state is not None and state.cooldown_until > time.monotonic()
