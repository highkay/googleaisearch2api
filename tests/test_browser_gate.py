from __future__ import annotations

import threading
import time

from googleaisearch2api.browser_gate import BrowserResourceGate


def test_shared_blocked_while_exclusive_held() -> None:
    gate = BrowserResourceGate()
    assert gate.acquire_exclusive("recovery", timeout_s=0.1) is True
    assert gate.try_acquire_shared("worker") is False
    gate.release_exclusive("recovery")
    assert gate.try_acquire_shared("worker") is True
    gate.release_shared()


def test_exclusive_waits_for_shared_release() -> None:
    gate = BrowserResourceGate()
    assert gate.try_acquire_shared("worker") is True
    started = threading.Event()
    acquired = threading.Event()

    def _hold_exclusive() -> None:
        started.set()
        ok = gate.acquire_exclusive("recovery", timeout_s=2.0)
        if ok:
            acquired.set()
            gate.release_exclusive("recovery")

    thread = threading.Thread(target=_hold_exclusive, daemon=True)
    thread.start()
    assert started.wait(1.0)
    time.sleep(0.05)
    assert not acquired.is_set()
    gate.release_shared()
    assert acquired.wait(1.0)
    thread.join(timeout=1.0)
