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
    gate.release_shared("worker")


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
    gate.release_shared("worker")
    assert acquired.wait(1.0)
    thread.join(timeout=1.0)


def test_release_shared_is_idempotent_per_holder() -> None:
    gate = BrowserResourceGate()
    assert gate.try_acquire_shared("a") is True
    gate.release_shared("a")
    gate.release_shared("a")
    assert gate.status()["shared_holders"] == 0
    assert gate.acquire_exclusive("recovery", timeout_s=0.1) is True
    assert gate.status()["shared_holders"] == 0


def test_release_shared_unknown_holder_is_noop() -> None:
    gate = BrowserResourceGate()
    before = gate.status()
    gate.release_shared("ghost")
    assert gate.status() == before
    assert before["shared_holders"] == 0


def test_two_distinct_holders_coexist() -> None:
    gate = BrowserResourceGate()
    assert gate.try_acquire_shared("a") is True
    assert gate.try_acquire_shared("b") is True
    assert gate.status()["shared_holders"] == 2
    gate.release_shared("a")
    assert gate.status()["shared_holders"] == 1
    gate.release_shared("b")
    assert gate.status()["shared_holders"] == 0


def test_stale_release_does_not_drop_new_generation_holder() -> None:
    gate = BrowserResourceGate()
    assert gate.try_acquire_shared("w-0-g1") is True
    gate.release_shared_for("w-0-g1")
    assert gate.try_acquire_shared("w-0-g2") is True
    gate.release_shared("w-0-g1")  # stale late release from the wedged generation
    assert gate.status()["shared_holders"] == 1
    gate.release_shared("w-0-g2")
    assert gate.status()["shared_holders"] == 0


def test_reacquire_same_holder_is_idempotent() -> None:
    gate = BrowserResourceGate()
    assert gate.try_acquire_shared("a") is True
    assert gate.try_acquire_shared("a") is True
    assert gate.status()["shared_holders"] == 1
