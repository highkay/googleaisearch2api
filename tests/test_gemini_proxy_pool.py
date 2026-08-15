from __future__ import annotations

from googleaisearch2api.gemini_proxy_pool import GeminiWarpPool


def _always_true(_url: str) -> bool:
    return True


def test_pick_healthy_round_robins_over_all_exits() -> None:
    pool = GeminiWarpPool(["p1", "p2", "p3"])

    picked = [pool.pick_healthy(_always_true) for _ in range(6)]

    assert picked == ["p1", "p2", "p3", "p1", "p2", "p3"]


def test_failed_probe_cools_exit_and_pick_skips_it() -> None:
    pool = GeminiWarpPool(["p1", "p2"])

    assert pool.pick_healthy(lambda url: url != "p1") == "p2"
    assert pool.is_cooling("p1") is True
    assert pool.pick_healthy(_always_true) == "p2"


def test_record_failure_cools_exit_so_pick_skips_it() -> None:
    pool = GeminiWarpPool(["p1", "p2"])

    pool.record_failure("p1")

    assert pool.is_cooling("p1") is True
    assert pool.pick_healthy(_always_true) == "p2"


def test_pick_healthy_returns_none_when_all_probes_fail() -> None:
    pool = GeminiWarpPool(["p1", "p2", "p3"])

    assert pool.pick_healthy(lambda _url: False) is None


def test_empty_pool_has_size_zero_and_pick_returns_none() -> None:
    pool = GeminiWarpPool([])

    assert pool.size == 0
    assert pool.pick_healthy(_always_true) is None


def test_max_tries_bounds_probe_calls_in_cursor_order() -> None:
    pool = GeminiWarpPool(["p1", "p2", "p3"])
    calls: list[str] = []

    def probe(url: str) -> bool:
        calls.append(url)
        return False

    assert pool.pick_healthy(probe, max_tries=2) is None
    assert calls == ["p1", "p2"]


def test_init_dedupes_proxies_preserving_order() -> None:
    pool = GeminiWarpPool(["p2", "p1", "p2", "p3", "p1"])

    assert pool.size == 3
    picked = [pool.pick_healthy(_always_true) for _ in range(3)]
    assert picked == ["p2", "p1", "p3"]
