"""One-glance health report for the googleaisearch2api deployment.

Reads the SQLite store (request logs + proxy session pool) and prints request
success rate, latency distribution, engine mix, proxy-pool health, and the most
recent request/error activity. Run inside the container:

    docker exec googleaisearch2api-googleaisearch2api-1 \
        /app/.venv/bin/python /app/scripts/monitor.py

or locally with an explicit DB path:

    uv run python scripts/monitor.py --db /path/to/googleaisearch2api.sqlite3
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


def _fmt_ms(ms: int | None) -> str:
    if ms is None:
        return "   -"
    return f"{ms:>5}ms"


def _fmt_rate(ok: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{ok / total * 100:.0f}% ({ok}/{total})"


def report(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    print("== 请求成功率（按引擎） ==")
    rows = con.execute(
        "SELECT engine, status, COUNT(*) n, "
        "ROUND(AVG(duration_ms)) avg_ms, MAX(duration_ms) max_ms "
        "FROM request_logs GROUP BY engine, status ORDER BY engine, status"
    ).fetchall()
    by_engine: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        by_engine[r["engine"]][r["status"]] = r["n"]
        label = "ok" if r["status"] == "ok" else "error"
        print(
            f"  {r['engine']:16} {label:6} n={r['n']:5} "
            f"avg={_fmt_ms(r['avg_ms'])} max={_fmt_ms(r['max_ms'])}"
        )
    print("  ---- 汇总 ----")
    for engine, counts in sorted(by_engine.items()):
        ok = counts.get("ok", 0)
        total = ok + counts.get("error", 0)
        print(f"  {engine:16} 成功率 {_fmt_rate(ok, total)}")

    print("\n== 请求延迟分位（近 200 条 ok） ==")
    lat = [
        r["duration_ms"]
        for r in con.execute(
            "SELECT duration_ms FROM request_logs WHERE status='ok' "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
    ]
    if lat:
        lat.sort()
        n = len(lat)

        def percentile(q: float) -> int:
            return lat[min(n - 1, int(n * q))]

        print(
            f"  n={n} p50={percentile(0.5)}ms p90={percentile(0.9)}ms "
            f"p95={percentile(0.95)}ms max={lat[-1]}ms"
        )

    print("\n== 代理会话池 ==")
    for r in con.execute(
        "SELECT status, COUNT(*) n FROM proxy_sessions GROUP BY status ORDER BY status"
    ).fetchall():
        print(f"  {r['status']:14} {r['n']}")
    agg = con.execute(
        "SELECT SUM(request_block_count) b, SUM(request_error_count) e, "
        "SUM(request_success_count) s, "
        "SUM(CASE WHEN google_canary_status='ok' THEN 1 ELSE 0 END) gc_ok "
        "FROM proxy_sessions WHERE status != 'retired'"
    ).fetchone()
    if agg:
        print(
            f"  blocks={agg['b']} errors={agg['e']} success={agg['s']} "
            f"gemini_canary_ok={agg['gc_ok']}"
        )

    print("\n== 最近 10 条请求 ==")
    for r in con.execute(
        "SELECT created_at, engine, status, duration_ms, proxy_username, "
        "substr(coalesce(response_preview, error_message), 1, 48) preview "
        "FROM request_logs ORDER BY created_at DESC LIMIT 10"
    ).fetchall():
        print(
            f"  {r['created_at'][5:19]} {r['engine']:12} {r['status']:7} "
            f"{_fmt_ms(r['duration_ms'])} {str(r['proxy_username'] or '-'):14} {r['preview']}"
        )

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="googleaisearch2api health report")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/googleaisearch2api.sqlite3"),
        help="Path to the SQLite store (default: data/googleaisearch2api.sqlite3)",
    )
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}")
    report(args.db)


if __name__ == "__main__":
    main()
