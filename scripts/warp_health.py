"""Read-only health monitor for the bepass-org/warp-plus SOCKS5 proxy fleet.

Probes each container's host-published SOCKS5 port with a REAL tunneled HTTPS
request (remote DNS, see warp_probe.py), then classifies from docker inspect
state plus probe evidence. Strictly read-only.

    uv run python scripts/warp_health.py --list | --json | --check
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from warp_docker import DockerError, inspect_container, list_container_names  # noqa: E402
from warp_probe import ProbeResult, probe_warp_exit  # noqa: E402
from warp_report import render_table  # noqa: E402

CLASS_HEALTHY = "healthy"
CLASS_ZOMBIE_TUNNEL = "zombie_tunnel"
CLASS_RESTART_LOOP = "restart_loop"
CLASS_STOPPED = "stopped"
CLASS_UNPROBED = "unprobed"
CLASS_UNKNOWN = "unknown"

DEFAULT_PREFIX = "warpplus"
DEFAULT_CONTAINER_PORT = 1080
DEFAULT_THRESHOLD = 60
DEFAULT_TIMEOUT = 8.0
RESTART_LOOP_RESTART_COUNT = 5
PROBE_WORKERS = 8


def _state_dict(inspect_json: dict[str, object]) -> dict[str, object]:
    state = inspect_json.get("State")
    return state if isinstance(state, dict) else {}


def _as_int(value: object) -> int:
    try:
        return int(str(value))
    except ValueError:
        return 0


def _has_restart_loop_evidence(state: dict[str, object]) -> bool:
    return (
        bool(state.get("OOMKilled"))
        or bool(state.get("Restarting"))
        or _as_int(state.get("RestartCount")) >= RESTART_LOOP_RESTART_COUNT
    )


def classify_container(inspect_json: dict[str, object], probe_ok: bool | None) -> str:
    """Pure classifier over a docker inspect payload + tunneled probe evidence."""
    state = _state_dict(inspect_json)
    status = str(state.get("Status") or "").lower()
    if status == "restarting":
        return CLASS_RESTART_LOOP
    if status != "running":
        return CLASS_STOPPED
    if _has_restart_loop_evidence(state):
        return CLASS_RESTART_LOOP
    if probe_ok is None:
        return CLASS_UNPROBED
    return CLASS_HEALTHY if probe_ok else CLASS_ZOMBIE_TUNNEL


def exit_history_summary(inspect_json: dict[str, object]) -> dict[str, object | None]:
    state = _state_dict(inspect_json)
    return {
        "restart_count": state.get("RestartCount"),
        "exit_code": state.get("ExitCode"),
        "finished_at": state.get("FinishedAt"),
        "oom_killed": bool(state.get("OOMKilled")),
    }


def has_healthcheck(inspect_json: dict[str, object]) -> bool:
    config = inspect_json.get("Config")
    return isinstance(config, dict) and config.get("Healthcheck") is not None


def is_ephemeral_identity(inspect_json: dict[str, object]) -> bool:
    """True when no volume/bind feeds the WARP identity (resets on restart)."""
    mounts = inspect_json.get("Mounts")
    if not isinstance(mounts, list):
        return True
    for mount in mounts:
        if not isinstance(mount, dict):
            continue
        destination = str(mount.get("Destination") or "").lower()
        if destination == "/app-data" or "warp-plus" in destination:
            return False
    return True


def host_port_for(
    inspect_json: dict[str, object], container_port: int = DEFAULT_CONTAINER_PORT
) -> int | None:
    settings = inspect_json.get("NetworkSettings")
    ports = settings.get("Ports") if isinstance(settings, dict) else None
    if not isinstance(ports, dict):
        return None
    entries = ports.get(f"{container_port}/tcp")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            host_port = int(str(entry.get("HostPort") or ""))
        except ValueError:
            continue
        if host_port > 0:
            return host_port
    return None


def healthy_percent(records: list[dict[str, object]]) -> float:
    if not records:
        return 0.0
    healthy = sum(1 for record in records if record.get("classification") == CLASS_HEALTHY)
    return healthy / len(records) * 100.0


def build_record(
    name: str, inspect_json: dict[str, object] | None, probe: ProbeResult | None
) -> dict[str, object]:
    record: dict[str, object] = {
        "name": name,
        "state": None,
        "host_port": None,
        "classification": CLASS_UNKNOWN,
        "probe": asdict(probe) if probe else None,
        "has_healthcheck": None,
        "ephemeral_identity": None,
        **exit_history_summary({}),
    }
    if inspect_json is not None:
        record["state"] = _state_dict(inspect_json).get("Status")
        record["host_port"] = host_port_for(inspect_json)
        record["classification"] = classify_container(inspect_json, probe.ok if probe else None)
        record["has_healthcheck"] = has_healthcheck(inspect_json)
        record["ephemeral_identity"] = is_ephemeral_identity(inspect_json)
        record.update(exit_history_summary(inspect_json))
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only health monitor for the bepass-org/warp-plus SOCKS5 fleet."
    )
    parser.add_argument("--list", action="store_true", help="Print the per-container table.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON array instead of a table.")
    parser.add_argument(
        "--check", action="store_true", help="Exit 0 when healthy-pct >= --threshold, else exit 1."
    )
    parser.add_argument(
        "--prefix", default=DEFAULT_PREFIX, help="Container name prefix to monitor."
    )
    parser.add_argument(
        "--ports",
        type=int,
        default=DEFAULT_CONTAINER_PORT,
        help="In-container SOCKS5 port used to find the host mapping.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Minimum healthy percentage for --check to exit 0.",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="Seconds per tunneled probe attempt."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 100:
        parser.error("--threshold must be between 0 and 100")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.ports <= 0:
        parser.error("--ports must be > 0")

    try:
        names = list_container_names(args.prefix)
    except DockerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not names:
        print(f"error: no containers match prefix {args.prefix!r}", file=sys.stderr)
        return 1

    inspected: dict[str, dict[str, object]] = {}
    for name in names:
        try:
            inspected[name] = inspect_container(name)
        except DockerError as exc:
            print(f"warning: {name}: {exc}", file=sys.stderr)

    targets = [
        (name, port)
        for name, payload in inspected.items()
        if _state_dict(payload).get("Status") == "running"
        and (port := host_port_for(payload, args.ports)) is not None
    ]
    probes: dict[str, ProbeResult] = {}
    with ThreadPoolExecutor(max_workers=min(PROBE_WORKERS, len(targets) or 1)) as pool:
        futures = {
            pool.submit(probe_warp_exit, host_port, timeout=args.timeout): name
            for name, host_port in targets
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                probes[name] = future.result()
            except Exception as exc:
                probes[name] = ProbeResult(ok=False, client="worker", error=repr(exc))

    records = [build_record(name, inspected.get(name), probes.get(name)) for name in sorted(names)]
    percentage = healthy_percent(records)

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        print(render_table(records))
        healthy = sum(1 for r in records if r.get("classification") == CLASS_HEALTHY)
        print(f"healthy: {healthy}/{len(records)} ({percentage:.1f}%) threshold {args.threshold}%")

    if args.check and percentage < args.threshold:
        print(f"check failed: {percentage:.1f}% < {args.threshold}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
