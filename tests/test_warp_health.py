from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Order matters: warp_health imports warp_probe as a plain sibling import.
_WARP_PROBE = _load_script("warp_probe", _SCRIPT_DIR / "warp_probe.py")
_WARP_HEALTH = _load_script("warp_health", _SCRIPT_DIR / "warp_health.py")

build_arg_parser = _WARP_HEALTH.build_arg_parser
build_record = _WARP_HEALTH.build_record
classify_container = _WARP_HEALTH.classify_container
exit_history_summary = _WARP_HEALTH.exit_history_summary
has_healthcheck = _WARP_HEALTH.has_healthcheck
healthy_percent = _WARP_HEALTH.healthy_percent
host_port_for = _WARP_HEALTH.host_port_for
is_ephemeral_identity = _WARP_HEALTH.is_ephemeral_identity
probe_response_ok = _WARP_PROBE.probe_response_ok
socks5_connect_request = _WARP_PROBE.socks5_connect_request
ProbeResult = _WARP_PROBE.ProbeResult

CLASS_HEALTHY = _WARP_HEALTH.CLASS_HEALTHY
CLASS_RESTART_LOOP = _WARP_HEALTH.CLASS_RESTART_LOOP
CLASS_STOPPED = _WARP_HEALTH.CLASS_STOPPED
CLASS_UNPROBED = _WARP_HEALTH.CLASS_UNPROBED
CLASS_ZOMBIE_TUNNEL = _WARP_HEALTH.CLASS_ZOMBIE_TUNNEL


def _inspect_json(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "Name": "/warpplus-de",
        "State": {
            "Status": "running",
            "Restarting": False,
            "OOMKilled": False,
            "RestartCount": 0,
            "ExitCode": 0,
            "FinishedAt": "2026-08-16T02:21:25.233003043Z",
            "StartedAt": "2026-08-17T02:21:25.233003043Z",
        },
        "Config": {"Healthcheck": None},
        "Mounts": [],
        "NetworkSettings": {
            "Ports": {
                "1080/tcp": [
                    {"HostIp": "0.0.0.0", "HostPort": "1081"},
                    {"HostIp": "::", "HostPort": "1081"},
                ]
            }
        },
    }
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            existing.update(value)  # type: ignore[union-attr]
        else:
            base[key] = value
    return base


# --- classify_container -------------------------------------------------------


def test_classify_returns_healthy_when_probe_ok_and_healthcheck_missing() -> None:
    inspect = _inspect_json()
    assert classify_container(inspect, True) == CLASS_HEALTHY


def test_classify_returns_zombie_tunnel_when_running_but_probe_fails() -> None:
    inspect = _inspect_json(State={"RestartCount": 0})
    assert classify_container(inspect, False) == CLASS_ZOMBIE_TUNNEL


def test_classify_returns_restart_loop_when_restart_count_reaches_threshold() -> None:
    for probe_ok in (True, None):
        inspect = _inspect_json(State={"RestartCount": 5})
        assert classify_container(inspect, probe_ok) == CLASS_RESTART_LOOP


def test_classify_returns_restart_loop_when_oom_killed() -> None:
    inspect = _inspect_json(State={"OOMKilled": True})
    assert classify_container(inspect, True) == CLASS_RESTART_LOOP


def test_classify_returns_restart_loop_when_state_restarting() -> None:
    inspect = _inspect_json(State={"Status": "restarting"})
    assert classify_container(inspect, None) == CLASS_RESTART_LOOP


def test_classify_returns_stopped_when_state_exited() -> None:
    inspect = _inspect_json(State={"Status": "exited"})
    assert classify_container(inspect, None) == CLASS_STOPPED


def test_classify_returns_stopped_when_state_dead() -> None:
    inspect = _inspect_json(State={"Status": "dead"})
    assert classify_container(inspect, None) == CLASS_STOPPED


def test_classify_returns_unprobed_when_running_without_probe_result() -> None:
    inspect = _inspect_json(State={"RestartCount": 0})
    assert classify_container(inspect, None) == CLASS_UNPROBED


def test_classify_treats_missing_restart_count_as_zero() -> None:
    inspect = _inspect_json(State={"RestartCount": None})
    assert classify_container(inspect, True) == CLASS_HEALTHY


# --- inspect helpers ----------------------------------------------------------


def test_exit_history_summary_reports_code_restarts_oom_and_finished_at() -> None:
    inspect = _inspect_json(State={"RestartCount": 3, "OOMKilled": True})
    summary = exit_history_summary(inspect)
    assert summary["restart_count"] == 3
    assert summary["exit_code"] == 0
    assert summary["oom_killed"] is True
    assert summary["finished_at"].startswith("2026-08-16")


def test_has_healthcheck_detects_configured_healthcheck() -> None:
    assert has_healthcheck(_inspect_json()) is False
    assert (
        has_healthcheck(_inspect_json(Config={"Healthcheck": {"Test": ["CMD", "warp-cli"]}}))
        is True
    )


def test_is_ephemeral_identity_true_without_warp_mounts() -> None:
    inspect = _inspect_json(Mounts=[{"Destination": "/cache", "Type": "bind"}])
    assert is_ephemeral_identity(inspect) is True


def test_is_ephemeral_identity_false_with_app_data_mount() -> None:
    inspect = _inspect_json(Mounts=[{"Destination": "/app-data", "Type": "volume"}])
    assert is_ephemeral_identity(inspect) is False


def test_is_ephemeral_identity_false_with_warp_cache_mount() -> None:
    inspect = _inspect_json(Mounts=[{"Destination": "/root/.cache/warp-plus", "Type": "volume"}])
    assert is_ephemeral_identity(inspect) is False


def test_host_port_resolution_reads_mapped_tcp_port() -> None:
    assert host_port_for(_inspect_json()) == 1081
    assert host_port_for(_inspect_json(NetworkSettings={"Ports": {}})) is None
    assert host_port_for(_inspect_json(), container_port=9999) is None


# --- probe response checks ----------------------------------------------------


def test_probe_response_ok_accepts_http_200() -> None:
    assert probe_response_ok(200, "") is True


def test_probe_response_ok_accepts_warp_trace_marker_without_200() -> None:
    assert probe_response_ok(403, "ip=1.2.3.4\nwarp=plus\n") is True


def test_probe_response_ok_rejects_failure() -> None:
    assert probe_response_ok(502, "<html>bad gateway</html>") is False
    assert probe_response_ok(None, "") is False


# --- socks5 wire helpers ------------------------------------------------------


def test_socks5_connect_request_uses_remote_dns_atyp() -> None:
    payload = socks5_connect_request("connectivity.cloudflareclient.com", 443)
    assert payload == (
        b"\x05\x01\x00\x03"
        + bytes([len(b"connectivity.cloudflareclient.com")])
        + b"connectivity.cloudflareclient.com"
        + b"\x01\xbb"
    )


# --- aggregation --------------------------------------------------------------


def test_healthy_percent_counts_healthy_share() -> None:
    records = [
        {"classification": CLASS_HEALTHY},
        {"classification": CLASS_HEALTHY},
        {"classification": CLASS_ZOMBIE_TUNNEL},
        {"classification": CLASS_STOPPED},
    ]
    assert healthy_percent(records) == 50.0


def test_healthy_percent_empty_inventory_is_zero() -> None:
    assert healthy_percent([]) == 0.0


def test_build_record_serializes_slotted_probe_result_as_dict() -> None:
    probe = ProbeResult(ok=True, client="curl_cffi", status_code=200, elapsed_s=1.2)
    record = build_record("warpplus-de", _inspect_json(), probe)
    assert record["classification"] == CLASS_HEALTHY
    assert record["probe"] == {
        "ok": True,
        "client": "curl_cffi",
        "status_code": 200,
        "elapsed_s": 1.2,
        "error": None,
        "evidence": None,
    }


# --- CLI ----------------------------------------------------------------------


def test_parser_defaults_match_fleet_monitor_contract() -> None:
    args = build_arg_parser().parse_args([])
    assert args.prefix == "warpplus"
    assert args.ports == 1080
    assert args.threshold == 60
    assert args.timeout == 8
    assert args.list is False
    assert args.json is False
    assert args.check is False
