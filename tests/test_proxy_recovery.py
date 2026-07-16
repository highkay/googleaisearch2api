from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from googleaisearch2api.config import AppSettings, ServiceConfig
from googleaisearch2api.proxy_recovery import ProxyAutoRecovery


class _FakeConfigStore:
    def __init__(self, config: ServiceConfig) -> None:
        self._config = config

    def get_config(self) -> ServiceConfig:
        return self._config


def _make_probe_script(tmp_path: Path) -> Path:
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script_path = script_dir / "probe_proxy_sessions.py"
    script_path.write_text("# test probe script\n", encoding="utf-8")
    return script_path


def _sticky_config(*, enabled: bool = True) -> ServiceConfig:
    return ServiceConfig(
        browser_proxy_server="http://192.0.2.1:2260",
        browser_proxy_username="openai",
        browser_proxy_password="pass",
        resin_sticky_session_enabled=enabled,
    )


def test_proxy_auto_recovery_runs_existing_session_probe(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        APP_DATA_DIR=tmp_path,
        PROXY_AUTO_RECOVERY_ENABLED=True,
    )
    script_path = _make_probe_script(tmp_path)
    calls: list[tuple[list[str], dict]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout='{"active_sessions": 1}', stderr="")

    recovery = ProxyAutoRecovery(
        settings,
        _FakeConfigStore(_sticky_config()),
        script_path=script_path,
        command_runner=runner,
    )

    assert recovery.run_once(reason="test") is True

    command, kwargs = calls[0]
    assert str(script_path) in command
    assert command[command.index("--base-username") + 1] == "openai"
    assert "--existing-sessions" in command
    assert command[command.index("--existing-session-limit") + 1] == "0"
    assert command[command.index("--max-probes") + 1] == "5"
    assert "--refresh-active" not in command
    assert "--retry-cooldown" in command
    assert "--retry-retired" not in command
    assert "--skip-egress" in command
    assert "--skip-iplark" in command
    assert "--fast-ipapi-egress" not in command
    assert "--fast-http-prefilter" in command
    assert "--full-fast-http-sweep" in command
    assert command[command.index("--fast-http-timeout") + 1] == "8.0"
    assert command[command.index("--fast-http-scan-limit") + 1] == "0"
    assert command[command.index("--fast-http-workers") + 1] == "16"
    assert "--allow-known-google-blocked-ip" in command
    assert "--allow-known-google-blocked-prefix" in command
    assert "--skip-duck-canary" in command
    assert command[command.index("--canary-repeats") + 1] == "1"
    assert command[command.index("--active-freshness-seconds") + 1] == "43200"
    assert command[command.index("--stop-after-active") + 1] == "1"
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["timeout"] == 1800
    assert recovery.status()["last_success"] is True


def test_proxy_auto_recovery_skips_when_sticky_sessions_are_disabled(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        APP_DATA_DIR=tmp_path,
        PROXY_AUTO_RECOVERY_ENABLED=True,
    )
    script_path = _make_probe_script(tmp_path)
    started = threading.Event()
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        started.set()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    recovery = ProxyAutoRecovery(
        settings,
        _FakeConfigStore(_sticky_config(enabled=False)),
        script_path=script_path,
        command_runner=runner,
    )

    assert recovery.run_once(reason="test") is False
    assert calls == []
    assert recovery.status()["last_skipped_reason"] == "sticky proxy sessions are disabled"


def test_proxy_auto_recovery_can_probe_generated_range(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        APP_DATA_DIR=tmp_path,
        PROXY_AUTO_RECOVERY_ENABLED=True,
        PROXY_AUTO_RECOVERY_EXISTING_SESSIONS=False,
        PROXY_AUTO_RECOVERY_START=10,
        PROXY_AUTO_RECOVERY_END=20,
        PROXY_AUTO_RECOVERY_TARGET_ACTIVE=0,
        PROXY_AUTO_RECOVERY_SKIP_EGRESS=False,
        PROXY_AUTO_RECOVERY_SKIP_IPLARK=False,
        PROXY_AUTO_RECOVERY_FAST_IPAPI_EGRESS=True,
        PROXY_AUTO_RECOVERY_ALLOW_KNOWN_GOOGLE_BLOCKED_IP=False,
        PROXY_AUTO_RECOVERY_ALLOW_KNOWN_GOOGLE_BLOCKED_PREFIX=False,
        PROXY_AUTO_RECOVERY_RETRY_RETIRED=False,
    )
    script_path = _make_probe_script(tmp_path)
    started = threading.Event()
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        started.set()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    recovery = ProxyAutoRecovery(
        settings,
        _FakeConfigStore(_sticky_config()),
        script_path=script_path,
        command_runner=runner,
    )

    assert recovery.run_once(reason="test") is True

    command = calls[0]
    assert "--existing-sessions" not in command
    assert "--retry-retired" not in command
    assert "--skip-egress" not in command
    assert "--skip-iplark" not in command
    assert "--fast-ipapi-egress" in command
    assert "--allow-known-google-blocked-ip" not in command
    assert "--allow-known-google-blocked-prefix" not in command
    assert "--stop-after-active" not in command
    assert command[command.index("--start") + 1] == "10"
    assert command[command.index("--end") + 1] == "20"


def test_proxy_auto_recovery_trigger_async_runs_probe_once(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        APP_DATA_DIR=tmp_path,
        PROXY_AUTO_RECOVERY_ENABLED=True,
    )
    script_path = _make_probe_script(tmp_path)
    started = threading.Event()
    release = threading.Event()
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        started.set()
        assert release.wait(timeout=2)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    recovery = ProxyAutoRecovery(
        settings,
        _FakeConfigStore(_sticky_config()),
        script_path=script_path,
        command_runner=runner,
    )

    assert recovery.trigger_async(reason="empty-pool") is True
    assert started.wait(timeout=2)
    assert recovery.trigger_async(reason="empty-pool-again") is False

    release.set()
    for _ in range(200):
        if not recovery.status()["running"]:
            break
        time.sleep(0.01)

    assert len(calls) == 1
    assert recovery.status()["running"] is False
    assert recovery.status()["last_success"] is True


def test_proxy_auto_recovery_trigger_async_respects_cooldown(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        APP_DATA_DIR=tmp_path,
        PROXY_AUTO_RECOVERY_ENABLED=True,
        PROXY_AUTO_RECOVERY_MIN_TRIGGER_INTERVAL_SECONDS=900,
    )
    script_path = _make_probe_script(tmp_path)
    started = threading.Event()
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        started.set()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    recovery = ProxyAutoRecovery(
        settings,
        _FakeConfigStore(_sticky_config()),
        script_path=script_path,
        command_runner=runner,
    )

    assert recovery.trigger_async(reason="empty-pool") is True
    assert started.wait(timeout=2)
    for _ in range(200):
        if not recovery.status()["running"]:
            break
        time.sleep(0.01)

    assert recovery.trigger_async(reason="empty-pool-again") is False
    assert len(calls) == 1
    assert "cooling down" in str(recovery.status()["last_skipped_reason"])
