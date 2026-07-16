from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .browser_gate import BrowserResourceGate
from .config import AppSettings, ServiceConfig
from .proxy_sessions import ProxySessionConfigError, resolve_proxy_base_username
from .store import ConfigStore

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class ProxyAutoRecoverySkipped(RuntimeError):
    pass


def _default_probe_script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "probe_proxy_sessions.py"


def _tail(value: str | None, *, limit: int = 4_000) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


class ProxyAutoRecovery:
    def __init__(
        self,
        settings: AppSettings,
        config_store: ConfigStore,
        *,
        script_path: Path | None = None,
        command_runner: CommandRunner = subprocess.run,
        browser_gate: BrowserResourceGate | None = None,
        exclusive_timeout_s: float = 60.0,
    ) -> None:
        self._settings = settings
        self._config_store = config_store
        self._script_path = script_path or _default_probe_script_path()
        self._repo_root = self._script_path.parent.parent
        self._command_runner = command_runner
        self._browser_gate = browser_gate
        self._exclusive_timeout_s = exclusive_timeout_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running = False
        self._last_started_at: datetime | None = None
        self._last_finished_at: datetime | None = None
        self._last_success: bool | None = None
        self._last_exit_code: int | None = None
        self._last_error: str | None = None
        self._last_skipped_reason: str | None = None
        self._next_trigger_after_monotonic = 0.0

    def start(self) -> None:
        if not self._settings.proxy_auto_recovery_enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="proxy-auto-recovery",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        with self._state_lock:
            return self._running

    def status(self) -> dict[str, object]:
        with self._state_lock:
            trigger_cooldown_remaining = max(
                int(self._next_trigger_after_monotonic - time.monotonic()),
                0,
            )
            return {
                "enabled": self._settings.proxy_auto_recovery_enabled,
                "running": self._running,
                "interval_seconds": self._settings.proxy_auto_recovery_interval_seconds,
                "existing_sessions": self._settings.proxy_auto_recovery_existing_sessions,
                "existing_session_limit": (
                    self._settings.proxy_auto_recovery_existing_session_limit
                ),
                "max_probes": self._settings.proxy_auto_recovery_max_probes,
                "target_active": self._settings.proxy_auto_recovery_target_active,
                "timeout_seconds": self._settings.proxy_auto_recovery_timeout_seconds,
                "min_trigger_interval_seconds": (
                    self._settings.proxy_auto_recovery_min_trigger_interval_seconds
                ),
                "trigger_cooldown_remaining_seconds": trigger_cooldown_remaining,
                "skip_egress": self._settings.proxy_auto_recovery_skip_egress,
                "skip_iplark": self._settings.proxy_auto_recovery_skip_iplark,
                "fast_ipapi_egress": self._settings.proxy_auto_recovery_fast_ipapi_egress,
                "fast_http_prefilter": self._settings.proxy_auto_recovery_fast_http_prefilter,
                "fast_http_timeout": self._settings.proxy_auto_recovery_fast_http_timeout,
                "fast_http_scan_limit": (
                    self._settings.proxy_auto_recovery_fast_http_scan_limit
                ),
                "fast_http_workers": self._settings.proxy_auto_recovery_fast_http_workers,
                "event_fast_http_scan_limit": (
                    self._settings.proxy_auto_recovery_event_fast_http_scan_limit
                ),
                "allow_known_google_blocked_ip": (
                    self._settings.proxy_auto_recovery_allow_known_google_blocked_ip
                ),
                "allow_known_google_blocked_prefix": (
                    self._settings.proxy_auto_recovery_allow_known_google_blocked_prefix
                ),
                "skip_duck_canary": self._settings.proxy_auto_recovery_skip_duck_canary,
                "canary_repeats": self._settings.proxy_auto_recovery_canary_repeats,
                "last_started_at": (
                    self._last_started_at.isoformat() if self._last_started_at else None
                ),
                "last_finished_at": (
                    self._last_finished_at.isoformat() if self._last_finished_at else None
                ),
                "last_success": self._last_success,
                "last_exit_code": self._last_exit_code,
                "last_error": self._last_error,
                "last_skipped_reason": self._last_skipped_reason,
            }

    def run_once(self, *, reason: str = "manual") -> bool:
        if not self._settings.proxy_auto_recovery_enabled:
            self._record_skip("auto recovery is disabled")
            return False
        if not self._run_lock.acquire(blocking=False):
            self._record_skip("auto recovery is already running")
            return False
        return self._run_once_locked(reason=reason)

    def trigger_async(self, *, reason: str = "event") -> bool:
        if not self._settings.proxy_auto_recovery_enabled:
            return False
        if not self._run_lock.acquire(blocking=False):
            return False
        if not self._consume_trigger_budget():
            self._run_lock.release()
            return False

        thread = threading.Thread(
            target=self._run_once_locked,
            name=f"proxy-auto-recovery-{reason}",
            kwargs={"reason": reason},
            daemon=True,
        )
        thread.start()
        return True

    def _run_once_locked(self, *, reason: str) -> bool:
        started_at = datetime.now(UTC)
        with self._state_lock:
            self._running = True
            self._last_started_at = started_at
            self._last_finished_at = None
            self._last_success = None
            self._last_exit_code = None
            self._last_error = None
            self._last_skipped_reason = None
        gate_holder = f"proxy-auto-recovery:{reason}"
        gate_acquired = False
        try:
            if self._browser_gate is not None:
                gate_acquired = self._browser_gate.acquire_exclusive(
                    gate_holder,
                    timeout_s=self._exclusive_timeout_s,
                )
                if not gate_acquired:
                    self._record_skip(
                        "browser resource gate busy; recovery deferred until API workers idle"
                    )
                    logger.info(
                        "Skipping proxy auto recovery probe: browser gate busy (reason={})",
                        reason,
                    )
                    return False
            config = self._config_store.get_config()
            command = self._build_command(config, reason=reason)
            logger.info(
                "Starting proxy auto recovery probe: reason={} interval={}s",
                reason,
                self._settings.proxy_auto_recovery_interval_seconds,
            )
            completed = self._command_runner(
                command,
                cwd=str(self._repo_root),
                capture_output=True,
                text=True,
                timeout=self._settings.proxy_auto_recovery_timeout_seconds,
                check=False,
            )
            stdout_tail = _tail(completed.stdout)
            stderr_tail = _tail(completed.stderr)
            if stdout_tail:
                logger.debug("Proxy auto recovery stdout tail:\n{}", stdout_tail)
            if stderr_tail:
                logger.warning("Proxy auto recovery stderr tail:\n{}", stderr_tail)
            success = completed.returncode == 0
            with self._state_lock:
                self._last_success = success
                self._last_exit_code = completed.returncode
                self._last_error = None if success else stderr_tail or stdout_tail or "probe failed"
            if success:
                logger.info("Proxy auto recovery probe completed successfully")
            else:
                logger.warning(
                    "Proxy auto recovery probe failed with exit code {}",
                    completed.returncode,
                )
            return success
        except ProxyAutoRecoverySkipped as exc:
            self._record_skip(str(exc))
            logger.info("Skipping proxy auto recovery probe: {}", exc)
            return False
        except (ProxySessionConfigError, FileNotFoundError, ValueError) as exc:
            self._record_error(exc)
            logger.warning("Proxy auto recovery probe could not start: {}", exc)
            return False
        except subprocess.TimeoutExpired as exc:
            self._record_error(exc)
            logger.warning(
                "Proxy auto recovery probe timed out after {}s",
                self._settings.proxy_auto_recovery_timeout_seconds,
            )
            return False
        except Exception as exc:
            self._record_error(exc)
            logger.exception("Proxy auto recovery probe failed unexpectedly")
            return False
        finally:
            if gate_acquired and self._browser_gate is not None:
                self._browser_gate.release_exclusive(gate_holder)
            with self._state_lock:
                self._running = False
                self._last_finished_at = datetime.now(UTC)
            self._run_lock.release()

    def _loop(self) -> None:
        if self._settings.proxy_auto_recovery_run_on_startup:
            self.run_once(reason="startup")
        while not self._stop_event.wait(self._settings.proxy_auto_recovery_interval_seconds):
            self.run_once(reason="interval")

    def _build_command(self, config: ServiceConfig, *, reason: str = "manual") -> list[str]:
        if not config.resin_sticky_session_enabled:
            raise ProxyAutoRecoverySkipped("sticky proxy sessions are disabled")
        if not config.proxy_enabled:
            raise ProxyAutoRecoverySkipped("browser proxy is disabled")
        if not self._script_path.exists():
            raise FileNotFoundError(f"probe script not found: {self._script_path}")
        if self._settings.proxy_auto_recovery_end < self._settings.proxy_auto_recovery_start:
            raise ValueError("PROXY_AUTO_RECOVERY_END must be >= PROXY_AUTO_RECOVERY_START")

        # Interval/startup: full-pool = existing inventory ∪ missing user{START}..END.
        # Event triggers: inventory-only + smaller fast-scan budget (request-path light).
        full_pool_sweep = reason in {"interval", "startup", "manual", "test"}
        fast_scan_limit = (
            self._settings.proxy_auto_recovery_fast_http_scan_limit
            if full_pool_sweep
            else self._settings.proxy_auto_recovery_event_fast_http_scan_limit
        )

        base_username = resolve_proxy_base_username(config)
        command = [
            sys.executable,
            str(self._script_path),
            "--base-username",
            base_username,
            "--retry-cooldown",
            "--shuffle",
            "--active-freshness-seconds",
            str(self._settings.proxy_auto_recovery_interval_seconds),
            "--canary-repeats",
            str(self._settings.proxy_auto_recovery_canary_repeats),
            "--full-fast-http-sweep" if full_pool_sweep else "--no-full-fast-http-sweep",
        ]
        if self._settings.proxy_auto_recovery_max_probes > 0:
            command.extend(["--max-probes", str(self._settings.proxy_auto_recovery_max_probes)])
        if self._settings.proxy_auto_recovery_skip_duck_canary:
            command.append("--skip-duck-canary")
        if self._settings.proxy_auto_recovery_retry_retired:
            command.append("--retry-retired")
        if self._settings.proxy_auto_recovery_fast_http_prefilter:
            command.extend(
                [
                    "--fast-http-prefilter",
                    "--fast-http-timeout",
                    str(self._settings.proxy_auto_recovery_fast_http_timeout),
                    "--fast-http-workers",
                    str(self._settings.proxy_auto_recovery_fast_http_workers),
                ]
            )
            # 0 means unlimited (scan whole dynamic pool). Always pass explicitly.
            command.extend(["--fast-http-scan-limit", str(fast_scan_limit)])
        else:
            command.append("--no-fast-http-prefilter")
        if self._settings.proxy_auto_recovery_skip_egress:
            command.append("--skip-egress")
        elif self._settings.proxy_auto_recovery_fast_ipapi_egress:
            command.append("--fast-ipapi-egress")
        if self._settings.proxy_auto_recovery_skip_iplark:
            command.append("--skip-iplark")
        if self._settings.proxy_auto_recovery_allow_known_google_blocked_ip:
            command.append("--allow-known-google-blocked-ip")
        if self._settings.proxy_auto_recovery_allow_known_google_blocked_prefix:
            command.append("--allow-known-google-blocked-prefix")
        if self._settings.proxy_auto_recovery_target_active > 0:
            command.extend(
                ["--stop-after-active", str(self._settings.proxy_auto_recovery_target_active)]
            )
        # Always pass the sticky index bound so empty inventory (and full-pool merge)
        # can discover missing user{START}..user{END} slots.
        command.extend(
            [
                "--start",
                str(self._settings.proxy_auto_recovery_start),
                "--end",
                str(self._settings.proxy_auto_recovery_end),
            ]
        )
        if self._settings.proxy_auto_recovery_existing_sessions:
            command.extend(
                [
                    "--existing-sessions",
                    "--existing-session-limit",
                    str(self._settings.proxy_auto_recovery_existing_session_limit),
                ]
            )
            # Interval/startup full sweep expands inventory: existing ∪ missing indices.
            # Event triggers stay inventory-only so request-path recovery stays light.
            if full_pool_sweep:
                command.append("--discover-missing-indices")
        return command

    def _consume_trigger_budget(self) -> bool:
        min_interval = self._settings.proxy_auto_recovery_min_trigger_interval_seconds
        if min_interval <= 0:
            return True
        now = time.monotonic()
        with self._state_lock:
            remaining = int(self._next_trigger_after_monotonic - now)
            if remaining > 0:
                self._last_skipped_reason = (
                    f"auto recovery trigger cooling down for {remaining}s"
                )
                return False
            self._next_trigger_after_monotonic = now + min_interval
        return True

    def _record_skip(self, reason: str) -> None:
        with self._state_lock:
            self._last_success = None
            self._last_exit_code = None
            self._last_error = None
            self._last_skipped_reason = reason

    def _record_error(self, exc: BaseException) -> None:
        with self._state_lock:
            self._last_success = False
            self._last_exit_code = None
            self._last_error = repr(exc)
            self._last_skipped_reason = None
