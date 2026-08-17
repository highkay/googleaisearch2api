"""Lazy read-only docker access for the warp fleet monitor.

Prefers the docker SDK when importable, else shells out to the docker CLI
(docker ps/docker inspect are read-only operations).
"""

from __future__ import annotations

import json
import shutil
import subprocess


class DockerError(RuntimeError):
    """Raised when the docker CLI/SDK cannot provide container state."""


def list_container_names(prefix: str) -> list[str]:
    if shutil.which("docker") is None:
        raise DockerError("docker CLI not found on PATH")
    try:
        output = subprocess.check_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}"], text=True, shell=False
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DockerError(f"docker ps failed: {exc}") from exc
    return sorted(name.strip() for name in output.splitlines() if name.strip().startswith(prefix))


def _via_sdk(name: str) -> dict[str, object] | None:
    try:
        import docker  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        attrs = docker.from_env().containers.get(name).attrs
    except Exception:
        # SDK present but daemon/socket unreachable: fall back to the CLI.
        return None
    return attrs if isinstance(attrs, dict) else None


def _via_cli(name: str) -> dict[str, object]:
    try:
        output = subprocess.check_output(["docker", "inspect", name], text=True, shell=False)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DockerError(f"docker inspect failed for {name}: {exc}") from exc
    payloads = json.loads(output)
    if not isinstance(payloads, list) or not payloads or not isinstance(payloads[0], dict):
        raise DockerError(f"docker inspect returned no data for {name}")
    return payloads[0]


def inspect_container(name: str) -> dict[str, object]:
    payload = _via_sdk(name)
    return payload if payload is not None else _via_cli(name)
