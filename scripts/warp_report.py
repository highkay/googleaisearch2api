"""Aligned table rendering for the warp fleet health monitor."""

from __future__ import annotations


def _dash(value: object) -> str:
    return "-" if value is None else str(value)


def _flag(value: object, *, yes: str = "yes", no: str = "no") -> str:
    if value is True:
        return yes
    if value is False:
        return no
    return "-"


def render_table(records: list[dict[str, object]]) -> str:
    def cells(record: dict[str, object]) -> list[str]:
        probe = record.get("probe")
        if not isinstance(probe, dict) or not probe:
            probe_cell = "skip"
        elif probe.get("ok"):
            probe_cell = f"ok/{probe.get('client')}"
        else:
            probe_cell = f"FAIL:{(str(probe.get('error') or ''))[:28]}"
        return [
            str(record.get("name")),
            _dash(record.get("host_port")),
            str(record.get("classification")),
            _dash(record.get("restart_count")),
            _flag(record.get("oom_killed")),
            _dash(record.get("exit_code")),
            _flag(record.get("has_healthcheck")),
            _flag(record.get("ephemeral_identity"), yes="ephemeral", no="volume"),
            probe_cell,
        ]

    header = (
        "CONTAINER",
        "PORT",
        "CLASSIFICATION",
        "RESTARTS",
        "OOM",
        "EXIT",
        "HCHECK",
        "IDENTITY",
        "PROBE",
    )
    rows = [list(header), *(cells(record) for record in records)]
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows
    )
