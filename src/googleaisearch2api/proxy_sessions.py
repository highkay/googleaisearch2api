from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from .browser import resolve_browser_proxy
from .config import ServiceConfig
from .db import ProxyIpObservationRow, ProxySessionEventRow, ProxySessionRow, utc_now

DEFAULT_STICKY_SUFFIX_TEMPLATE = ".user{n}"
DEFAULT_BLOCK_COOLDOWN_HOURS = 24
STATUS_NEW = "new"
STATUS_EGRESS_CHECKED = "egress_checked"
STATUS_RISK_CHECKED = "risk_checked"
STATUS_ACTIVE = "active"
STATUS_COOLDOWN = "cooldown"
STATUS_RETIRED = "retired"
SELECTABLE_STATUSES = {STATUS_ACTIVE}

_IP_TOKEN_RE = re.compile(r"[0-9A-Fa-f:.]{3,}")


class ProxySessionConfigError(RuntimeError):
    pass


class ProxySessionUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProxySessionSnapshot:
    id: int
    proxy_base_username: str
    session_name: str
    proxy_username: str
    status: str
    epoch: int
    primary_ip: str | None
    ip_vector_hash: str | None
    iplark_min_quality_score: int | None
    google_canary_status: str
    google_canary_error: str | None
    google_canary_checked_at: datetime | None
    request_success_count: int
    request_block_count: int
    request_error_count: int
    canary_success_count: int
    canary_block_count: int
    duplicate_of_session_id: int | None
    last_selected_at: datetime | None
    last_success_at: datetime | None
    last_blocked_at: datetime | None
    cooldown_until: datetime | None
    retire_reason: str | None

    @classmethod
    def from_row(cls, row: ProxySessionRow) -> ProxySessionSnapshot:
        return cls(
            id=row.id,
            proxy_base_username=row.proxy_base_username,
            session_name=row.session_name,
            proxy_username=row.proxy_username,
            status=row.status,
            epoch=row.epoch,
            primary_ip=row.primary_ip,
            ip_vector_hash=row.ip_vector_hash,
            iplark_min_quality_score=row.iplark_min_quality_score,
            google_canary_status=row.google_canary_status,
            google_canary_error=row.google_canary_error,
            google_canary_checked_at=row.google_canary_checked_at,
            request_success_count=row.request_success_count,
            request_block_count=row.request_block_count,
            request_error_count=row.request_error_count,
            canary_success_count=row.canary_success_count,
            canary_block_count=row.canary_block_count,
            duplicate_of_session_id=row.duplicate_of_session_id,
            last_selected_at=row.last_selected_at,
            last_success_at=row.last_success_at,
            last_blocked_at=row.last_blocked_at,
            cooldown_until=row.cooldown_until,
            retire_reason=row.retire_reason,
        )


@dataclass(frozen=True, slots=True)
class ProxySessionSelection:
    session: ProxySessionSnapshot
    config: ServiceConfig


def _normalize_username(value: str | None) -> str:
    return (value or "").strip()


def format_sticky_username(
    base_username: str,
    index: int,
    suffix_template: str = DEFAULT_STICKY_SUFFIX_TEMPLATE,
) -> str:
    base = _normalize_username(base_username)
    if not base:
        raise ProxySessionConfigError("Sticky proxy session requires a base proxy username.")
    if index < 1:
        raise ValueError("Sticky proxy session index must be >= 1.")
    if "{base}" in suffix_template:
        return suffix_template.format(base=base, n=index)
    return f"{base}{suffix_template.format(n=index)}"


def resolve_proxy_base_username(config: ServiceConfig) -> str:
    proxy = resolve_browser_proxy(config)
    username = _normalize_username(proxy.get("username") if proxy else None)
    if not username:
        raise ProxySessionConfigError(
            "Resin sticky sessions require a proxy username prefix, for example openai."
        )
    return username


def build_proxy_config_for_session(config: ServiceConfig, proxy_username: str) -> ServiceConfig:
    return config.model_copy(update={"browser_proxy_username": proxy_username}, deep=True)


def normalize_ip_vector(ips: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    normalized: set[str] = set()
    for value in ips:
        try:
            normalized.add(str(ipaddress.ip_address(str(value).strip())))
        except ValueError:
            continue
    return sorted(normalized)


def hash_ip_vector(ips: list[str] | tuple[str, ...] | set[str]) -> str | None:
    normalized = normalize_ip_vector(ips)
    if not normalized:
        return None
    return hashlib.sha256(",".join(normalized).encode("utf-8")).hexdigest()


def parse_google_block_ips(message: str) -> list[str]:
    ips: list[str] = []
    seen: set[str] = set()
    for match in _IP_TOKEN_RE.finditer(message):
        token = match.group(0).strip(".,;()[]{}<>\"'")
        try:
            ip = str(ipaddress.ip_address(token))
        except ValueError:
            continue
        if ip not in seen:
            ips.append(ip)
            seen.add(ip)
    return ips


def google_block_has_ip_mismatch(ips: list[str]) -> bool:
    return len(set(ips)) >= 2


class ProxySessionStore:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def upsert_proxy_session(
        self,
        *,
        proxy_base_username: str,
        session_name: str,
        proxy_username: str,
        status: str | None = None,
        epoch: int | None = None,
    ) -> ProxySessionSnapshot:
        now = utc_now()
        with self._session_factory() as session:
            row = session.scalars(
                select(ProxySessionRow).where(ProxySessionRow.proxy_username == proxy_username)
            ).first()
            if row is None:
                row = ProxySessionRow(
                    proxy_base_username=proxy_base_username,
                    session_name=session_name,
                    proxy_username=proxy_username,
                    status=status or STATUS_NEW,
                    epoch=epoch if epoch is not None else 0,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.proxy_base_username = proxy_base_username
                row.session_name = session_name
                if epoch is not None:
                    row.epoch = epoch
                if status is not None:
                    row.status = status
                row.updated_at = now
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def record_ip_observation(
        self,
        *,
        proxy_session_id: int,
        epoch: int,
        source: str,
        ip: str,
        asn: str | None = None,
        organization: str | None = None,
        raw_json: dict[str, Any] | None = None,
    ) -> None:
        with self._session_factory() as session:
            row = ProxyIpObservationRow(
                proxy_session_id=proxy_session_id,
                epoch=epoch,
                source=source,
                ip=ip,
                asn=asn,
                organization=organization,
                raw_json=json.dumps(raw_json, ensure_ascii=False) if raw_json else None,
                observed_at=utc_now(),
            )
            session.add(row)
            session.commit()

    def record_event(
        self,
        *,
        proxy_session_id: int,
        event_type: str,
        message: str = "",
        raw_json: dict[str, Any] | list[Any] | None = None,
    ) -> None:
        with self._session_factory() as session:
            row = ProxySessionEventRow(
                proxy_session_id=proxy_session_id,
                event_type=event_type,
                message=message,
                raw_json=json.dumps(raw_json, ensure_ascii=False) if raw_json is not None else None,
                created_at=utc_now(),
            )
            session.add(row)
            session.commit()

    def update_egress(
        self,
        *,
        proxy_session_id: int,
        ips: list[str],
        source: str,
        asn: str | None = None,
        organization: str | None = None,
        raw_json: dict[str, Any] | None = None,
    ) -> ProxySessionSnapshot:
        vector = normalize_ip_vector(ips)
        primary_ip = vector[0] if vector else None
        vector_hash = hash_ip_vector(vector)
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                raise ProxySessionUnavailableError(
                    f"Proxy session {proxy_session_id} does not exist."
                )
            row.primary_ip = primary_ip
            row.ip_vector_json = json.dumps(vector, ensure_ascii=False) if vector else None
            row.ip_vector_hash = vector_hash
            row.asn = asn
            row.organization = organization
            row.status = STATUS_EGRESS_CHECKED
            row.duplicate_of_session_id = None
            row.last_checked_at = now
            row.updated_at = now

            if vector_hash:
                duplicate = session.scalars(
                    select(ProxySessionRow)
                    .where(ProxySessionRow.id != proxy_session_id)
                    .where(ProxySessionRow.ip_vector_hash == vector_hash)
                    .where(ProxySessionRow.duplicate_of_session_id.is_(None))
                    .where(ProxySessionRow.status != STATUS_RETIRED)
                    .order_by(ProxySessionRow.created_at.asc())
                    .limit(1)
                ).first()
                if duplicate is not None:
                    row.duplicate_of_session_id = duplicate.id
                    row.status = STATUS_RETIRED
                    row.retired_at = now
                    row.retire_reason = f"duplicate egress with session {duplicate.id}"

            for ip in vector:
                session.add(
                    ProxyIpObservationRow(
                        proxy_session_id=proxy_session_id,
                        epoch=row.epoch,
                        source=source,
                        ip=ip,
                        asn=asn,
                        organization=organization,
                        raw_json=json.dumps(raw_json, ensure_ascii=False) if raw_json else None,
                        observed_at=now,
                    )
                )
            session.add(row)
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def update_iplark_result(
        self,
        *,
        proxy_session_id: int,
        quality_score: int | None,
        usage_type: str | None = None,
        category: str | None = None,
        public_proxy: bool = False,
        threat: bool = False,
        tag: str | None = None,
        min_quality_score: int = 70,
    ) -> ProxySessionSnapshot:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                raise ProxySessionUnavailableError(
                    f"Proxy session {proxy_session_id} does not exist."
                )
            row.iplark_min_quality_score = quality_score
            row.iplark_usage_type = usage_type
            row.iplark_category = category
            row.iplark_public_proxy = public_proxy
            row.iplark_threat = threat
            row.iplark_tag = tag
            row.last_checked_at = now
            row.updated_at = now
            if public_proxy or threat:
                row.status = STATUS_RETIRED
                row.retired_at = now
                row.retire_reason = "iplark flagged public proxy/threat"
            elif quality_score is None:
                row.status = STATUS_COOLDOWN
                row.cooldown_until = now + timedelta(hours=DEFAULT_BLOCK_COOLDOWN_HOURS)
                row.retire_reason = "iplark score unavailable"
            elif quality_score is not None and quality_score < min_quality_score:
                row.status = STATUS_COOLDOWN
                row.cooldown_until = now + timedelta(hours=DEFAULT_BLOCK_COOLDOWN_HOURS)
                row.retire_reason = f"iplark score below threshold {min_quality_score}"
            elif row.status != STATUS_RETIRED:
                row.status = STATUS_RISK_CHECKED
                row.cooldown_until = None
                row.retire_reason = None
            session.add(row)
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def mark_canary_success(self, proxy_session_id: int) -> ProxySessionSnapshot:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                raise ProxySessionUnavailableError(
                    f"Proxy session {proxy_session_id} does not exist."
                )
            row.status = STATUS_ACTIVE
            row.google_canary_status = "ok"
            row.google_canary_error = None
            row.google_canary_checked_at = now
            row.canary_success_count += 1
            row.cooldown_until = None
            row.last_success_at = now
            row.updated_at = now
            session.add(row)
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def mark_canary_blocked(
        self,
        proxy_session_id: int,
        *,
        error_message: str,
        block_ips: list[str] | None = None,
    ) -> ProxySessionSnapshot:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                raise ProxySessionUnavailableError(
                    f"Proxy session {proxy_session_id} does not exist."
                )
            row.status = STATUS_COOLDOWN
            row.google_canary_status = "blocked"
            row.google_canary_error = error_message[:3000]
            row.google_canary_checked_at = now
            row.canary_block_count += 1
            row.last_blocked_at = now
            row.cooldown_until = now + timedelta(hours=DEFAULT_BLOCK_COOLDOWN_HOURS)
            row.updated_at = now
            session.add(row)
            for ip in block_ips or []:
                session.add(
                    ProxyIpObservationRow(
                        proxy_session_id=proxy_session_id,
                        epoch=row.epoch,
                        source="google_block",
                        ip=ip,
                        observed_at=now,
                    )
                )
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def mark_session_cooldown(
        self,
        proxy_session_id: int,
        *,
        reason: str,
        hours: int = DEFAULT_BLOCK_COOLDOWN_HOURS,
    ) -> ProxySessionSnapshot:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                raise ProxySessionUnavailableError(
                    f"Proxy session {proxy_session_id} does not exist."
                )
            row.status = STATUS_COOLDOWN
            row.retire_reason = reason[:3000]
            row.cooldown_until = now + timedelta(hours=hours)
            row.last_checked_at = now
            row.updated_at = now
            session.add(row)
            session.add(
                ProxySessionEventRow(
                    proxy_session_id=proxy_session_id,
                    event_type="cooldown",
                    message=reason[:3000],
                    created_at=now,
                )
            )
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def select_active_session(self, proxy_base_username: str) -> ProxySessionSnapshot | None:
        now = utc_now()
        with self._session_factory() as session:
            row = session.scalars(
                select(ProxySessionRow)
                .where(ProxySessionRow.proxy_base_username == proxy_base_username)
                .where(ProxySessionRow.status.in_(SELECTABLE_STATUSES))
                .where(ProxySessionRow.duplicate_of_session_id.is_(None))
                .where(
                    (ProxySessionRow.cooldown_until.is_(None))
                    | (ProxySessionRow.cooldown_until <= now)
                )
                .order_by(
                    ProxySessionRow.request_block_count.asc(),
                    ProxySessionRow.request_error_count.asc(),
                    ProxySessionRow.request_success_count.desc(),
                    ProxySessionRow.last_selected_at.asc().nullsfirst(),
                    ProxySessionRow.id.asc(),
                )
                .limit(1)
            ).first()
            if row is None:
                return None
            row.last_selected_at = now
            row.updated_at = now
            session.add(row)
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def finish_request_success(self, proxy_session_id: int) -> None:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                return
            row.request_success_count += 1
            row.status = STATUS_ACTIVE
            row.last_success_at = now
            row.cooldown_until = None
            row.updated_at = now
            session.add(row)
            session.commit()

    def finish_request_error(
        self,
        proxy_session_id: int,
        *,
        blocked: bool,
        error_message: str,
        block_ips: list[str] | None = None,
    ) -> None:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                return
            if blocked:
                row.request_block_count += 1
                row.status = STATUS_COOLDOWN
                row.last_blocked_at = now
                row.cooldown_until = now + timedelta(hours=DEFAULT_BLOCK_COOLDOWN_HOURS)
            else:
                row.request_error_count += 1
            row.updated_at = now
            session.add(row)
            session.add(
                ProxySessionEventRow(
                    proxy_session_id=proxy_session_id,
                    event_type="request_blocked" if blocked else "request_error",
                    message=error_message[:3000],
                    raw_json=(
                        json.dumps({"block_ips": block_ips}, ensure_ascii=False)
                        if block_ips
                        else None
                    ),
                    created_at=now,
                )
            )
            for ip in block_ips or []:
                session.add(
                    ProxyIpObservationRow(
                        proxy_session_id=proxy_session_id,
                        epoch=row.epoch,
                        source="google_block",
                        ip=ip,
                        observed_at=now,
                    )
                )
            session.commit()

    def get_summary(self) -> dict[str, int]:
        with self._session_factory() as session:
            rows = session.execute(
                select(ProxySessionRow.status, func.count()).group_by(ProxySessionRow.status)
            ).all()
        summary = {status: int(count) for status, count in rows}
        summary["total"] = sum(summary.values())
        return summary

    def list_proxy_sessions(self, limit: int = 20) -> list[ProxySessionSnapshot]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(ProxySessionRow)
                .order_by(ProxySessionRow.updated_at.desc(), ProxySessionRow.id.desc())
                .limit(limit)
            ).all()
            return [ProxySessionSnapshot.from_row(row) for row in rows]


class ProxySessionSelector:
    def __init__(self, store: ProxySessionStore, *, allow_fallback_to_base: bool = False):
        self._store = store
        self._allow_fallback_to_base = allow_fallback_to_base

    def select(self, config: ServiceConfig) -> ProxySessionSelection | None:
        if not config.resin_sticky_session_enabled:
            return None
        base_username = resolve_proxy_base_username(config)
        session = self._store.select_active_session(base_username)
        if session is None:
            if self._allow_fallback_to_base:
                return None
            raise ProxySessionUnavailableError(
                f"No active sticky proxy session is available for base username {base_username!r}."
            )
        return ProxySessionSelection(
            session=session,
            config=build_proxy_config_for_session(config, session.proxy_username),
        )
