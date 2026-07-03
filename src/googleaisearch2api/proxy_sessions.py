from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, func, or_, select

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
RISK_METADATA_RETIRE_REASON_PREFIXES = (
    "iplark flagged public proxy/threat",
    "iplark score ",
    "iplark score missing",
    "ipapi score ",
)

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
    duck_canary_status: str
    duck_canary_error: str | None
    duck_canary_checked_at: datetime | None
    request_success_count: int
    request_block_count: int
    request_error_count: int
    canary_success_count: int
    canary_block_count: int
    duck_canary_success_count: int
    duck_canary_rate_limit_count: int
    duck_canary_error_count: int
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
            google_canary_status=row.google_canary_status or "unknown",
            google_canary_error=row.google_canary_error,
            google_canary_checked_at=row.google_canary_checked_at,
            duck_canary_status=getattr(row, "duck_canary_status", None) or "unknown",
            duck_canary_error=getattr(row, "duck_canary_error", None),
            duck_canary_checked_at=getattr(row, "duck_canary_checked_at", None),
            request_success_count=row.request_success_count,
            request_block_count=row.request_block_count,
            request_error_count=row.request_error_count,
            canary_success_count=row.canary_success_count,
            canary_block_count=row.canary_block_count,
            duck_canary_success_count=getattr(row, "duck_canary_success_count", 0) or 0,
            duck_canary_rate_limit_count=getattr(row, "duck_canary_rate_limit_count", 0) or 0,
            duck_canary_error_count=getattr(row, "duck_canary_error_count", 0) or 0,
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


@dataclass(frozen=True, slots=True)
class ProxyBlockedPrefixSnapshot:
    prefix: str
    blocked_count: int
    success_count: int
    matched_session: ProxySessionSnapshot


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


def _google_block_prefix_network(
    primary_ip: str,
) -> ipaddress.IPv4Network | ipaddress.IPv6Network | None:
    normalized = normalize_ip_vector([primary_ip])
    if not normalized:
        return None
    address = ipaddress.ip_address(normalized[0])
    prefix_len = 24 if address.version == 4 else 48
    return ipaddress.ip_network(f"{address}/{prefix_len}", strict=False)


def is_risk_metadata_retire_reason(reason: str | None) -> bool:
    text = (reason or "").strip().lower()
    return any(text.startswith(prefix) for prefix in RISK_METADATA_RETIRE_REASON_PREFIXES)


def _selectable_session_filter(now: datetime):
    previously_proven = or_(
        ProxySessionRow.request_success_count > 0,
        ProxySessionRow.canary_success_count > 0,
    )
    return or_(
        ProxySessionRow.status == STATUS_ACTIVE,
        and_(
            ProxySessionRow.status == STATUS_COOLDOWN,
            ProxySessionRow.cooldown_until.is_not(None),
            ProxySessionRow.cooldown_until <= now,
            previously_proven,
        ),
    )


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
            egress_changed = row.ip_vector_hash != vector_hash
            row.primary_ip = primary_ip
            row.ip_vector_json = json.dumps(vector, ensure_ascii=False) if vector else None
            row.ip_vector_hash = vector_hash
            row.asn = asn
            row.organization = organization
            row.status = STATUS_EGRESS_CHECKED
            row.duplicate_of_session_id = None
            row.last_checked_at = now
            row.updated_at = now
            if egress_changed:
                row.google_canary_status = "unknown"
                row.google_canary_error = None
                row.google_canary_checked_at = None
                row.duck_canary_status = "unknown"
                row.duck_canary_error = None
                row.duck_canary_checked_at = None

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
        del min_quality_score
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
            if row.status != STATUS_RETIRED:
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
            row.retire_reason = None
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

    def mark_duck_canary_success(self, proxy_session_id: int) -> ProxySessionSnapshot:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                raise ProxySessionUnavailableError(
                    f"Proxy session {proxy_session_id} does not exist."
                )
            row.duck_canary_status = "ok"
            row.duck_canary_error = None
            row.duck_canary_checked_at = now
            row.duck_canary_success_count = (row.duck_canary_success_count or 0) + 1
            row.last_success_at = now
            row.updated_at = now
            session.add(row)
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def mark_duck_canary_rate_limited(
        self,
        proxy_session_id: int,
        *,
        error_message: str,
    ) -> ProxySessionSnapshot:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                raise ProxySessionUnavailableError(
                    f"Proxy session {proxy_session_id} does not exist."
                )
            row.duck_canary_status = "rate_limited"
            row.duck_canary_error = error_message[:3000]
            row.duck_canary_checked_at = now
            row.duck_canary_rate_limit_count = (row.duck_canary_rate_limit_count or 0) + 1
            row.updated_at = now
            session.add(row)
            session.commit()
            session.refresh(row)
            return ProxySessionSnapshot.from_row(row)

    def mark_duck_canary_error(
        self,
        proxy_session_id: int,
        *,
        error_message: str,
    ) -> ProxySessionSnapshot:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                raise ProxySessionUnavailableError(
                    f"Proxy session {proxy_session_id} does not exist."
                )
            row.duck_canary_status = "error"
            row.duck_canary_error = error_message[:3000]
            row.duck_canary_checked_at = now
            row.duck_canary_error_count = (row.duck_canary_error_count or 0) + 1
            row.updated_at = now
            session.add(row)
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
        proven_success_rank = case((ProxySessionRow.request_success_count > 0, 0), else_=1)
        with self._session_factory() as session:
            row = session.scalars(
                select(ProxySessionRow)
                .where(ProxySessionRow.proxy_base_username == proxy_base_username)
                .where(_selectable_session_filter(now))
                .where(ProxySessionRow.duplicate_of_session_id.is_(None))
                .where(
                    (ProxySessionRow.cooldown_until.is_(None))
                    | (ProxySessionRow.cooldown_until <= now)
                )
                .order_by(
                    ProxySessionRow.request_block_count.asc(),
                    ProxySessionRow.request_error_count.asc(),
                    proven_success_rank.asc(),
                    ProxySessionRow.last_selected_at.asc().nullsfirst(),
                    ProxySessionRow.request_success_count.desc(),
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

    def select_duck_session(self, proxy_base_username: str) -> ProxySessionSnapshot | None:
        now = utc_now()
        with self._session_factory() as session:
            row = session.scalars(
                select(ProxySessionRow)
                .where(ProxySessionRow.proxy_base_username == proxy_base_username)
                .where(ProxySessionRow.duck_canary_status == "ok")
                .where(ProxySessionRow.status != STATUS_RETIRED)
                .where(ProxySessionRow.duplicate_of_session_id.is_(None))
                .order_by(
                    ProxySessionRow.duck_canary_rate_limit_count.asc(),
                    ProxySessionRow.request_error_count.asc(),
                    ProxySessionRow.last_selected_at.asc().nullsfirst(),
                    ProxySessionRow.duck_canary_success_count.desc(),
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

    def finish_request_success(self, proxy_session_id: int, *, engine: str = "google") -> None:
        now = utc_now()
        with self._session_factory() as session:
            row = session.get(ProxySessionRow, proxy_session_id)
            if row is None:
                return
            row.last_success_at = now
            if engine == "duck":
                row.duck_canary_status = "ok"
                row.duck_canary_error = None
                row.duck_canary_checked_at = now
                row.duck_canary_success_count = (row.duck_canary_success_count or 0) + 1
            else:
                row.request_success_count += 1
                row.status = STATUS_ACTIVE
                row.cooldown_until = None
                row.retire_reason = None
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

    def count_selectable_sessions(self, proxy_base_username: str) -> int:
        now = utc_now()
        with self._session_factory() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(ProxySessionRow)
                    .where(ProxySessionRow.proxy_base_username == proxy_base_username)
                    .where(_selectable_session_filter(now))
                    .where(ProxySessionRow.duplicate_of_session_id.is_(None))
                    .where(
                        (ProxySessionRow.cooldown_until.is_(None))
                        | (ProxySessionRow.cooldown_until <= now)
                    )
                )
                or 0
            )

    def count_active_sessions(
        self,
        proxy_base_username: str,
        *,
        checked_after: datetime | None = None,
    ) -> int:
        now = utc_now()
        with self._session_factory() as session:
            statement = (
                select(func.count())
                .select_from(ProxySessionRow)
                .where(ProxySessionRow.proxy_base_username == proxy_base_username)
                .where(ProxySessionRow.status == STATUS_ACTIVE)
                .where(ProxySessionRow.duplicate_of_session_id.is_(None))
                .where(
                    (ProxySessionRow.cooldown_until.is_(None))
                    | (ProxySessionRow.cooldown_until <= now)
                )
            )
            if checked_after is not None:
                statement = statement.where(
                    ProxySessionRow.google_canary_checked_at >= checked_after
                )
            return int(
                session.scalar(statement)
                or 0
            )

    def find_google_blocked_session_for_ip(
        self,
        proxy_base_username: str,
        primary_ip: str,
        *,
        exclude_session_id: int | None = None,
    ) -> ProxySessionSnapshot | None:
        normalized = normalize_ip_vector([primary_ip])
        if not normalized:
            return None

        def blocked_status_filter():
            return or_(
                ProxySessionRow.google_canary_status == "blocked",
                ProxySessionRow.canary_block_count > 0,
                ProxySessionRow.request_block_count > 0,
                ProxySessionRow.last_blocked_at.is_not(None),
            )

        statement = (
            select(ProxySessionRow)
            .where(ProxySessionRow.proxy_base_username == proxy_base_username)
            .where(ProxySessionRow.primary_ip == normalized[0])
            .where(blocked_status_filter())
            .order_by(
                ProxySessionRow.last_blocked_at.desc().nullslast(),
                ProxySessionRow.updated_at.desc(),
                ProxySessionRow.id.asc(),
            )
            .limit(1)
        )
        if exclude_session_id is not None:
            statement = statement.where(ProxySessionRow.id != exclude_session_id)

        with self._session_factory() as session:
            row = session.scalars(statement).first()
            if row is not None:
                return ProxySessionSnapshot.from_row(row)

            observation_statement = (
                select(ProxySessionRow)
                .join(
                    ProxyIpObservationRow,
                    ProxyIpObservationRow.proxy_session_id == ProxySessionRow.id,
                )
                .where(ProxySessionRow.proxy_base_username == proxy_base_username)
                .where(ProxyIpObservationRow.source == "google_block")
                .where(ProxyIpObservationRow.ip == normalized[0])
                .where(blocked_status_filter())
                .order_by(
                    ProxySessionRow.last_blocked_at.desc().nullslast(),
                    ProxySessionRow.updated_at.desc(),
                    ProxySessionRow.id.asc(),
                )
                .limit(1)
            )
            if exclude_session_id is not None:
                observation_statement = observation_statement.where(
                    ProxySessionRow.id != exclude_session_id
                )
            row = session.scalars(observation_statement).first()
            if row is not None:
                return ProxySessionSnapshot.from_row(row)
            return None

    def find_google_blocked_prefix_for_ip(
        self,
        proxy_base_username: str,
        primary_ip: str,
        *,
        min_blocked_count: int = 1,
        exclude_session_id: int | None = None,
    ) -> ProxyBlockedPrefixSnapshot | None:
        network = _google_block_prefix_network(primary_ip)
        if network is None:
            return None

        blocked_rows: list[ProxySessionRow] = []
        blocked_row_ids: set[int] = set()
        success_row_ids: set[int] = set()
        success_count = 0
        min_blocked_count = max(min_blocked_count, 1)

        def has_success(row: ProxySessionRow) -> bool:
            return (
                row.status == STATUS_ACTIVE
                or row.canary_success_count > 0
                or row.request_success_count > 0
            )

        def has_google_block(row: ProxySessionRow) -> bool:
            return (
                row.google_canary_status == "blocked"
                or row.canary_block_count > 0
                or row.request_block_count > 0
                or row.last_blocked_at is not None
            )

        def record_success(row: ProxySessionRow) -> None:
            nonlocal success_count
            if row.id not in success_row_ids:
                success_row_ids.add(row.id)
                success_count += 1

        def record_block(row: ProxySessionRow) -> None:
            if row.id not in blocked_row_ids:
                blocked_row_ids.add(row.id)
                blocked_rows.append(row)

        with self._session_factory() as session:
            statement = (
                select(ProxySessionRow)
                .where(ProxySessionRow.proxy_base_username == proxy_base_username)
                .where(ProxySessionRow.primary_ip.is_not(None))
                .order_by(
                    ProxySessionRow.last_blocked_at.desc().nullslast(),
                    ProxySessionRow.updated_at.desc(),
                    ProxySessionRow.id.asc(),
                )
            )
            if exclude_session_id is not None:
                statement = statement.where(ProxySessionRow.id != exclude_session_id)
            rows = session.scalars(statement).all()
            for row in rows:
                try:
                    address = ipaddress.ip_address(str(row.primary_ip))
                except ValueError:
                    continue
                if address not in network:
                    continue
                if has_success(row):
                    record_success(row)
                if has_google_block(row):
                    record_block(row)

            observation_statement = (
                select(ProxySessionRow, ProxyIpObservationRow.ip)
                .join(
                    ProxyIpObservationRow,
                    ProxyIpObservationRow.proxy_session_id == ProxySessionRow.id,
                )
                .where(ProxySessionRow.proxy_base_username == proxy_base_username)
                .where(ProxyIpObservationRow.source == "google_block")
                .order_by(
                    ProxySessionRow.last_blocked_at.desc().nullslast(),
                    ProxySessionRow.updated_at.desc(),
                    ProxySessionRow.id.asc(),
                )
            )
            if exclude_session_id is not None:
                observation_statement = observation_statement.where(
                    ProxySessionRow.id != exclude_session_id
                )
            for row, observed_ip in session.execute(observation_statement).all():
                try:
                    address = ipaddress.ip_address(str(observed_ip))
                except ValueError:
                    continue
                if address not in network:
                    continue
                if has_success(row):
                    record_success(row)
                if has_google_block(row):
                    record_block(row)

            if success_count > 0 or len(blocked_rows) < min_blocked_count:
                return None
            return ProxyBlockedPrefixSnapshot(
                prefix=str(network),
                blocked_count=len(blocked_rows),
                success_count=success_count,
                matched_session=ProxySessionSnapshot.from_row(blocked_rows[0]),
            )

    def list_proxy_sessions(
        self,
        limit: int = 20,
        *,
        proxy_base_username: str | None = None,
    ) -> list[ProxySessionSnapshot]:
        with self._session_factory() as session:
            statement = select(ProxySessionRow).order_by(
                ProxySessionRow.updated_at.desc(),
                ProxySessionRow.id.desc(),
            )
            if proxy_base_username is not None:
                statement = statement.where(
                    ProxySessionRow.proxy_base_username == proxy_base_username
                )
            rows = session.scalars(
                statement.limit(limit)
            ).all()
            return [ProxySessionSnapshot.from_row(row) for row in rows]


class ProxySessionSelector:
    def __init__(self, store: ProxySessionStore, *, allow_fallback_to_base: bool = False):
        self._store = store
        self._allow_fallback_to_base = allow_fallback_to_base

    def select(
        self,
        config: ServiceConfig,
        *,
        engine: str = "google",
    ) -> ProxySessionSelection | None:
        if not config.resin_sticky_session_enabled:
            return None
        base_username = resolve_proxy_base_username(config)
        if engine == "duck":
            session = self._store.select_duck_session(base_username)
        else:
            session = self._store.select_active_session(base_username)
        if session is None:
            if self._allow_fallback_to_base:
                return None
            session_label = "Duck.ai" if engine == "duck" else "active"
            raise ProxySessionUnavailableError(
                f"No {session_label} sticky proxy session is available "
                f"for base username {base_username!r}."
            )
        return ProxySessionSelection(
            session=session,
            config=build_proxy_config_for_session(config, session.proxy_username),
        )
