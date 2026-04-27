from __future__ import annotations

import json
import re
import uuid
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import delete, func, select

from .config import ServiceConfig, ServiceConfigUpdate
from .db import RequestLogRow, ServiceConfigRow, utc_now
from .schemas import Citation, DashboardSummary, GoogleAiResult, RecentRequest


def _coalesce_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


_INLINE_CREDENTIALS_PATTERN = re.compile(r"(?i)(https?://)([^:@/\s]+):([^@/\s]+)@")
_BEARER_PATTERN = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)([^\s\"'&,;]+)")
_LABELLED_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|password|passwd|pwd|secret)\b(\s*[:=]\s*)([^\s\"'&,;]+)"
)
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b")
_GOOGLE_KEY_PATTERN = re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")
_SENSITIVE_QUERY_KEYS = {"q", "query", "prompt", "input"}


def _mask_secret_fragment(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}***{value[-4:]}"


def _sanitize_logged_text(value: str, limit: int) -> str:
    text = (value or "").strip()
    if not text:
        return ""

    text = _INLINE_CREDENTIALS_PATTERN.sub(r"\1***:***@", text)
    text = _BEARER_PATTERN.sub(
        lambda match: f"{match.group(1)}{_mask_secret_fragment(match.group(2))}",
        text,
    )
    text = _LABELLED_SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_mask_secret_fragment(match.group(3))}",
        text,
    )
    text = _OPENAI_KEY_PATTERN.sub(lambda match: _mask_secret_fragment(match.group(0)), text)
    text = _GOOGLE_KEY_PATTERN.sub(lambda match: _mask_secret_fragment(match.group(0)), text)
    return text[:limit]


def _sanitize_logged_url(url: str | None) -> str | None:
    if not url:
        return None

    sanitized = _INLINE_CREDENTIALS_PATTERN.sub(r"\1***:***@", url.strip())
    parts = urlsplit(sanitized)
    if not parts.query:
        return sanitized

    filtered_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in _SENSITIVE_QUERY_KEYS
    ]
    query = urlencode(filtered_pairs, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


class ConfigStore:
    def __init__(self, session_factory, defaults: ServiceConfig, request_log_max_rows: int = 200):
        self._session_factory = session_factory
        self._defaults = defaults
        self._request_log_max_rows = request_log_max_rows

    def _row_to_config(self, row: ServiceConfigRow) -> ServiceConfig:
        return ServiceConfig(
            default_model=row.default_model,
            api_token=row.api_token,
            browser_headless=row.browser_headless,
            browser_user_agent=row.browser_user_agent,
            browser_locale=row.browser_locale,
            browser_base_url=row.browser_base_url,
            browser_timeout_ms=row.browser_timeout_ms,
            answer_timeout_ms=row.answer_timeout_ms,
            browser_proxy_server=row.browser_proxy_server,
            browser_proxy_username=row.browser_proxy_username,
            browser_proxy_password=row.browser_proxy_password,
            browser_proxy_bypass=row.browser_proxy_bypass,
        )

    def _get_or_create_row(self, session) -> ServiceConfigRow:
        row = session.get(ServiceConfigRow, 1)
        if row is None:
            row = ServiceConfigRow(
                id=1,
                default_model=self._defaults.default_model,
                api_token=self._defaults.api_token,
                browser_channel="chrome",
                browser_executable_path=None,
                browser_headless=self._defaults.browser_headless,
                browser_user_agent=self._defaults.browser_user_agent,
                browser_locale=self._defaults.browser_locale,
                browser_base_url=self._defaults.browser_base_url,
                browser_timeout_ms=self._defaults.browser_timeout_ms,
                answer_timeout_ms=self._defaults.answer_timeout_ms,
                browser_proxy_server=self._defaults.browser_proxy_server,
                browser_proxy_username=self._defaults.browser_proxy_username,
                browser_proxy_password=self._defaults.browser_proxy_password,
                browser_proxy_bypass=self._defaults.browser_proxy_bypass,
                updated_at=utc_now(),
            )
            session.add(row)
            session.commit()
            session.refresh(row)
        return row

    def get_config(self) -> ServiceConfig:
        with self._session_factory() as session:
            row = self._get_or_create_row(session)
            return self._row_to_config(row)

    def update_config(self, update: ServiceConfigUpdate) -> ServiceConfig:
        with self._session_factory() as session:
            row = self._get_or_create_row(session)
            row.default_model = update.default_model.strip()
            row.api_token = update.api_token.strip()
            row.browser_channel = "chrome"
            row.browser_executable_path = None
            row.browser_headless = update.browser_headless
            row.browser_user_agent = _coalesce_blank(update.browser_user_agent)
            row.browser_locale = update.browser_locale.strip()
            row.browser_base_url = update.browser_base_url.strip()
            row.browser_timeout_ms = update.browser_timeout_ms
            row.answer_timeout_ms = update.answer_timeout_ms
            row.browser_proxy_server = _coalesce_blank(update.browser_proxy_server)
            row.browser_proxy_username = _coalesce_blank(update.browser_proxy_username)
            row.browser_proxy_password = _coalesce_blank(update.browser_proxy_password)
            row.browser_proxy_bypass = _coalesce_blank(update.browser_proxy_bypass)
            row.updated_at = utc_now()
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._row_to_config(row)

    def start_request(
        self,
        *,
        endpoint: str,
        model_name: str,
        prompt_preview: str,
        client_ip: str | None,
        stream: bool,
        config: ServiceConfig,
    ) -> str:
        request_id = uuid.uuid4().hex
        with self._session_factory() as session:
            row = RequestLogRow(
                id=request_id,
                endpoint=endpoint,
                status="pending",
                model_name=model_name,
                prompt_preview=_sanitize_logged_text(prompt_preview, limit=2000),
                client_ip=client_ip,
                stream=stream,
                headless=config.browser_headless,
                proxy_enabled=config.proxy_enabled,
            )
            session.add(row)
            self._trim_request_logs(session)
            session.commit()
        return request_id

    def finish_request_success(
        self, request_id: str, result: GoogleAiResult, duration_ms: int
    ) -> None:
        with self._session_factory() as session:
            row = session.get(RequestLogRow, request_id)
            if row is None:
                return
            row.status = "ok"
            row.response_preview = _sanitize_logged_text(result.answer_text, limit=3000)
            row.final_url = _sanitize_logged_url(result.final_url)
            row.citations_json = json.dumps(
                [citation.model_dump() for citation in result.citations],
                ensure_ascii=False,
            )
            row.duration_ms = duration_ms
            row.finished_at = utc_now()
            session.add(row)
            self._trim_request_logs(session)
            session.commit()

    def finish_request_error(self, request_id: str, error_message: str, duration_ms: int) -> None:
        with self._session_factory() as session:
            row = session.get(RequestLogRow, request_id)
            if row is None:
                return
            row.status = "error"
            row.error_message = _sanitize_logged_text(error_message, limit=3000)
            row.duration_ms = duration_ms
            row.finished_at = utc_now()
            session.add(row)
            self._trim_request_logs(session)
            session.commit()

    def get_summary(self) -> DashboardSummary:
        with self._session_factory() as session:
            total_requests = session.scalar(select(func.count()).select_from(RequestLogRow)) or 0
            successful_requests = (
                session.scalar(
                    select(func.count())
                    .select_from(RequestLogRow)
                    .where(RequestLogRow.status == "ok")
                )
                or 0
            )
            failed_requests = (
                session.scalar(
                    select(func.count())
                    .select_from(RequestLogRow)
                    .where(RequestLogRow.status == "error")
                )
                or 0
            )
            average_latency_ms = session.scalar(
                select(func.avg(RequestLogRow.duration_ms)).where(RequestLogRow.status == "ok")
            )
            last_request = session.scalars(
                select(RequestLogRow).order_by(RequestLogRow.created_at.desc()).limit(1)
            ).first()
            last_success = session.scalars(
                select(RequestLogRow)
                .where(RequestLogRow.status == "ok")
                .order_by(RequestLogRow.finished_at.desc())
                .limit(1)
            ).first()
            return DashboardSummary(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_latency_ms=(
                    float(average_latency_ms) if average_latency_ms is not None else None
                ),
                last_request_at=last_request.created_at if last_request else None,
                last_success_at=last_success.finished_at if last_success else None,
            )

    def list_recent_requests(self, limit: int = 20) -> list[RecentRequest]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(RequestLogRow).order_by(RequestLogRow.created_at.desc()).limit(limit)
            ).all()
            records: list[RecentRequest] = []
            for row in rows:
                citations: list[Citation] = []
                if row.citations_json:
                    try:
                        citations = [Citation(**item) for item in json.loads(row.citations_json)]
                    except json.JSONDecodeError:
                        citations = []
                records.append(
                    RecentRequest(
                        id=row.id,
                        endpoint=row.endpoint,
                        status=row.status,
                        model_name=row.model_name,
                        prompt_preview=row.prompt_preview,
                        response_preview=row.response_preview,
                        error_message=row.error_message,
                        final_url=row.final_url,
                        duration_ms=row.duration_ms,
                        client_ip=row.client_ip,
                        created_at=row.created_at,
                        finished_at=row.finished_at,
                        citations=citations,
                    )
                )
            return records

    def _trim_request_logs(self, session) -> None:
        if self._request_log_max_rows < 1:
            return

        expired_ids = session.scalars(
            select(RequestLogRow.id)
            .order_by(RequestLogRow.created_at.desc(), RequestLogRow.id.desc())
            .offset(self._request_log_max_rows)
        ).all()
        if not expired_ids:
            return

        session.execute(delete(RequestLogRow).where(RequestLogRow.id.in_(expired_ids)))
