from __future__ import annotations

import json
import uuid

from sqlalchemy import func, select

from .config import ServiceConfig, ServiceConfigUpdate
from .db import RequestLogRow, ServiceConfigRow, utc_now
from .schemas import Citation, DashboardSummary, GoogleAiResult, RecentRequest


def _coalesce_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class ConfigStore:
    def __init__(self, session_factory, defaults: ServiceConfig):
        self._session_factory = session_factory
        self._defaults = defaults

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
                prompt_preview=prompt_preview[:2000],
                client_ip=client_ip,
                stream=stream,
                headless=config.browser_headless,
                proxy_enabled=config.proxy_enabled,
            )
            session.add(row)
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
            row.response_preview = result.answer_text[:3000]
            row.final_url = result.final_url
            row.citations_json = json.dumps(
                [citation.model_dump() for citation in result.citations],
                ensure_ascii=False,
            )
            row.duration_ms = duration_ms
            row.finished_at = utc_now()
            session.add(row)
            session.commit()

    def finish_request_error(self, request_id: str, error_message: str, duration_ms: int) -> None:
        with self._session_factory() as session:
            row = session.get(RequestLogRow, request_id)
            if row is None:
                return
            row.status = "error"
            row.error_message = error_message[:3000]
            row.duration_ms = duration_ms
            row.finished_at = utc_now()
            session.add(row)
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
