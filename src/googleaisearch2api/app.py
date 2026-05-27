from __future__ import annotations

import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode, urlsplit

import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import ValidationError

from .browser import (
    GoogleAiBlockedError,
    GoogleAiRuntimeError,
    GoogleAiTimeoutError,
)
from .config import (
    DEFAULT_API_TOKEN,
    AppSettings,
    ServiceConfig,
    ServiceConfigUpdate,
    get_settings,
)
from .db import create_db_engine, create_session_factory, create_tables
from .duck_ai import DuckAiRateLimitedError, DuckAiRunner, DuckAiRuntimeError, DuckAiTimeoutError
from .logging import configure_logging
from .openai_adapter import (
    OpenAICompatibilityError,
    build_chat_completion_response,
    build_prompt_from_messages,
    build_prompt_from_responses_request,
    build_responses_api_response,
    iter_chat_completion_stream,
    iter_responses_api_stream,
)
from .pool import (
    BrowserPool,
    BrowserPoolClosedError,
    BrowserPoolSaturatedError,
    BrowserPoolTimeoutError,
)
from .proxy_sessions import (
    ProxySessionConfigError,
    ProxySessionSelection,
    ProxySessionSelector,
    ProxySessionStore,
    ProxySessionUnavailableError,
    google_block_has_ip_mismatch,
    parse_google_block_ips,
)
from .query_adapter import (
    build_prompt_from_query_request,
    build_query_response,
    iter_query_stream,
)
from .schemas import ChatCompletionsRequest, QueryRequest, ResponsesRequest
from .store import ConfigStore

security = HTTPBearer(auto_error=False)
CONSOLE_SESSION_COOKIE = "googleaisearch2api_console_token"


@dataclass
class Services:
    settings: AppSettings
    store: ConfigStore
    pool: BrowserPool
    duck_pool: BrowserPool
    duck_circuit: DuckAiCircuitBreaker
    proxy_session_store: ProxySessionStore
    proxy_selector: ProxySessionSelector


class DuckAiCircuitBreaker:
    def __init__(self, cooldown_seconds: int) -> None:
        self._cooldown_seconds = max(cooldown_seconds, 0)
        self._open_until = 0.0
        self._lock = threading.Lock()

    def remaining_seconds(self) -> int:
        with self._lock:
            remaining = self._open_until - time.time()
        return max(int(remaining), 0)

    def record_rate_limited(self) -> None:
        if self._cooldown_seconds <= 0:
            return
        with self._lock:
            self._open_until = time.time() + self._cooldown_seconds

    def record_success(self) -> None:
        with self._lock:
            self._open_until = 0.0


def _templates() -> Jinja2Templates:
    return Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _static_dir() -> Path:
    return Path(__file__).parent / "static"


def create_services(settings: AppSettings) -> Services:
    _validate_settings(settings)
    configure_logging(settings.app_log_level)
    engine = create_db_engine(str(settings.db_path))
    create_tables(engine)
    session_factory = create_session_factory(engine)
    store = ConfigStore(
        session_factory,
        ServiceConfig.from_settings(settings),
        request_log_max_rows=settings.request_log_max_rows,
    )
    proxy_session_store = ProxySessionStore(session_factory)
    proxy_selector = ProxySessionSelector(
        proxy_session_store,
        allow_fallback_to_base=settings.proxy_allow_fallback_to_base,
    )
    pool = BrowserPool(
        worker_count=settings.max_concurrent_requests,
        queue_capacity=settings.request_queue_size,
        blocked_retry_count=settings.google_ai_blocked_retry_count,
    )
    duck_pool = BrowserPool(
        worker_count=settings.duck_ai_workers,
        queue_capacity=settings.duck_ai_queue_size,
        runner_factory=DuckAiRunner,
        blocked_retry_count=0,
    )
    return Services(
        settings=settings,
        store=store,
        pool=pool,
        duck_pool=duck_pool,
        duck_circuit=DuckAiCircuitBreaker(settings.duck_ai_cooldown_seconds),
        proxy_session_store=proxy_session_store,
        proxy_selector=proxy_selector,
    )


def get_services(request: Request) -> Services:
    return request.app.state.services


def _token_matches(expected_token: str, candidate_token: str | None) -> bool:
    if not expected_token:
        return True
    if not candidate_token:
        return False
    return secrets.compare_digest(candidate_token, expected_token)


def require_api_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> None:
    config = get_services(request).store.get_config()
    if not config.api_token:
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
        )
    if not _token_matches(config.api_token, credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token.",
        )


def _login_status_message(status_code: str | None) -> str | None:
    if status_code == "invalid":
        return "Console login failed."
    return None


def _status_message(status_code: str | None) -> str | None:
    if status_code == "saved":
        return "Configuration saved."
    if status_code == "probe-ok":
        return "Live browser probe succeeded."
    if status_code == "probe-failed":
        return "Live browser probe failed. Check the recent request table for details."
    return None


def _coerce_checkbox(value: str | None) -> bool:
    return value == "on"


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _validate_settings(settings: AppSettings) -> None:
    if _is_loopback_host(settings.app_host):
        return
    if not settings.api_token.strip() or settings.api_token == DEFAULT_API_TOKEN:
        raise RuntimeError(
            "API_TOKEN must be set to a non-default value before binding the service "
            f"to {settings.app_host}."
        )


def _resolve_model(requested_model: str | None, config: ServiceConfig) -> str:
    if requested_model is None or requested_model == config.default_model:
        return config.default_model
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Unknown model: {requested_model}",
    )


def _record_proxy_session_error(
    services: Services,
    selection: ProxySessionSelection | None,
    *,
    blocked: bool,
    error_message: str,
    block_ips: list[str] | None = None,
) -> None:
    if selection is None:
        return
    services.proxy_session_store.finish_request_error(
        selection.session.id,
        blocked=blocked,
        error_message=error_message,
        block_ips=block_ips,
    )


def _run_google_ai(
    *,
    request: Request,
    endpoint: str,
    prompt: str,
    stream: bool,
    requested_model: str | None,
) -> tuple[ServiceConfig, str, str, object]:
    services = get_services(request)
    config = services.store.get_config()
    model_name = _resolve_model(requested_model, config)
    max_attempts = (
        services.settings.google_ai_blocked_retry_count + 1
        if config.resin_sticky_session_enabled
        else 1
    )

    for attempt_index in range(max_attempts):
        selection: ProxySessionSelection | None = None
        effective_config = config
        try:
            selection = services.proxy_selector.select(config)
        except (ProxySessionConfigError, ProxySessionUnavailableError) as exc:
            request_id = services.store.start_request(
                endpoint=endpoint,
                engine="google",
                model_name=model_name,
                prompt_preview=prompt,
                client_ip=request.client.host if request.client else None,
                stream=stream,
                config=config,
            )
            services.store.finish_request_error(request_id, str(exc), 0)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        if selection is not None:
            effective_config = selection.config

        request_id = services.store.start_request(
            endpoint=endpoint,
            engine="google",
            model_name=model_name,
            prompt_preview=prompt,
            client_ip=request.client.host if request.client else None,
            stream=stream,
            config=effective_config,
            proxy_session_id=selection.session.id if selection else None,
            proxy_base_username=selection.session.proxy_base_username if selection else None,
            proxy_username=selection.session.proxy_username if selection else None,
            proxy_primary_ip=selection.session.primary_ip if selection else None,
            proxy_ip_vector_hash=selection.session.ip_vector_hash if selection else None,
            proxy_iplark_score=selection.session.iplark_min_quality_score if selection else None,
        )

        started_at = time.perf_counter()
        try:
            result = services.pool.execute(
                effective_config,
                prompt,
                blocked_retry_count=0 if selection is not None else None,
            )
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            services.store.finish_request_success(request_id, result, duration_ms)
            if selection is not None:
                services.proxy_session_store.finish_request_success(selection.session.id)
            return effective_config, model_name, request_id, result
        except BrowserPoolSaturatedError as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            services.store.finish_request_error(request_id, str(exc), duration_ms)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc
        except BrowserPoolClosedError as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            services.store.finish_request_error(request_id, str(exc), duration_ms)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except BrowserPoolTimeoutError as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            services.store.finish_request_error(request_id, str(exc), duration_ms)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(exc),
            ) from exc
        except GoogleAiBlockedError as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            block_ips = parse_google_block_ips(str(exc))
            services.store.finish_request_error(
                request_id,
                str(exc),
                duration_ms,
                google_block_ips=block_ips,
                google_block_mismatch=google_block_has_ip_mismatch(block_ips),
            )
            _record_proxy_session_error(
                services,
                selection,
                blocked=True,
                error_message=str(exc),
                block_ips=block_ips,
            )
            if selection is not None:
                services.pool.reset()
                if attempt_index + 1 < max_attempts:
                    logger.warning(
                        "Google blocked sticky proxy session {}; retrying request "
                        "with another session ({}/{})",
                        selection.session.proxy_username,
                        attempt_index + 1,
                        max_attempts - 1,
                    )
                    continue
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        except GoogleAiTimeoutError as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            services.store.finish_request_error(request_id, str(exc), duration_ms)
            _record_proxy_session_error(
                services,
                selection,
                blocked=False,
                error_message=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(exc),
            ) from exc
        except GoogleAiRuntimeError as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            services.store.finish_request_error(request_id, str(exc), duration_ms)
            _record_proxy_session_error(
                services,
                selection,
                blocked=False,
                error_message=str(exc),
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            services.store.finish_request_error(request_id, repr(exc), duration_ms)
            _record_proxy_session_error(
                services,
                selection,
                blocked=False,
                error_message=repr(exc),
            )
            logger.exception("Unhandled request failure")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unhandled Google AI request failure.",
            ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Google AI request retry loop exited unexpectedly.",
    )


def _run_duck_ai(
    *,
    request: Request,
    endpoint: str,
    prompt: str,
    stream: bool,
    requested_model: str | None,
) -> tuple[ServiceConfig, str, str, object]:
    services = get_services(request)
    config = services.store.get_config()
    model_name = _resolve_model(requested_model, config)
    selection: ProxySessionSelection | None = None
    effective_config = config

    cooldown_remaining = services.duck_circuit.remaining_seconds()
    if cooldown_remaining > 0:
        request_id = services.store.start_request(
            endpoint=endpoint,
            engine="duck",
            model_name=model_name,
            prompt_preview=prompt,
            client_ip=request.client.host if request.client else None,
            stream=stream,
            config=config,
        )
        message = f"Duck.ai is cooling down for {cooldown_remaining}s after rate limiting."
        services.store.finish_request_error(request_id, message, 0)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=message,
        )

    try:
        selection = services.proxy_selector.select(config)
    except (ProxySessionConfigError, ProxySessionUnavailableError) as exc:
        request_id = services.store.start_request(
            endpoint=endpoint,
            engine="duck",
            model_name=model_name,
            prompt_preview=prompt,
            client_ip=request.client.host if request.client else None,
            stream=stream,
            config=config,
        )
        services.store.finish_request_error(request_id, str(exc), 0)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if selection is not None:
        effective_config = selection.config

    request_id = services.store.start_request(
        endpoint=endpoint,
        engine="duck",
        model_name=model_name,
        prompt_preview=prompt,
        client_ip=request.client.host if request.client else None,
        stream=stream,
        config=effective_config,
        proxy_session_id=selection.session.id if selection else None,
        proxy_base_username=selection.session.proxy_base_username if selection else None,
        proxy_username=selection.session.proxy_username if selection else None,
        proxy_primary_ip=selection.session.primary_ip if selection else None,
        proxy_ip_vector_hash=selection.session.ip_vector_hash if selection else None,
        proxy_iplark_score=selection.session.iplark_min_quality_score if selection else None,
    )

    started_at = time.perf_counter()
    try:
        result = services.duck_pool.execute(
            effective_config,
            prompt,
            blocked_retry_count=0,
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_success(request_id, result, duration_ms)
        services.duck_circuit.record_success()
        if selection is not None:
            services.proxy_session_store.finish_request_success(selection.session.id)
        return effective_config, model_name, request_id, result
    except BrowserPoolSaturatedError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except BrowserPoolClosedError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except BrowserPoolTimeoutError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except DuckAiRateLimitedError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.duck_circuit.record_rate_limited()
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        _record_proxy_session_error(
            services,
            selection,
            blocked=False,
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except DuckAiTimeoutError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        _record_proxy_session_error(
            services,
            selection,
            blocked=False,
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except DuckAiRuntimeError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        _record_proxy_session_error(
            services,
            selection,
            blocked=False,
            error_message=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, repr(exc), duration_ms)
        _record_proxy_session_error(
            services,
            selection,
            blocked=False,
            error_message=repr(exc),
        )
        logger.exception("Unhandled Duck.ai request failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unhandled Duck.ai request failure.",
        ) from exc


def _should_try_duck_fallback(exc: HTTPException) -> bool:
    return exc.status_code in {
        status.HTTP_502_BAD_GATEWAY,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        status.HTTP_504_GATEWAY_TIMEOUT,
    }


def _run_search_ai(
    *,
    request: Request,
    endpoint: str,
    prompt: str,
    stream: bool,
    requested_model: str | None,
) -> tuple[ServiceConfig, str, str, object]:
    services = get_services(request)
    config = services.store.get_config()
    if config.search_engine == "duck":
        return _run_duck_ai(
            request=request,
            endpoint=endpoint,
            prompt=prompt,
            stream=stream,
            requested_model=requested_model,
        )
    if config.search_engine != "auto":
        return _run_google_ai(
            request=request,
            endpoint=endpoint,
            prompt=prompt,
            stream=stream,
            requested_model=requested_model,
        )

    try:
        return _run_google_ai(
            request=request,
            endpoint=endpoint,
            prompt=prompt,
            stream=stream,
            requested_model=requested_model,
        )
    except HTTPException as google_exc:
        if not _should_try_duck_fallback(google_exc):
            raise
        logger.warning(
            "Google engine failed with {}; trying Duck.ai fallback: {}",
            google_exc.status_code,
            google_exc.detail,
        )
        try:
            return _run_duck_ai(
                request=request,
                endpoint=endpoint,
                prompt=prompt,
                stream=stream,
                requested_model=requested_model,
            )
        except HTTPException as duck_exc:
            detail = (
                "Both search engines failed. "
                f"Google: {google_exc.detail}; Duck.ai: {duck_exc.detail}"
            )
            raise HTTPException(
                status_code=duck_exc.status_code,
                detail=detail,
            ) from duck_exc


def _safe_next_target(target: str | None) -> str:
    if not target:
        return "/console"
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/"):
        return "/console"
    return target


def _console_login_url(request: Request) -> str:
    next_target = _safe_next_target(
        f"{request.url.path}?{request.url.query}" if request.url.query else request.url.path
    )
    return f"/console/login?{urlencode({'next': next_target})}"


def _has_console_session(request: Request, config: ServiceConfig) -> bool:
    if not config.api_token:
        return True
    return _token_matches(config.api_token, request.cookies.get(CONSOLE_SESSION_COOKIE))


def _set_console_cookie(response: RedirectResponse, request: Request, token: str) -> None:
    response.set_cookie(
        key=CONSOLE_SESSION_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/console",
    )


def _clear_console_cookie(response: RedirectResponse) -> None:
    response.delete_cookie(CONSOLE_SESSION_COOKIE, path="/console")


def create_app() -> FastAPI:
    settings = get_settings()
    templates = _templates()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = create_services(settings)
        app.state.services = services
        try:
            yield
        finally:
            services.pool.close()
            services.duck_pool.close()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(_static_dir())), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/console", status_code=status.HTTP_302_FOUND)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/healthz")
    def healthz(request: Request) -> dict:
        services = get_services(request)
        config = services.store.get_config()
        pool_summary = services.pool.get_summary()
        duck_pool_summary = services.duck_pool.get_summary()
        return {
            "ok": True,
            "service": settings.app_name,
            "model": config.default_model,
            "search_engine": config.search_engine,
            "browser": config.browser_label,
            "proxy_enabled": config.proxy_enabled,
            "resin_sticky_session_enabled": config.resin_sticky_session_enabled,
            "headless": config.browser_headless,
            "workers": pool_summary.worker_count,
            "busy_workers": pool_summary.busy_workers,
            "queued_requests": pool_summary.queued_requests,
            "queue_capacity": pool_summary.queue_capacity,
            "workers_with_errors": pool_summary.workers_with_errors,
            "duck_ai_workers": duck_pool_summary.worker_count,
            "duck_ai_busy_workers": duck_pool_summary.busy_workers,
            "duck_ai_queued_requests": duck_pool_summary.queued_requests,
            "duck_ai_cooldown_remaining_seconds": services.duck_circuit.remaining_seconds(),
            "accepting_requests": pool_summary.accepting_requests,
        }

    @app.get("/console/login", response_class=HTMLResponse, include_in_schema=False)
    def console_login(
        request: Request,
        next: str | None = None,
        status_code: str | None = None,
    ):
        services = get_services(request)
        config = services.store.get_config()
        next_target = _safe_next_target(next)
        if _has_console_session(request, config):
            return RedirectResponse(url=next_target, status_code=status.HTTP_303_SEE_OTHER)
        return templates.TemplateResponse(
            request=request,
            name="console_login.html",
            context={
                "request": request,
                "next_target": next_target,
                "status_message": _login_status_message(status_code),
            },
        )

    @app.post("/console/login", include_in_schema=False)
    def submit_console_login(
        request: Request,
        console_token: str = Form(...),
        next: str = Form("/console"),
    ) -> RedirectResponse:
        services = get_services(request)
        config = services.store.get_config()
        next_target = _safe_next_target(next)
        if not _token_matches(config.api_token, console_token.strip()):
            return RedirectResponse(
                url=f"/console/login?{urlencode({'next': next_target, 'status_code': 'invalid'})}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        response = RedirectResponse(url=next_target, status_code=status.HTTP_303_SEE_OTHER)
        _set_console_cookie(response, request, config.api_token)
        return response

    @app.post("/console/logout", include_in_schema=False)
    def console_logout() -> RedirectResponse:
        response = RedirectResponse(url="/console/login", status_code=status.HTTP_303_SEE_OTHER)
        _clear_console_cookie(response)
        return response

    @app.get("/v1/models", dependencies=[Depends(require_api_token)])
    def list_models(request: Request) -> dict:
        config = get_services(request).store.get_config()
        return {
            "object": "list",
            "data": [
                {
                    "id": config.default_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "googleaisearch2api",
                }
            ],
        }

    def _query_response(payload: QueryRequest, request: Request):
        prompt = build_prompt_from_query_request(payload)
        _, model_name, request_id, result = _run_search_ai(
            request=request,
            endpoint="/query",
            prompt=prompt,
            stream=payload.stream,
            requested_model=payload.model,
        )
        if payload.stream:
            return StreamingResponse(
                iter_query_stream(
                    result=result,
                    model_name=model_name,
                    prompt=prompt,
                    request_id=request_id,
                    payload=payload,
                ),
                media_type="text/event-stream",
            )
        return build_query_response(
            result=result,
            model_name=model_name,
            prompt=prompt,
            request_id=request_id,
            payload=payload,
        )

    @app.get("/query", dependencies=[Depends(require_api_token)])
    def query_get(
        request: Request,
        q: str = Query(..., min_length=1),
        model: str | None = Query(None),
        stream: bool = Query(False),
        include_citations: bool = Query(True),
        include_google_metadata: bool = Query(True),
    ):
        try:
            payload = QueryRequest(
                query=q,
                model=model,
                stream=stream,
                include_citations=include_citations,
                include_google_metadata=include_google_metadata,
            )
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        return _query_response(payload, request)

    @app.post("/query", dependencies=[Depends(require_api_token)])
    def query_post(payload: QueryRequest, request: Request):
        return _query_response(payload, request)

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_token)])
    def chat_completions(payload: ChatCompletionsRequest, request: Request):
        try:
            prompt = build_prompt_from_messages(payload.messages)
        except OpenAICompatibilityError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        _, model_name, _, result = _run_search_ai(
            request=request,
            endpoint="/v1/chat/completions",
            prompt=prompt,
            stream=payload.stream,
            requested_model=payload.model,
        )
        if payload.stream:
            return StreamingResponse(
                iter_chat_completion_stream(result=result, model_name=model_name),
                media_type="text/event-stream",
            )
        return build_chat_completion_response(result=result, model_name=model_name, prompt=prompt)

    @app.post("/v1/responses", dependencies=[Depends(require_api_token)])
    def responses_api(payload: ResponsesRequest, request: Request):
        try:
            prompt = build_prompt_from_responses_request(payload)
        except OpenAICompatibilityError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        _, model_name, _, result = _run_search_ai(
            request=request,
            endpoint="/v1/responses",
            prompt=prompt,
            stream=payload.stream,
            requested_model=payload.model,
        )
        if payload.stream:
            return StreamingResponse(
                iter_responses_api_stream(result=result, model_name=model_name),
                media_type="text/event-stream",
            )
        return build_responses_api_response(result=result, model_name=model_name, prompt=prompt)

    @app.get("/console", response_class=HTMLResponse, include_in_schema=False)
    def console(request: Request, status_code: str | None = None):
        services = get_services(request)
        config = services.store.get_config()
        if not _has_console_session(request, config):
            return RedirectResponse(
                url=_console_login_url(request),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        summary = services.store.get_summary()
        pool_summary = services.pool.get_summary()
        duck_pool_summary = services.duck_pool.get_summary()
        proxy_session_summary = services.proxy_session_store.get_summary()
        proxy_sessions = services.proxy_session_store.list_proxy_sessions(limit=20)
        recent_requests = services.store.list_recent_requests(limit=20)
        return templates.TemplateResponse(
            request=request,
            name="console.html",
            context={
                "request": request,
                "config": config,
                "summary": summary,
                "pool_summary": pool_summary,
                "duck_pool_summary": duck_pool_summary,
                "duck_cooldown_remaining_seconds": services.duck_circuit.remaining_seconds(),
                "proxy_session_summary": proxy_session_summary,
                "proxy_sessions": proxy_sessions,
                "recent_requests": recent_requests,
                "status_message": _status_message(status_code),
            },
        )

    @app.get("/console/summary.json", include_in_schema=False)
    def console_summary(request: Request):
        services = get_services(request)
        config = services.store.get_config()
        if not _has_console_session(request, config):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Console login required.",
            )
        summary = services.store.get_summary()
        pool_summary = services.pool.get_summary()
        duck_pool_summary = services.duck_pool.get_summary()
        proxy_session_summary = services.proxy_session_store.get_summary()
        return {
            "summary": summary.model_dump(mode="json"),
            "pool": pool_summary.model_dump(mode="json"),
            "duck_pool": duck_pool_summary.model_dump(mode="json"),
            "duck_cooldown_remaining_seconds": services.duck_circuit.remaining_seconds(),
            "proxy_sessions": proxy_session_summary,
            "config": {
                "default_model": config.default_model,
                "search_engine": config.search_engine,
                "browser_label": config.browser_label,
                "proxy_enabled": config.proxy_enabled,
                "resin_sticky_session_enabled": config.resin_sticky_session_enabled,
                "browser_headless": config.browser_headless,
            },
        }

    @app.post("/console/settings", include_in_schema=False)
    def update_settings(
        request: Request,
        default_model: str = Form(...),
        search_engine: str = Form("google"),
        api_token: str = Form(""),
        browser_headless: str | None = Form(None),
        browser_user_agent: str = Form(""),
        browser_locale: str = Form(...),
        browser_base_url: str = Form(...),
        browser_timeout_ms: int = Form(...),
        answer_timeout_ms: int = Form(...),
        browser_proxy_server: str = Form(""),
        browser_proxy_username: str = Form(""),
        browser_proxy_password: str = Form(""),
        clear_browser_proxy_password: str | None = Form(None),
        browser_proxy_bypass: str = Form(""),
        resin_sticky_session_enabled: str | None = Form(None),
    ) -> RedirectResponse:
        services = get_services(request)
        current_config = services.store.get_config()
        if not _has_console_session(request, current_config):
            return RedirectResponse(
                url=_console_login_url(request),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        update = ServiceConfigUpdate(
            default_model=default_model,
            search_engine=search_engine,
            api_token=api_token.strip() or current_config.api_token,
            browser_headless=_coerce_checkbox(browser_headless),
            browser_user_agent=browser_user_agent,
            browser_locale=browser_locale,
            browser_base_url=browser_base_url,
            browser_timeout_ms=browser_timeout_ms,
            answer_timeout_ms=answer_timeout_ms,
            browser_proxy_server=browser_proxy_server,
            browser_proxy_username=browser_proxy_username,
            browser_proxy_password=(
                ""
                if _coerce_checkbox(clear_browser_proxy_password)
                else browser_proxy_password.strip() or current_config.browser_proxy_password or ""
            ),
            browser_proxy_bypass=browser_proxy_bypass,
            resin_sticky_session_enabled=_coerce_checkbox(resin_sticky_session_enabled),
        )
        updated_config = services.store.update_config(update)
        services.pool.reset()
        services.duck_pool.reset()
        response = RedirectResponse(
            url="/console?status_code=saved",
            status_code=status.HTTP_303_SEE_OTHER,
        )
        _set_console_cookie(response, request, updated_config.api_token)
        return response

    @app.post("/console/actions/probe", include_in_schema=False)
    def run_probe(
        request: Request,
        probe_prompt: str = Form(
            "What is the difference between Responses API and Chat Completions API?"
        ),
    ) -> RedirectResponse:
        services = get_services(request)
        config = services.store.get_config()
        if not _has_console_session(request, config):
            return RedirectResponse(
                url=_console_login_url(request),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            _run_search_ai(
                request=request,
                endpoint="/console/probe",
                prompt=probe_prompt,
                stream=False,
                requested_model=None,
            )
            code = "probe-ok"
        except HTTPException:
            code = "probe-failed"
        return RedirectResponse(
            url=f"/console?status_code={code}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    return app


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "googleaisearch2api.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )


app = create_app()
