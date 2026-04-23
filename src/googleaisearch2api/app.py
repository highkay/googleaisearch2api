from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from .browser import (
    GoogleAiBlockedError,
    GoogleAiRuntimeError,
    GoogleAiTimeoutError,
)
from .config import AppSettings, ServiceConfig, ServiceConfigUpdate, get_settings
from .db import create_db_engine, create_session_factory, create_tables
from .logging import configure_logging
from .openai_adapter import (
    build_chat_completion_response,
    build_prompt_from_messages,
    build_prompt_from_responses_request,
    build_responses_api_response,
    iter_chat_completion_stream,
    iter_responses_api_stream,
)
from .pool import BrowserPool, BrowserPoolClosedError, BrowserPoolSaturatedError
from .schemas import ChatCompletionsRequest, ResponsesRequest
from .store import ConfigStore

security = HTTPBearer(auto_error=False)


@dataclass
class Services:
    settings: AppSettings
    store: ConfigStore
    pool: BrowserPool


def _templates() -> Jinja2Templates:
    return Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _static_dir() -> Path:
    return Path(__file__).parent / "static"


def create_services(settings: AppSettings) -> Services:
    configure_logging(settings.app_log_level)
    engine = create_db_engine(str(settings.db_path))
    create_tables(engine)
    session_factory = create_session_factory(engine)
    store = ConfigStore(session_factory, ServiceConfig.from_settings(settings))
    pool = BrowserPool(
        worker_count=settings.max_concurrent_requests,
        queue_capacity=settings.request_queue_size,
    )
    return Services(settings=settings, store=store, pool=pool)


def get_services(request: Request) -> Services:
    return request.app.state.services


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
    if credentials.credentials != config.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token.",
        )


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


def _resolve_model(requested_model: str | None, config: ServiceConfig) -> str:
    if requested_model is None or requested_model == config.default_model:
        return config.default_model
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Unknown model: {requested_model}",
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
    request_id = services.store.start_request(
        endpoint=endpoint,
        model_name=model_name,
        prompt_preview=prompt,
        client_ip=request.client.host if request.client else None,
        stream=stream,
        config=config,
    )

    started_at = time.perf_counter()
    try:
        result = services.pool.execute(config, prompt)
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_success(request_id, result, duration_ms)
        return config, model_name, request_id, result
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
    except GoogleAiBlockedError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except GoogleAiTimeoutError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except GoogleAiRuntimeError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, str(exc), duration_ms)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        services.store.finish_request_error(request_id, repr(exc), duration_ms)
        logger.exception("Unhandled request failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unhandled Google AI request failure.",
        ) from exc


def create_app() -> FastAPI:
    settings = get_settings()
    services = create_services(settings)
    templates = _templates()

    app = FastAPI(title=settings.app_name)
    app.state.services = services
    app.mount("/static", StaticFiles(directory=str(_static_dir())), name="static")
    app.router.on_shutdown.append(services.pool.close)

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
        return {
            "ok": True,
            "service": settings.app_name,
            "model": config.default_model,
            "browser": config.browser_label,
            "proxy_enabled": config.proxy_enabled,
            "headless": config.browser_headless,
            "workers": pool_summary.worker_count,
            "busy_workers": pool_summary.busy_workers,
            "queued_requests": pool_summary.queued_requests,
            "queue_capacity": pool_summary.queue_capacity,
            "workers_with_errors": pool_summary.workers_with_errors,
            "accepting_requests": pool_summary.accepting_requests,
        }

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

    @app.post("/v1/chat/completions", dependencies=[Depends(require_api_token)])
    def chat_completions(payload: ChatCompletionsRequest, request: Request):
        prompt = build_prompt_from_messages(payload.messages)
        _, model_name, _, result = _run_google_ai(
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
        prompt = build_prompt_from_responses_request(payload)
        _, model_name, _, result = _run_google_ai(
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
        summary = services.store.get_summary()
        pool_summary = services.pool.get_summary()
        recent_requests = services.store.list_recent_requests(limit=20)
        return templates.TemplateResponse(
            request=request,
            name="console.html",
            context={
                "request": request,
                "config": config,
                "summary": summary,
                "pool_summary": pool_summary,
                "recent_requests": recent_requests,
                "status_message": _status_message(status_code),
            },
        )

    @app.get("/console/summary.json", include_in_schema=False)
    def console_summary(request: Request):
        services = get_services(request)
        config = services.store.get_config()
        summary = services.store.get_summary()
        pool_summary = services.pool.get_summary()
        return {
            "summary": summary.model_dump(mode="json"),
            "pool": pool_summary.model_dump(mode="json"),
            "config": {
                "default_model": config.default_model,
                "browser_label": config.browser_label,
                "proxy_enabled": config.proxy_enabled,
                "browser_headless": config.browser_headless,
            },
        }

    @app.post("/console/settings", include_in_schema=False)
    def update_settings(
        request: Request,
        default_model: str = Form(...),
        api_token: str = Form(...),
        browser_headless: str | None = Form(None),
        browser_user_agent: str = Form(""),
        browser_locale: str = Form(...),
        browser_base_url: str = Form(...),
        browser_timeout_ms: int = Form(...),
        answer_timeout_ms: int = Form(...),
        browser_proxy_server: str = Form(""),
        browser_proxy_username: str = Form(""),
        browser_proxy_password: str = Form(""),
        browser_proxy_bypass: str = Form(""),
    ) -> RedirectResponse:
        update = ServiceConfigUpdate(
            default_model=default_model,
            api_token=api_token,
            browser_headless=_coerce_checkbox(browser_headless),
            browser_user_agent=browser_user_agent,
            browser_locale=browser_locale,
            browser_base_url=browser_base_url,
            browser_timeout_ms=browser_timeout_ms,
            answer_timeout_ms=answer_timeout_ms,
            browser_proxy_server=browser_proxy_server,
            browser_proxy_username=browser_proxy_username,
            browser_proxy_password=browser_proxy_password,
            browser_proxy_bypass=browser_proxy_bypass,
        )
        services = get_services(request)
        services.store.update_config(update)
        services.pool.reset()
        return RedirectResponse(
            url="/console?status_code=saved",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/console/actions/probe", include_in_schema=False)
    def run_probe(
        request: Request,
        probe_prompt: str = Form(
            "What is the difference between Responses API and Chat Completions API?"
        ),
    ) -> RedirectResponse:
        try:
            _run_google_ai(
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
