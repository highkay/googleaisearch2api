from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Citation(BaseModel):
    title: str
    url: str


class GoogleAiResult(BaseModel):
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    final_url: str
    page_title: str
    body_excerpt: str = ""


class RecentRequest(BaseModel):
    id: str
    endpoint: str
    status: str
    model_name: str
    prompt_preview: str
    response_preview: str
    error_message: str | None = None
    final_url: str | None = None
    duration_ms: int | None = None
    client_ip: str | None = None
    created_at: datetime
    finished_at: datetime | None = None
    citations: list[Citation] = Field(default_factory=list)


class DashboardSummary(BaseModel):
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float | None = None
    last_request_at: datetime | None = None
    last_success_at: datetime | None = None


class RuntimePoolSummary(BaseModel):
    worker_count: int = 0
    busy_workers: int = 0
    idle_workers: int = 0
    queued_requests: int = 0
    queue_capacity: int = 0
    accepting_requests: bool = True
    generation: int = 0
    workers_with_errors: int = 0


class ChatContentPart(BaseModel):
    type: str = "text"
    text: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ChatContentPart]
    name: str | None = None


class ChatCompletionsRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None


class ResponseInputPart(BaseModel):
    type: str = "input_text"
    text: str | None = None


class ResponseInputItem(BaseModel):
    role: str = "user"
    content: str | list[ResponseInputPart]


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: str | list[ResponseInputItem]
    instructions: str | None = None
    stream: bool = False
    user: str | None = None
