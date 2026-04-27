from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

CHAT_TEXT_PART_TYPES = {"text"}
CHAT_ROLES = {"system", "developer", "user", "assistant"}
RESPONSE_INPUT_PART_TYPES = {"input_text"}
RESPONSE_INPUT_ROLES = {"system", "developer", "user", "assistant"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(StrictModel):
    title: str
    url: str


class GoogleAiResult(StrictModel):
    answer_text: str
    citations: list[Citation] = Field(default_factory=list)
    final_url: str
    page_title: str
    body_excerpt: str = ""


class RecentRequest(StrictModel):
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


class DashboardSummary(StrictModel):
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float | None = None
    last_request_at: datetime | None = None
    last_success_at: datetime | None = None


class RuntimePoolSummary(StrictModel):
    worker_count: int = 0
    busy_workers: int = 0
    idle_workers: int = 0
    queued_requests: int = 0
    queue_capacity: int = 0
    accepting_requests: bool = True
    generation: int = 0
    workers_with_errors: int = 0


class ChatContentPart(StrictModel):
    type: str = "text"
    text: str | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in CHAT_TEXT_PART_TYPES:
            raise ValueError("Only text chat content parts are supported by the browser backend.")
        return normalized


class ChatMessage(StrictModel):
    role: str
    content: str | list[ChatContentPart]
    name: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in CHAT_ROLES:
            raise ValueError(
                "Only system, developer, user, and assistant chat roles are supported."
            )
        return normalized


class ChatCompletionsRequest(StrictModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None


class ResponseInputPart(StrictModel):
    type: str = "input_text"
    text: str | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in RESPONSE_INPUT_PART_TYPES:
            raise ValueError(
                "Only input_text response content parts are supported by the browser backend."
            )
        return normalized


class ResponseInputItem(StrictModel):
    role: str = "user"
    content: str | list[ResponseInputPart]

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.lower().strip()
        if normalized not in RESPONSE_INPUT_ROLES:
            raise ValueError(
                "Only system, developer, user, and assistant response roles are supported."
            )
        return normalized


class ResponsesRequest(StrictModel):
    model: str | None = None
    input: str | list[ResponseInputItem]
    instructions: str | None = None
    stream: bool = False
    user: str | None = None
