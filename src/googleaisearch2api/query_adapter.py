from __future__ import annotations

import json
import time
from collections.abc import Iterable

from .openai_adapter import estimate_tokens
from .schemas import GoogleAiResult, QueryRequest


def build_prompt_from_query_request(payload: QueryRequest) -> str:
    parts: list[str] = []
    if payload.instructions and payload.instructions.strip():
        parts.append("System instructions:\n" + payload.instructions.strip())

    if isinstance(payload.context, str):
        context = payload.context.strip()
        if context:
            parts.append("Context:\n" + context)
    elif payload.context:
        transcript: list[str] = []
        for item in payload.context:
            transcript.append(f"{item.role.upper()}: {item.content.strip()}")
        if transcript:
            parts.append("Conversation context:\n" + "\n\n".join(transcript))

    parts.append("User request:\n" + payload.query.strip())
    return "\n\n".join(parts).strip()


def _chunk_text(text: str, target_size: int = 120) -> list[str]:
    if len(text) <= target_size:
        return [text]
    chunks: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= target_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return chunks or [text]


def build_query_response(
    *,
    result: GoogleAiResult,
    model_name: str,
    prompt: str,
    request_id: str,
    payload: QueryRequest,
) -> dict:
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(result.answer_text)
    response = {
        "id": f"query-{request_id}",
        "object": "query.result",
        "created": int(time.time()),
        "model": model_name,
        "answer": result.answer_text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if payload.include_citations:
        response["citations"] = [citation.model_dump() for citation in result.citations]
    if payload.include_google_metadata:
        response["google_ai"] = {
            "final_url": result.final_url,
            "page_title": result.page_title,
            "body_excerpt": result.body_excerpt,
        }
    return response


def iter_query_stream(
    *,
    result: GoogleAiResult,
    model_name: str,
    prompt: str,
    request_id: str,
    payload: QueryRequest,
) -> Iterable[str]:
    response_id = f"query-{request_id}"
    created = {
        "id": response_id,
        "object": "query.result",
        "created": int(time.time()),
        "model": model_name,
        "status": "in_progress",
    }
    yield f"event: query.created\ndata: {json.dumps(created, ensure_ascii=False)}\n\n"

    for chunk in _chunk_text(result.answer_text):
        delta = {"id": response_id, "delta": chunk}
        yield f"event: answer.delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n"

    completed = build_query_response(
        result=result,
        model_name=model_name,
        prompt=prompt,
        request_id=request_id,
        payload=payload,
    )
    completed["status"] = "completed"
    yield f"event: query.completed\ndata: {json.dumps(completed, ensure_ascii=False)}\n\n"
