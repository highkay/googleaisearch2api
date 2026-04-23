from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Iterable

from .schemas import ChatMessage, GoogleAiResult, ResponsesRequest


def flatten_message_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                texts.append(text.strip())
        return "\n".join(part for part in texts if part).strip()
    return ""


def build_prompt_from_messages(messages: list[ChatMessage]) -> str:
    system_messages: list[str] = []
    transcript: list[str] = []
    for message in messages:
        text = flatten_message_content(message.content)
        if not text:
            continue
        role = message.role.lower()
        if role == "system":
            system_messages.append(text)
        else:
            transcript.append(f"{role.upper()}: {text}")

    if not transcript:
        raise ValueError("No non-system messages were provided.")

    parts: list[str] = []
    if system_messages:
        parts.append("System instructions:\n" + "\n\n".join(system_messages))
    if len(transcript) == 1 and transcript[0].startswith("USER: "):
        parts.append("User request:\n" + transcript[0][6:])
    else:
        parts.append("Conversation transcript:\n" + "\n\n".join(transcript))
        parts.append("Respond to the latest user request.")
    return "\n\n".join(parts).strip()


def build_prompt_from_responses_request(payload: ResponsesRequest) -> str:
    parts: list[str] = []
    if payload.instructions:
        parts.append("System instructions:\n" + payload.instructions.strip())

    if isinstance(payload.input, str):
        parts.append("User request:\n" + payload.input.strip())
        return "\n\n".join(part for part in parts if part).strip()

    transcript: list[str] = []
    for item in payload.input:
        text = flatten_message_content(item.content)
        if text:
            transcript.append(f"{item.role.upper()}: {text}")
    if not transcript:
        raise ValueError("No response input text was provided.")
    parts.append("Conversation transcript:\n" + "\n\n".join(transcript))
    parts.append("Respond to the latest user request.")
    return "\n\n".join(parts).strip()


def estimate_tokens(text: str) -> int:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


def _chunk_text(text: str, target_size: int = 120) -> list[str]:
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?。！？])\s+", text) if chunk.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= target_size:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
        current = sentence
    if current:
        chunks.append(current)
    if chunks:
        return chunks
    return [text]


def build_chat_completion_response(
    *,
    result: GoogleAiResult,
    model_name: str,
    prompt: str,
) -> dict:
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(result.answer_text)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.answer_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "citations": [citation.model_dump() for citation in result.citations],
        "google_ai": {
            "final_url": result.final_url,
            "page_title": result.page_title,
        },
    }


def iter_chat_completion_stream(*, result: GoogleAiResult, model_name: str) -> Iterable[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    role_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n"

    for chunk in _chunk_text(result.answer_text):
        content_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(content_chunk, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "citations": [citation.model_dump() for citation in result.citations],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def build_responses_api_response(*, result: GoogleAiResult, model_name: str, prompt: str) -> dict:
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(result.answer_text)
    message_id = f"msg_{uuid.uuid4().hex}"
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_name,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.answer_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": result.answer_text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "citations": [citation.model_dump() for citation in result.citations],
        "google_ai": {
            "final_url": result.final_url,
            "page_title": result.page_title,
        },
    }


def iter_responses_api_stream(*, result: GoogleAiResult, model_name: str) -> Iterable[str]:
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created = {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "status": "in_progress",
            "model": model_name,
        },
    }
    yield f"event: response.created\ndata: {json.dumps(created, ensure_ascii=False)}\n\n"

    for chunk in _chunk_text(result.answer_text):
        delta = {
            "type": "response.output_text.delta",
            "response_id": response_id,
            "item_id": message_id,
            "delta": chunk,
        }
        payload = json.dumps(delta, ensure_ascii=False)
        yield f"event: response.output_text.delta\ndata: {payload}\n\n"

    completed = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "model": model_name,
            "output_text": result.answer_text,
            "citations": [citation.model_dump() for citation in result.citations],
        },
    }
    yield f"event: response.completed\ndata: {json.dumps(completed, ensure_ascii=False)}\n\n"
