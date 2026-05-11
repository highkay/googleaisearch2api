from googleaisearch2api.query_adapter import (
    build_prompt_from_query_request,
    build_query_response,
    iter_query_stream,
)
from googleaisearch2api.schemas import Citation, GoogleAiResult, QueryRequest


def test_build_prompt_from_query_request_supports_instructions_and_context() -> None:
    payload = QueryRequest(
        query="Summarize the result",
        instructions="Use verified facts only.",
        context=[
            {"role": "developer", "content": "Use three bullets."},
            {"role": "assistant", "content": "Previous answer"},
        ],
    )

    prompt = build_prompt_from_query_request(payload)

    assert "System instructions" in prompt
    assert "Use verified facts only." in prompt
    assert "DEVELOPER: Use three bullets." in prompt
    assert "ASSISTANT: Previous answer" in prompt
    assert "User request:\nSummarize the result" in prompt


def test_query_response_contains_tool_friendly_answer_usage_and_citations() -> None:
    result = GoogleAiResult(
        answer_text="Tool friendly answer.",
        citations=[Citation(title="Source", url="https://example.com")],
        final_url="https://www.google.com/search?udm=50",
        page_title="Google Search",
        body_excerpt="Excerpt",
    )
    payload = QueryRequest(query="Question")

    response = build_query_response(
        result=result,
        model_name="google-search",
        prompt="User request:\nQuestion",
        request_id="abc123",
        payload=payload,
    )

    assert response["id"] == "query-abc123"
    assert response["object"] == "query.result"
    assert response["answer"] == "Tool friendly answer."
    assert response["citations"][0]["url"] == "https://example.com"
    assert response["usage"]["total_tokens"] >= response["usage"]["input_tokens"]
    assert response["google_ai"]["page_title"] == "Google Search"


def test_query_response_can_hide_optional_payload_sections() -> None:
    result = GoogleAiResult(
        answer_text="Tool friendly answer.",
        citations=[Citation(title="Source", url="https://example.com")],
        final_url="https://www.google.com/search?udm=50",
        page_title="Google Search",
    )
    payload = QueryRequest(
        query="Question",
        include_citations=False,
        include_google_metadata=False,
    )

    response = build_query_response(
        result=result,
        model_name="google-search",
        prompt="User request:\nQuestion",
        request_id="abc123",
        payload=payload,
    )

    assert "citations" not in response
    assert "google_ai" not in response


def test_query_stream_uses_simple_sse_events() -> None:
    result = GoogleAiResult(
        answer_text="Streaming answer.",
        citations=[],
        final_url="https://www.google.com/search?udm=50",
        page_title="Google Search",
    )
    payload = QueryRequest(query="Question", stream=True)

    chunks = list(
        iter_query_stream(
            result=result,
            model_name="google-search",
            prompt="User request:\nQuestion",
            request_id="abc123",
            payload=payload,
        )
    )

    joined = "".join(chunks)
    assert "event: query.created" in joined
    assert "event: answer.delta" in joined
    assert '"delta": "Streaming answer."' in joined
    assert "event: query.completed" in joined
    assert '"answer": "Streaming answer."' in joined
