import json

from googleaisearch2api.openai_adapter import (
    build_chat_completion_response,
    build_prompt_from_messages,
    build_prompt_from_responses_request,
    iter_chat_completion_stream,
    iter_responses_api_stream,
)
from googleaisearch2api.schemas import (
    ChatMessage,
    Citation,
    GoogleAiResult,
    ResponseInputItem,
    ResponseInputPart,
    ResponsesRequest,
)


def test_build_prompt_from_messages_keeps_system_and_transcript() -> None:
    prompt = build_prompt_from_messages(
        [
            ChatMessage(role="developer", content="Be concise."),
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="First answer"),
            ChatMessage(role="user", content="Second question"),
        ]
    )

    assert "System instructions" in prompt
    assert "USER: First question" in prompt
    assert "ASSISTANT: First answer" in prompt
    assert "USER: Second question" in prompt


def test_build_prompt_from_responses_request_supports_structured_input() -> None:
    payload = ResponsesRequest(
        input=[
            ResponseInputItem(
                role="developer",
                content=[ResponseInputPart(type="input_text", text="Use verified facts only.")],
            ),
            ResponseInputItem(
                role="user",
                content=[ResponseInputPart(type="input_text", text="Summarize the result")],
            ),
        ],
        instructions="Use three bullet points.",
    )

    prompt = build_prompt_from_responses_request(payload)
    assert "Use three bullet points." in prompt
    assert "Use verified facts only." in prompt
    assert "USER: Summarize the result" in prompt


def test_chat_completion_response_contains_usage_and_citations() -> None:
    result = GoogleAiResult(
        answer_text="Three point answer.",
        citations=[Citation(title="OpenAI", url="https://openai.com")],
        final_url="https://www.google.com/search?udm=50",
        page_title="Google Search",
    )

    payload = build_chat_completion_response(
        result=result,
        model_name="google-search",
        prompt="question",
    )

    assert payload["model"] == "google-search"
    assert payload["choices"][0]["message"]["content"] == "Three point answer."
    assert payload["usage"]["total_tokens"] >= payload["usage"]["prompt_tokens"]
    assert payload["citations"][0]["url"] == "https://openai.com"


def test_openai_stream_deltas_preserve_original_formatting() -> None:
    answer = (
        "Header\n\n"
        "- first item keeps  double spaces\n"
        "- second item keeps indentation:\n"
        "    nested detail with enough text to force a streamed chunk split after the default "
        "target size"
    )
    result = GoogleAiResult(
        answer_text=answer,
        citations=[],
        final_url="https://www.google.com/search?udm=50",
        page_title="Google Search",
    )

    chat_deltas = []
    for event in iter_chat_completion_stream(result=result, model_name="google-search"):
        if event == "data: [DONE]\n\n":
            continue
        data = json.loads(event.removeprefix("data: ").strip())
        delta = data["choices"][0]["delta"]
        if "content" in delta:
            chat_deltas.append(delta["content"])

    response_deltas = []
    for event in iter_responses_api_stream(result=result, model_name="google-search"):
        if not event.startswith("event: response.output_text.delta"):
            continue
        data_line = next(line for line in event.splitlines() if line.startswith("data: "))
        response_deltas.append(json.loads(data_line.removeprefix("data: "))["delta"])

    assert "".join(chat_deltas) == answer
    assert "".join(response_deltas) == answer
