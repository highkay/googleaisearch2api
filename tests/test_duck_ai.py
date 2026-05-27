from googleaisearch2api.duck_ai import build_duck_search_prompt, extract_duck_answer_text


def test_build_duck_search_prompt_forces_direct_search_answer() -> None:
    prompt = build_duck_search_prompt("台积电 3nm 涨价 最多返回 5 条")

    assert "Do not return only search suggestions." in prompt
    assert "Do not ask whether to search." in prompt
    assert "台积电 3nm 涨价 最多返回 5 条" in prompt


def test_extract_duck_answer_text_removes_prompt_echo_and_shell_text() -> None:
    body_text = """
Duck.ai
GPT-5 Mini
Return one sentence about DuckDuckGo.
DuckDuckGo is a privacy-focused search engine.
All chats are private.
"""

    answer = extract_duck_answer_text(
        body_text,
        "Return one sentence about DuckDuckGo.",
    )

    assert answer == "DuckDuckGo is a privacy-focused search engine."


def test_extract_duck_answer_text_removes_tool_status_lines() -> None:
    answer = extract_duck_answer_text(
        """
Duck.ai
User request:
Return JSON.
Searching the web
Searching the web market query
Hide Reasoning
{"results": []}
""",
        "User request:\nReturn JSON.",
    )

    assert answer == '{"results": []}'
