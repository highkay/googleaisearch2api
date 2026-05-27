from googleaisearch2api.duck_ai import extract_duck_answer_text


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
