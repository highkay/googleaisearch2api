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


def test_extract_duck_answer_text_keeps_only_json_results_for_json_prompt() -> None:
    answer = extract_duck_answer_text(
        """
Duck.ai
User request:
Return JSON with results.
唯特偶 A股 股票代码 上市公司 2026-05-27
money.finance.sina.com.cn
quote.eastmoney.com
{"results":[{"title":"唯特偶","content":"股票代码为301319。","source":"新浪财经","url":"https://example.com","published_date":"2026-05-27"}]}
""",
        "User request:\nReturn JSON with results.",
    )

    assert answer == (
        '{"results": [{"title": "唯特偶", "content": "股票代码为301319。", '
        '"source": "新浪财经", "url": "https://example.com", '
        '"published_date": "2026-05-27"}]}'
    )


def test_extract_duck_answer_text_removes_follow_up_tail() -> None:
    answer = extract_duck_answer_text(
        """
Duck.ai
台积电 3nm 涨价可能利好半导体设备和材料供应链。
若需我把每家公司的最新财务数据补充成表格，请告诉我。
""",
        "台积电 3nm 涨价 最多返回 5 条",
    )

    assert answer == "台积电 3nm 涨价可能利好半导体设备和材料供应链。"
