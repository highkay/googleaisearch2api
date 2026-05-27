from googleaisearch2api.duck_ai import build_duck_search_prompt, extract_duck_answer_text


def test_build_duck_search_prompt_forces_direct_search_answer() -> None:
    prompt = build_duck_search_prompt("台积电 3nm 涨价 最多返回 5 条")

    assert "Do not return only search suggestions." in prompt
    assert "Do not ask whether to search." in prompt
    assert "only include ticker symbols you are confident are real" in prompt
    assert "real six-digit A-share codes" in prompt
    assert "Use one concrete publisher name in each source field" in prompt
    assert "If the URL host is not the named publisher's host" in prompt
    assert "Each result must have a concrete publication date from the page" in prompt
    assert "Do not use Japanese labels such as リンク." in prompt
    assert "Do not output placeholder URLs such as example.com." in prompt
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


def test_extract_duck_answer_text_removes_raw_search_result_ui_noise() -> None:
    answer = extract_duck_answer_text(
        """
Duck.ai
User request:
台积电 3nm 涨价 最多返回 5 条。
台积电 3nm 涨价 AI A股 受益 股 供应链 半导体 相关新闻 2024 2025
cls.cn
stcn.com
标题：AI浪潮下台积电涨价传导顺畅，大客户争相预订先进制程产能
来源：财联社
日期：2024-07-08
链接：https://www.cls.cn/detail/1726012
为什么相关：报道指出台积电先进制程涨价谈判正在推进。
Search Results
AI浪潮下台积电涨价传导顺畅大客户争相预订先进制程产能
cls.cn
""",
        "User request:\n台积电 3nm 涨价 最多返回 5 条。",
    )

    assert answer == (
        "标题：AI浪潮下台积电涨价传导顺畅，大客户争相预订先进制程产能\n"
        "来源：财联社\n"
        "日期：2024-07-08\n"
        "链接：https://www.cls.cn/detail/1726012\n"
        "为什么相关：报道指出台积电先进制程涨价谈判正在推进。"
    )


def test_extract_duck_answer_text_removes_leading_query_before_inline_items() -> None:
    answer = extract_duck_answer_text(
        """
Duck.ai
User request:
台积电 3nm 涨价 最多返回 5 条。
台积电 3nm 涨价 AI A股 受益 供应链 半导体 受益 股 相关新闻 2024 2025
台积电，突传重磅！ — 证券时报网 — 2024-07-07 — https://www.stcn.com/article/detail/1250957.html
为什么相关：报道引用机构供应链访查，称台积电将对3nm家族涨价。
""",
        "User request:\n台积电 3nm 涨价 最多返回 5 条。",
    )

    assert answer == (
        "台积电，突传重磅！ — 证券时报网 — 2024-07-07 — "
        "https://www.stcn.com/article/detail/1250957.html\n"
        "为什么相关：报道引用机构供应链访查，称台积电将对3nm家族涨价。"
    )


def test_extract_duck_answer_text_keeps_only_json_results_for_json_prompt() -> None:
    answer = extract_duck_answer_text(
        """
Duck.ai
User request:
Return JSON with results.
唯特偶 A股 股票代码 上市公司 2026-05-27
money.finance.sina.com.cn
quote.eastmoney.com
{"results":[{"title":"唯特偶","content":"股票代码为301319。","source":"新浪财经","url":"https://finance.sina.com.cn/stock/2026-05-27/doc-news.shtml","published_date":"2026-05-27"}]}
""",
        "User request:\nReturn JSON with results.",
    )

    assert answer == (
        '{"results": [{"title": "唯特偶", "content": "股票代码为301319。", '
        '"source": "新浪财经", '
        '"url": "https://finance.sina.com.cn/stock/2026-05-27/doc-news.shtml", '
        '"published_date": "2026-05-27"}]}'
    )


def test_extract_duck_answer_text_removes_follow_up_tail() -> None:
    answer = extract_duck_answer_text(
        """
Duck.ai
台积电 3nm 涨价可能利好半导体设备和材料供应链。
如果您想深入了解，我可以把每家公司的最新财务数据补充成表格。
""",
        "台积电 3nm 涨价 最多返回 5 条",
    )

    assert answer == "台积电 3nm 涨价可能利好半导体设备和材料供应链。"
