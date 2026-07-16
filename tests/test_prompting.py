from googleaisearch2api.prompting import simplify_search_prompt


def test_simplify_search_prompt_turns_json_results_wrapper_into_natural_language() -> None:
    prompt = (
        "User request:\n"
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"","published_date":"YYYY-MM-DD"}]}。'
        '若找不到足够直接相关的结果，返回 {"results": []}。'
        "问题：数字政通 300075.SZ 最新公告 新闻 催化 风险 最多返回 5 条"
    )

    simplified = simplify_search_prompt(prompt)

    assert simplified == (
        "搜索并用自然语言简要回答，列出关键发现、来源和日期；"
        "如果没有足够直接相关的信息，直接说明未找到：\n"
        "数字政通 300075.SZ 最新公告 新闻 催化 风险 最多返回 5 条"
    )
    assert "JSON" not in simplified
    assert "results" not in simplified


def test_simplify_search_prompt_leaves_plain_natural_language_prompt_unchanged() -> None:
    prompt = "User request:\n数字政通 300075.SZ 最近一周有什么公告或新闻？"

    assert simplify_search_prompt(prompt) == prompt
