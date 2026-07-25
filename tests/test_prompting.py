from googleaisearch2api.prompting import adapt_prompt_for_engine, simplify_search_prompt


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


def test_adapt_prompt_for_duck_rewrites_keyword_pile_without_site_operators() -> None:
    prompt = (
        "搜索并用自然语言简要回答，列出关键发现、来源和日期；"
        "如果没有足够直接相关的信息，直接说明未找到：\n"
        "美国袭击伊朗 霍尔木兹海峡 A股 风险传导 site:gov.cn "
        "最多返回 5 条。时间范围必须限制在 2026-07-15 至 2026-07-17（含）之间。"
    )

    duck = adapt_prompt_for_engine(prompt, engine="duck")
    google = adapt_prompt_for_engine(prompt, engine="google")

    assert google == prompt
    assert "site:gov.cn" not in duck
    assert "最多返回 5 条" not in duck
    assert "时间范围必须" not in duck
    assert "请用自然语言完整回答" in duck
    assert "霍尔木兹" in duck
    assert "不要堆砌搜索词" in duck


def test_adapt_prompt_for_duck_keeps_clear_natural_questions() -> None:
    prompt = "请说明 2026-07-17 电力板块为什么走强，并给出可核对来源。"

    duck = adapt_prompt_for_engine(prompt, engine="duck")

    assert "请用自然语言完整回答" in duck
    assert "电力板块为什么走强" in duck

