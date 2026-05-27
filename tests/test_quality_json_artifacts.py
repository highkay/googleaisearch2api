from googleaisearch2api.quality import assess_google_answer_quality


def test_rejects_json_result_with_malformed_stock_code_for_stock_prompts() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。台积电 3nm 涨价 AI A股 受益股 最多返回 5 条',
        '{"results":[{"title":"汇顶科技","content":"汇顶科技（603160 / 0x）受益于 AI 终端需求。",'
        '"source":"东方财富","url":"https://example.com/news","published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer contains malformed stock code"


def test_rejects_json_result_with_citation_marker_artifact() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}',
        '{"results":[{"title":"板块异动","content":"电子板块盘中走强。[3]",'
        '"source":"新浪财经","url":"https://example.com/news","published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result contains citation marker artifacts"
