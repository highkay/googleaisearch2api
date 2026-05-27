from googleaisearch2api.quality import assess_google_answer_quality


def test_rejects_json_result_with_malformed_stock_code_for_stock_prompts() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。台积电 3nm 涨价 AI A股 受益股 最多返回 5 条',
        '{"results":[{"title":"汇顶科技","content":"汇顶科技（603160 / 0x）受益于 AI 终端需求。",'
        '"source":"东方财富","url":"https://finance.eastmoney.com/a/202605270001.html",'
        '"published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer contains malformed stock code"


def test_rejects_json_result_with_citation_marker_artifact() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}',
        '{"results":[{"title":"板块异动","content":"电子板块盘中走强。[3]",'
        '"source":"新浪财经","url":"https://finance.sina.com.cn/stock/2026-05-27/doc-news.shtml",'
        '"published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result contains citation marker artifacts"


def test_rejects_empty_json_results() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON results list is empty"


def test_rejects_json_result_with_future_published_date() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[{"title":"未来新闻","content":"尚未发生的内容。","source":"财联社",'
        '"url":"https://www.cls.cn/detail/299901010001","published_date":"2999-01-01"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result published_date is in the future"


def test_rejects_json_result_with_placeholder_url() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[{"title":"伪新闻","content":"缺少真实来源。","source":"财联社",'
        '"url":"https://example.com/fake","published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result has an unusable URL"


def test_rejects_json_result_with_generic_source_label() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[{"title":"泛化来源","content":"来源字段不是具体媒体或网页。","source":"综合行业报道与公司公告",'
        '"url":"https://finance.eastmoney.com/a/202605270001.html",'
        '"published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result source is generic"
