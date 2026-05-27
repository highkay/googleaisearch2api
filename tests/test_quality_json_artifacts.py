import json

from googleaisearch2api.quality import (
    assess_google_answer_quality,
    normalize_answer_for_prompt,
)


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


def test_rejects_json_result_with_url_artifact() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[{"title":"伪新闻","content":"URL 含模型拼接痕迹。","source":"新浪财经",'
        '"url":"https://finance.sina.com.cn/tech/2024-02-17/doc-xyz.html",'
        '"published_date":"2024-02-17"}]}',
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


def test_rejects_json_result_with_aggregate_source_label() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[{"title":"聚合来源","content":"来源字段拼接了多个发布方。",'
        '"source":"每日经济新闻 / 新浪财经转载",'
        '"url":"https://finance.sina.com.cn/stock/2026-05-27/doc-news.shtml",'
        '"published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result source is aggregate"


def test_rejects_json_results_with_duplicate_urls() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[{"title":"第一条","content":"同一网页的第一条。","source":"凤凰网财经",'
        '"url":"https://finance.ifeng.com/c/8b1M0339ZZB","published_date":"2024-07-07"},'
        '{"title":"第二条","content":"同一网页拆成第二条。","source":"凤凰网财经",'
        '"url":"https://finance.ifeng.com/c/8b1M0339ZZB","published_date":"2024-07-07"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result URL is duplicated"


def test_rejects_json_result_with_non_specific_url() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[{"title":"汇总页","content":"只给出了媒体首页。","source":"财联社",'
        '"url":"https://www.cls.cn/","published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result URL is not specific"


def test_rejects_json_result_with_topic_url() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条',
        '{"results":[{"title":"专题页","content":"只给出了标签页。","source":"电子工程专辑",'
        '"url":"https://www.eet-china.com/mp/tags/88974","published_date":"2025-12-30"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result URL is not specific"


def test_normalize_answer_for_prompt_filters_bad_json_result_items() -> None:
    prompt = (
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条'
    )
    answer = (
        '{"results":['
        '{"title":"第一条","content":"第一条可追溯结果。","source":"凤凰网财经",'
        '"url":"https://finance.ifeng.com/c/8b1M0339ZZB","published_date":"2026-05-27"},'
        '{"title":"重复条","content":"同一网页被拆成第二条。","source":"凤凰网财经",'
        '"url":"https://finance.ifeng.com/c/8b1M0339ZZB","published_date":"2026-05-27"},'
        '{"title":"聚合来源","content":"来源字段拼接了多个发布方。",'
        '"source":"东方财富数据中心 / 彭博社金融数据",'
        '"url":"https://data.eastmoney.com/zjlx/600487.html","published_date":"2026-05-27"},'
        '{"title":"汇总页","content":"只给出了媒体首页。","source":"财联社",'
        '"url":"https://www.cls.cn/","published_date":"2026-05-27"},'
        '{"title":"第二条","content":"第二条可追溯结果。","source":"新浪财经",'
        '"url":"https://finance.sina.com.cn/stock/2026-05-27/doc-valid.shtml",'
        '"published_date":"2026-05-27"}]}'
    )

    normalized = normalize_answer_for_prompt(prompt, answer)
    payload = json.loads(normalized)

    assert [item["title"] for item in payload["results"]] == ["第一条", "第二条"]
    assert assess_google_answer_quality(prompt, normalized).ok is True


def test_normalize_answer_for_prompt_keeps_all_bad_json_results() -> None:
    prompt = (
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。最多返回 5 条'
    )
    answer = (
        '{"results":[{"title":"聚合来源","content":"来源字段拼接了多个发布方。",'
        '"source":"每日经济新闻 / 新浪财经转载",'
        '"url":"https://finance.sina.com.cn/stock/2026-05-27/doc-news.shtml",'
        '"published_date":"2026-05-27"}]}'
    )

    assert normalize_answer_for_prompt(prompt, answer) == answer


def test_normalize_answer_for_prompt_returns_allowed_empty_when_all_json_results_bad() -> None:
    prompt = (
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}。若找不到足够直接相关的结果，返回 '
        '{"results": []}。最多返回 5 条'
    )
    answer = (
        '{"results":[{"title":"天岳先进(688234) 行情走势",'
        '"content":"天岳先进在科创板上市，股票代码为688234。",'
        '"source":"Google Finance","url":"google.com",'
        '"published_date":"2026-05-27"}]}'
    )

    normalized = normalize_answer_for_prompt(prompt, answer)

    assert normalized == '{"results": []}'
    assert assess_google_answer_quality(prompt, normalized).ok is True
