from googleaisearch2api.quality import assess_google_answer_quality
from googleaisearch2api.schemas import Citation


def test_rejects_short_answer_when_prompt_requests_multiple_items() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 OR 供应链 OR 半导体 最多返回 5 条",
        "台积电 3nm 先进制程涨价以及 AI 算力需求的爆发，将推动半导体供应链重估。",
        [Citation(title="Source", url="https://example.com/report")],
    )

    assert quality.ok is False
    assert quality.reason == "answer is too short for the requested list"


def test_rejects_short_answer_without_usable_citations() -> None:
    quality = assess_google_answer_quality("NVIDIA latest news", "NVIDIA shares rose today.")

    assert quality.ok is False
    assert quality.reason == "short answer has no usable citations"


def test_accepts_short_answer_with_usable_citation() -> None:
    quality = assess_google_answer_quality(
        "NVIDIA latest news",
        "NVIDIA shares rose today.",
        [Citation(title="Market report", url="https://example.com/market")],
    )

    assert quality.ok is True


def test_accepts_valid_json_results_without_citations() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}',
        '{"results":[{"title":"新闻标题","content":"直接相关内容","source":"东方财富","url":"https://example.com/news","published_date":"2026-05-27"}]}',
    )

    assert quality.ok is True


def test_rejects_non_json_when_prompt_requires_json_results() -> None:
    quality = assess_google_answer_quality(
        '只返回一个 JSON 对象，输出格式固定为 {"results":[]}',
        "这里是一个普通回答。",
    )

    assert quality.ok is False
    assert quality.reason == "answer is not valid JSON for requested results"


def test_rejects_json_result_with_internal_source_label() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}',
        '{"results":[{"title":"唯特偶股票A股实时行情及股票代码","content":"深圳市唯特偶新材料股份有限公司信息。","source":"FinanceResult","url":"https://quote.eastmoney.com/sz301319.html","published_date":"2026-05-27"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result source is an internal label"
