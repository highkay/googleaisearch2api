from googleaisearch2api.quality import assess_google_answer_quality, normalize_answer_for_prompt
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


def test_rejects_json_result_outside_requested_date_range() -> None:
    quality = assess_google_answer_quality(
        "时间范围必须限制在 2026-05-27 至 2026-05-27。只返回一个 JSON 对象，"
        '输出格式固定为 {"results":[]}',
        '{"results":[{"title":"唯特偶公告","content":"一季度报告。","source":"新浪财经",'
        '"url":"https://example.com/news","published_date":"2026-04-22"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result published_date is outside requested date range"


def test_normalize_answer_for_prompt_filters_out_of_range_json_results() -> None:
    answer = normalize_answer_for_prompt(
        "时间范围必须限制在 2026-05-27 至 2026-05-27。只返回一个 JSON 对象，"
        '输出格式固定为 {"results":[]}',
        '{"results":[{"title":"旧公告","content":"一季度报告。","source":"新浪财经",'
        '"url":"https://example.com/old","published_date":"2026-04-22"},'
        '{"title":"当天新闻","content":"当天事件。","source":"财联社",'
        '"url":"https://example.com/today","published_date":"2026-05-27"}]}',
    )

    assert answer == (
        '{"results": [{"title": "当天新闻", "content": "当天事件。", '
        '"source": "财联社", "url": "https://example.com/today", '
        '"published_date": "2026-05-27"}]}'
    )
