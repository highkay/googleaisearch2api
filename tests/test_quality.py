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


def test_rejects_follow_up_prompt_tail() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        "台积电 3nm 涨价将带动先进封装、设备和材料供应链需求。"
        "如果您想深入了解，可以告诉我您更看重哪个方向。",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains a follow-up prompt tail"


def test_rejects_multiple_standalone_source_labels() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
台积电 3nm 涨价将推动 A 股半导体供应链重估。
1. 半导体设备：中微公司、北方华创受益于扩产。
财联社
2. 先进封装：长电科技、通富微电受益于 AI 芯片封测需求。
东方财富
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains standalone source labels"


def test_rejects_raw_answer_with_search_results_tail() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
标题：AI浪潮下台积电涨价传导顺畅
来源：财联社
日期：2024-07-08
链接：https://www.cls.cn/detail/1726012
为什么相关：报道说明先进制程涨价可能传导至供应链。
Search Results
AI浪潮下台积电涨价传导顺畅
cls.cn
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains search results UI tail"


def test_rejects_raw_answer_with_standalone_hostnames() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
标题：AI浪潮下台积电涨价传导顺畅
cls.cn
标题：台积电，突传重磅！
stcn.com
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains standalone hostnames"


def test_rejects_raw_answer_with_duplicate_urls() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
标题：台积电，突传重磅！
来源：证券时报网
日期：2024-07-07
链接：https://www.stcn.com/article/detail/1250957.html
为什么相关：报道说明 3nm 产能紧张与涨价预期。
标题：AI推动下半导体涨价潮
来源：证券时报网
日期：2024-07-07
链接：https://www.stcn.com/article/detail/1250957.html
为什么相关：同一个网页被拆成重复结果。
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains duplicate URLs"


def test_rejects_raw_answer_with_aggregate_source_label() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
台积电3nm产能与涨价相关报道汇总 — 财联社 / 多家财经媒体汇编 — 2024-07-08 — https://www.cls.cn/detail/1726012
为什么相关：来源字段不是单一具体发布方。
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains aggregate source labels"


def test_normalize_answer_for_prompt_filters_bad_raw_result_blocks() -> None:
    prompt = (
        "台积电 3nm 涨价 AI A股 受益股 OR 供应链 OR 半导体 最多返回 5 条。"
        "请直接返回可读的搜索摘要，每条包含标题、来源、日期、链接和一句为什么相关。"
        "不要返回 JSON。"
    )
    answer = """
AI浪潮下台积电涨价传导顺畅 大客户争相预订先进制程产能
来源：财联社
日期：2024-07-08
链接：https://www.cls.cn/detail/1726012
为什么相关：这篇报道明确提到台积电先进制程涨价、AI 大客户预订产能以及供应链传导，
对 A 股半导体设备和材料线索有直接参考价值。
（行业综述）台积电3nm产能与价格影响
来源：证券时报（专题合集）
日期：2024-07-07
链接：https://www.stcn.com/special/taishijidian_3nm
为什么相关：专题页汇集多篇内容，不是单篇可核验新闻。
"""

    normalized = normalize_answer_for_prompt(prompt, answer)

    assert "https://www.cls.cn/detail/1726012" in normalized
    assert "taishijidian_3nm" not in normalized
    assert assess_google_answer_quality(prompt, normalized).ok is True


def test_allows_raw_answer_with_single_reprint_note() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价",
        """
台积电将对3-5nm AI芯片涨价5%-10%
来源：电子技术应用（ChinaAET，转载自芯智讯）
日期：2024-07-08
链接：https://chinaaet.com/article/3000166304
相关性：报道给出3nm和5nm先进制程针对AI产品的涨价幅度，并说明先进封装需求对产业链的影响。
""",
    )

    assert quality.ok is True


def test_rejects_raw_answer_with_non_chinese_link_label() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
台积电将对3-5nm AI芯片涨价5%-10%
来源：电子技术应用（ChinaAET，转载自芯智讯）
日期：2024-07-08
リンク：https://chinaaet.com/article/3000166304
为什么相关：字段标签混入了非中文链接标签。
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains non-standard link labels"


def test_rejects_raw_answer_with_source_host_mismatch() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
全球芯片代工巨头有大消息
来源：每日经济新闻（新浪财经转载）
日期：2024-07-07
链接：https://xincai.com/article/ncchqki5543745
为什么相关：来源写的是已知媒体，但链接 host 不是该媒体站点。
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains source URL host mismatches"


def test_rejects_raw_answer_with_non_specific_publication_date() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
台积电3nm产能与涨价相关报道汇总 — 财联社 — 2024–2026（多篇） — https://www.cls.cn/detail/1726012
为什么相关：日期不是单篇新闻的发布时间。
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains non-specific publication dates"


def test_rejects_raw_answer_with_placeholder_publication_date_line() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
AI浪潮下台积电涨价传导顺畅大客户争相预订先进制程产能
来源：财联社
日期：见页（文章内未标注明确出版年）
链接：https://www.cls.cn/detail/1726012
为什么相关：日期字段不是可核验的单篇发布时间。
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains non-specific publication dates"


def test_rejects_raw_answer_with_non_specific_url_for_result_list() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
台积电3nm产能与涨价相关报道 — 财联社 — 2024-07-08 — https://www.cls.cn/
为什么相关：首页不是单篇报道链接。
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains non-specific URLs"


def test_rejects_raw_answer_with_topic_url_for_result_list() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 最多返回 5 条",
        """
台积电涨价专题汇编 — 电子工程专辑 — 2025-12-30 — https://www.eet-china.com/mp/tags/88974
为什么相关：标签页不是单篇报道链接。
""",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains non-specific URLs"


def test_rejects_malformed_stock_code_for_stock_prompts() -> None:
    quality = assess_google_answer_quality(
        "台积电 3nm 涨价 AI A股 受益股 OR 供应链 OR 半导体 最多返回 5 条",
        "汇顶科技（603160 / 0x） — AI 终端需求提升，间接受益。",
    )

    assert quality.ok is False
    assert quality.reason == "answer contains malformed stock code"


def test_allows_hex_text_for_non_stock_prompts() -> None:
    quality = assess_google_answer_quality(
        "解释一下十六进制地址",
        "十六进制字面量通常以 0x 开头，例如 0x10 表示十进制的 16。",
        [Citation(title="Hex docs", url="https://example.com/hex")],
    )

    assert quality.ok is True


def test_accepts_valid_json_results_without_citations() -> None:
    quality = assess_google_answer_quality(
        "只返回一个 JSON 对象，输出格式固定为 "
        '{"results":[{"title":"","content":"","source":"","url":"",'
        '"published_date":"YYYY-MM-DD"}]}',
        '{"results":[{"title":"新闻标题","content":"直接相关内容","source":"东方财富",'
        '"url":"https://finance.eastmoney.com/a/202605270001.html",'
        '"published_date":"2026-05-27"}]}',
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
        '"url":"https://finance.sina.com.cn/stock/2026-04-22/doc-news.shtml",'
        '"published_date":"2026-04-22"}]}',
    )

    assert quality.ok is False
    assert quality.reason == "answer JSON result published_date is outside requested date range"


def test_normalize_answer_for_prompt_filters_out_of_range_json_results() -> None:
    answer = normalize_answer_for_prompt(
        "时间范围必须限制在 2026-05-27 至 2026-05-27。只返回一个 JSON 对象，"
        '输出格式固定为 {"results":[]}',
        '{"results":[{"title":"旧公告","content":"一季度报告。","source":"新浪财经",'
        '"url":"https://finance.sina.com.cn/stock/2026-04-22/doc-old.shtml",'
        '"published_date":"2026-04-22"},'
        '{"title":"当天新闻","content":"当天事件。","source":"财联社",'
        '"url":"https://www.cls.cn/detail/202605270001",'
        '"published_date":"2026-05-27"}]}',
    )

    assert answer == (
        '{"results": [{"title": "当天新闻", "content": "当天事件。", '
        '"source": "财联社", "url": "https://www.cls.cn/detail/202605270001", '
        '"published_date": "2026-05-27"}]}'
    )


def test_normalize_answer_for_prompt_filters_future_json_results() -> None:
    answer = normalize_answer_for_prompt(
        '只返回一个 JSON 对象，输出格式固定为 {"results":[]}',
        '{"results":[{"title":"未来新闻","content":"尚未发生的内容。","source":"财联社",'
        '"url":"https://www.cls.cn/detail/299901010001","published_date":"2999-01-01"},'
        '{"title":"已发布新闻","content":"已发生的内容。","source":"新浪财经",'
        '"url":"https://finance.sina.com.cn/stock/2020-01-01/doc-news.shtml",'
        '"published_date":"2020-01-01"}]}',
    )

    assert answer == (
        '{"results": [{"title": "已发布新闻", "content": "已发生的内容。", '
        '"source": "新浪财经", '
        '"url": "https://finance.sina.com.cn/stock/2020-01-01/doc-news.shtml", '
        '"published_date": "2020-01-01"}]}'
    )
