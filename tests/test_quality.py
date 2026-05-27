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
