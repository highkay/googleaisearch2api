from __future__ import annotations

from googleaisearch2api.iplark import _payloads_from_body_text, parse_iplark_payloads


def test_parse_iplark_payloads_handles_score_and_flags() -> None:
    result = parse_iplark_payloads(
        "203.0.113.10",
        score_json={"data": {"quality_score": "76"}},
        intelligence_json={
            "data": {
                "usageType": "residential",
                "category": "normal",
                "publicProxy": False,
                "threat": "false",
                "tags": ["clean"],
            }
        },
    )

    assert result.quality_score == 76
    assert result.usage_type == "residential"
    assert result.category == "normal"
    assert result.public_proxy is False
    assert result.threat is False
    assert result.tag == "clean"


def test_parse_iplark_body_text_extracts_free_page_score() -> None:
    score_json, intelligence_json = _payloads_from_body_text(
        "8.8.8.8",
        """
        IP评分
        72
        72/100
        IP情报
        使用类型:
        数据中心
        威胁:
        -
        IP类型:
        数据中心
        代理:
        否
        标签:
        vpn & hosting
        """,
    )

    result = parse_iplark_payloads(
        "8.8.8.8",
        score_json=score_json,
        intelligence_json=intelligence_json,
    )

    assert result.quality_score == 72
    assert result.usage_type == "数据中心"
    assert result.category == "数据中心"
    assert result.public_proxy is False
    assert result.threat is False
    assert result.tag == "vpn & hosting"
