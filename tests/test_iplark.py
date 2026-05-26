from __future__ import annotations

from googleaisearch2api.iplark import (
    _payloads_from_body_text,
    _payloads_from_ipapi_payload,
    parse_iplark_payloads,
)


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


def test_parse_ipapi_payload_maps_datacenter_low_risk_to_quality_score() -> None:
    score_json, intelligence_json = _payloads_from_ipapi_payload(
        "8.8.8.8",
        {
            "ip": "8.8.8.8",
            "is_datacenter": True,
            "is_proxy": False,
            "is_vpn": False,
            "is_tor": False,
            "is_abuser": True,
            "company": {"type": "hosting", "abuser_score": "0.0039 (Low)"},
            "asn": {"abuser_score": "0.001 (Low)"},
        },
    )

    result = parse_iplark_payloads(
        "8.8.8.8",
        score_json=score_json,
        intelligence_json=intelligence_json,
        source="ipapi.is",
    )

    assert result.quality_score == 75
    assert result.source == "ipapi.is"
    assert result.usage_type == "datacenter"
    assert result.category == "hosting"
    assert result.public_proxy is False
    assert result.threat is False
    assert result.tag == "datacenter, abuser, risk:low"


def test_parse_ipapi_payload_marks_proxy_as_public_proxy() -> None:
    score_json, intelligence_json = _payloads_from_ipapi_payload(
        "203.0.113.10",
        {
            "ip": "203.0.113.10",
            "is_proxy": True,
            "is_vpn": True,
            "is_abuser": True,
            "company": {"abuser_score": "0.44 (Medium)"},
        },
    )

    result = parse_iplark_payloads(
        "203.0.113.10",
        score_json=score_json,
        intelligence_json=intelligence_json,
        source="ipapi.is",
    )

    assert result.quality_score < 70
    assert result.public_proxy is True
    assert result.threat is False
    assert "proxy" in (result.tag or "")
