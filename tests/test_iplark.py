from __future__ import annotations

from googleaisearch2api.iplark import parse_iplark_payloads


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
