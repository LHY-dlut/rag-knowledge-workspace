import json

import httpx
import pytest

from app.evidence_protocol import bind_decision, evidence_spans
from app.providers import DashScopeProvider, GradeProtocolError
from app.schemas import GradeDecision, GradeSpanAssessment
from tests.test_providers_parsers import remote_settings

SOURCE = {"source_id": "S1", "content": "预约服务只向本区居民开放。"}
EVIDENCE = [{"source_id": "S1", "span_id": "S1:E1", "subject": "预约服务", "attribute": "适用对象"}]


async def response_case(reply):
    def handler(request):
        body = json.loads(request.content)
        assert (
            json.dumps(GradeSpanAssessment.model_json_schema(), ensure_ascii=False)
            in body["messages"][0]["content"]
        )
        assert "passed" not in GradeSpanAssessment.model_json_schema()["properties"]
        wire = json.loads(body["messages"][1]["content"])
        assert wire["query"] == "哪些人可以使用预约服务？"
        assert wire["sources"][0]["reading_text"] == SOURCE["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        return await provider.structured(
            GradeDecision,
            "grade",
            {
                "query": "哪些人可以使用预约服务？",
                "sources": [{**SOURCE, "evidence_spans": evidence_spans(SOURCE)}],
            },
        )
    finally:
        await provider.close()


@pytest.mark.parametrize("support", ["supported_answer", "supported_limitation", "insufficient"])
async def test_single_classification_computes_domain_boolean_with_original_evidence_guards(support):
    reply = {"support": support, "reason": "仅在所给原文范围判断。"}
    if support != "insufficient":
        reply.update(evidence=EVIDENCE, answer_scope="预约服务只向本区居民开放。")
    result = await response_case(reply)
    assert type(result.passed) is bool and result.passed == (support != "insufficient")
    assert result.support == support
    if result.passed:
        bound, audit = bind_decision(result, [SOURCE])
        assert audit["passed"] and bound.evidence[0].quote == SOURCE["content"]


@pytest.mark.parametrize(
    "passed,support",
    [
        (False, "supported_answer"),
        (True, "insufficient"),
        ("true", "supported_answer"),
        (1, "supported_answer"),
    ],
)
async def test_old_contradictory_or_non_boolean_response_never_auto_corrected(passed, support):
    with pytest.raises(GradeProtocolError):
        await response_case(
            {
                "passed": passed,
                "support": support,
                "evidence": EVIDENCE,
                "answer_scope": "仅在所给原文范围",
                "reason": "不能自动丢掉passed字段",
            }
        )


@pytest.mark.parametrize("support", ["supported_answer", "supported_limitation", "insufficient"])
async def test_consistent_legacy_response_retains_strict_backward_compatibility(support):
    reply = {
        "passed": support != "insufficient",
        "support": support,
        "reason": "旧协议仍严格一致",
        "evidence": EVIDENCE if support != "insufficient" else [],
        "answer_scope": "仅在所给原文范围" if support != "insufficient" else "",
    }
    result = await response_case(reply)
    assert result.passed == reply["passed"] and result.support == support


@pytest.mark.parametrize(
    "invalid",
    [
        {"support": "maybe", "reason": "未知分类"},
        {"support": True, "reason": "分类不能是布尔值"},
        {
            "support": "supported_answer",
            "evidence": [],
            "answer_scope": "无证据",
            "reason": "空证据",
        },
        {
            "support": "supported_answer",
            "evidence": EVIDENCE,
            "answer_scope": "",
            "reason": "空范围",
        },
        {"support": "insufficient", "reason": "不允许额外字段", "override": True},
    ],
)
async def test_invalid_enum_missing_evidence_scope_and_unknown_fields_remain_rejected(invalid):
    with pytest.raises(GradeProtocolError):
        await response_case(invalid)


async def test_valid_enum_does_not_bypass_unknown_literal_evidence_id():
    result = await response_case(
        {
            "support": "supported_answer",
            "reason": "模型选择了无效片段",
            "evidence": [{**EVIDENCE[0], "span_id": "S1:E999"}],
            "answer_scope": "本区居民可使用",
        }
    )
    bound, audit = bind_decision(result, [SOURCE])
    assert not bound.passed and audit["failure_types"] == ["evidence_span_unknown"]
