import copy
import json

import httpx
import pytest

from app.evidence_protocol import bind_decision, evidence_spans
from app.providers import DashScopeProvider, ProviderError
from app.schemas import GradeDecision, GradeSpanDecision
from tests.test_providers_parsers import remote_settings


def response(decision):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(decision)}}]})


def selected(span_id="S1:E1", source_id="S1"):
    return {"source_id": source_id, "span_id": span_id, "subject": "检修", "attribute": "记录要求"}


def supported(evidence):
    return {
        "passed": True,
        "support": "supported_answer",
        "reason": "原文明确规定记录要求",
        "evidence": evidence,
        "answer_scope": "仅说明该资料的记录要求",
    }


def payload(source):
    return {
        "query": "检修记录有哪些要求？",
        "sources": [{**source, "evidence_spans": evidence_spans(source)}],
    }


async def test_span_transport_uses_ids_only_and_server_preserves_chinese_raw_offsets():
    source = {
        "source_id": "S1",
        "content": "检修记录应注明部位；\n发现裂纹时附原始照片。\n复查后签字。",
    }
    data = payload(source)
    before = copy.deepcopy(data)

    def handler(request):
        body = json.loads(request.content)
        text = body["messages"][0]["content"]
        schema_text = text.split("JSON 对象：", 1)[1].split(" 本次片段ID", 1)[0]
        schema = json.loads(schema_text)
        fields = schema["$defs"]["GradeSpanEvidence"]["properties"]
        assert (
            "quote" not in fields and "span_id" in schema["$defs"]["GradeSpanEvidence"]["required"]
        )
        assert schema["$defs"]["GradeSpanEvidence"]["additionalProperties"] is False
        return response(
            supported([selected(s["span_id"]) for s in data["sources"][0]["evidence_spans"]])
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        decision = await provider.structured(GradeDecision, "grade", data)
        assert type(decision) is GradeDecision
        assert all(e.quote == "" for e in decision.evidence)
        bound, audit = bind_decision(decision, [source])
        assert bound.passed and audit["passed"]
        for row in audit["evidence"]:
            assert (
                source["content"][row["source_start"] : row["source_end"]] == row["original_quote"]
            )
        assert data == before
    finally:
        await provider.close()


@pytest.mark.parametrize("quote", ["", "检修记录应注明部位；发现裂纹时附原始照片。"])
async def test_model_supplied_quote_is_rejected_even_if_empty_or_present_in_context(quote):
    source = {"source_id": "S1", "content": "检修记录应注明部位；发现裂纹时附原始照片。"}
    provider = DashScopeProvider(
        remote_settings(),
        httpx.MockTransport(lambda request: response(supported([{**selected(), "quote": quote}]))),
    )
    try:
        with pytest.raises(ProviderError, match="grade 未返回符合约束"):
            await provider.structured(GradeDecision, "grade", payload(source))
    finally:
        await provider.close()


@pytest.mark.parametrize(
    "evidence, failure",
    [
        ([selected("S1:E999")], "evidence_span_unknown"),
        ([selected(source_id="S9")], "citation_id_unknown"),
    ],
)
async def test_id_only_output_does_not_let_unknown_span_or_source_pass(evidence, failure):
    source = {"source_id": "S1", "content": "检修记录应注明部位。"}
    provider = DashScopeProvider(
        remote_settings(), httpx.MockTransport(lambda request: response(supported(evidence)))
    )
    try:
        decision = await provider.structured(GradeDecision, "grade", payload(source))
        bound, audit = bind_decision(decision, [source])
        assert not bound.passed and failure in audit["failure_types"]
    finally:
        await provider.close()


async def test_full_context_budget_fallback_restores_quote_schema_and_exact_text():
    source = {"source_id": "S1", "content": "检修记录应注明部位。\n" * 1000}
    data = payload(source)

    def handler(request):
        body = json.loads(request.content)
        actual = json.loads(body["messages"][1]["content"])
        assert actual["sources"] == [source]
        assert actual["evidence_protocol_mode"] == "continuous_quote_budget_fallback"
        assert "禁止quote字段" not in body["messages"][0]["content"]
        assert '"GradeEvidence"' in body["messages"][0]["content"]
        return response(
            supported(
                [
                    {
                        "source_id": "S1",
                        "quote": "检修记录应注明部位。",
                        "subject": "检修",
                        "attribute": "记录要求",
                    }
                ]
            )
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        decision = await provider.structured(GradeDecision, "grade", data)
        assert decision.evidence[0].quote == "检修记录应注明部位。"
        assert bind_decision(decision, [source])[0].passed
    finally:
        await provider.close()


def test_span_schema_keeps_support_evidence_and_scope_requirements():
    for invalid in [
        supported([]),
        {**supported([selected()]), "answer_scope": ""},
        {**supported([selected()]), "support": "insufficient"},
    ]:
        with pytest.raises(ValueError):
            GradeSpanDecision.model_validate(invalid)
