import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from app.agent import RAGAgent
from app.evidence_protocol import bind_decision, evidence_spans
from app.providers import DashScopeProvider
from app.schemas import (
    Citation,
    GradeBindingDecision,
    GradeDecision,
    Judgment,
    RetrievalConfig,
    RetrievalResult,
)
from tests.test_providers_parsers import remote_settings


def supported(items, limitation=False):
    return GradeDecision(
        passed=True,
        support="supported_limitation" if limitation else "supported_answer",
        reason="原文支持所问属性",
        answer_scope="仅限所提供设备维护制度，不能扩展至其他制度。",
        evidence=[dict(subject="设备维护", attribute="响应期限", **item) for item in items],
    )


def test_nonadjacent_facts_require_separate_real_span_references():
    source = {
        "source_id": "S1",
        "content": "首次响应为2小时。材料经管理员登记。维修完成时限未规定。",
    }
    spans = evidence_spans(source)
    invalid, _ = bind_decision(
        supported([dict(source_id="S1", quote="首次响应为2小时。维修完成时限未规定。")]), [source]
    )
    assert not invalid.passed  # Retain the original continuous-quote boundary.
    valid, audit = bind_decision(
        supported([dict(source_id="S1", span_id=spans[i]["span_id"]) for i in [0, 2]]), [source]
    )
    assert valid.passed and audit["passed"]
    assert [e.quote for e in valid.evidence] == ["首次响应为2小时。", "维修完成时限未规定。"]
    assert audit["evidence"][1]["source_start"] == source["content"].index("维修")
    assert all(e["model_quote"] == "" for e in audit["evidence"])


@pytest.mark.parametrize(
    "source_id,span_id,quote,error",
    [
        ("S99", "S1:E1", "", "citation_id_unknown"),
        ("S1", "S2:E1", "", "evidence_span_unknown"),
        ("S1", "S1:E999", "", "evidence_span_unknown"),
        ("S1", "S1:E1", "次日一定完成。", "evidence_span_quote_conflict"),
        ("S1", "S1:E1", "首次响应", "evidence_span_quote_conflict"),
    ],
)
def test_unknown_cross_source_or_conflicting_reference_fails_closed(
    source_id, span_id, quote, error
):
    source = {"source_id": "S1", "content": "首次响应为2小时。"}
    original = supported([dict(source_id=source_id, span_id=span_id, quote=quote)])
    before = original.model_dump()
    result, audit = bind_decision(original, [source])
    assert not result.passed and error in audit["failure_types"]
    assert original.model_dump() == before
    assert "original_quote" not in audit["evidence"][0]


def test_untrusted_attached_registry_cannot_manufacture_evidence():
    source = {
        "source_id": "S1",
        "content": "仅规定材料。",
        "evidence_spans": [
            dict(span_id="S1:E99", quote="保证次日到账。", source_start=0, source_end=8)
        ],
    }
    result, audit = bind_decision(supported([dict(source_id="S1", span_id="S1:E99")]), [source])
    assert not result.passed and audit["failure_types"] == ["evidence_span_unknown"]


def test_duplicate_text_uses_selected_position_and_duplicate_reference_is_audited():
    source = {"source_id": "S1", "content": "先登记。再核实。先登记。"}
    spans = evidence_spans(source)
    result, audit = bind_decision(
        supported([dict(source_id="S1", span_id=spans[2]["span_id"])] * 2), [source]
    )
    assert result.passed and len(audit["evidence"]) == 2
    assert all(e["source_start"] == 8 for e in audit["evidence"])


def test_span_positions_preserve_chinese_punctuation_newlines_and_long_text():
    text = "  金额1 00元；\r\n\t办理期限：未规定。\n" + "甲" * 803 + "。"
    source = {"source_id": "S1", "content": text}
    before = copy.deepcopy(source)
    spans = evidence_spans(source)
    assert source == before and all(0 < len(s["quote"]) <= 400 for s in spans)
    assert all(text[s["source_start"] : s["source_end"]] == s["quote"] for s in spans)
    covered = {i for s in spans for i in range(s["source_start"], s["source_end"])}
    assert all(i in covered for i, c in enumerate(text) if not c.isspace())
    bound, audit = bind_decision(
        supported([dict(source_id="S1", span_id=s["span_id"]) for s in spans]), [source]
    )
    assert bound.passed and all(m["match_mode"] == "server_span" for m in audit["evidence"])
    assert "1 00" in bound.evidence[0].quote


def test_empty_context_or_missing_reference_never_creates_evidence():
    with pytest.raises(ValidationError):
        supported([dict(source_id="S1")])
    result, audit = bind_decision(
        supported([dict(source_id="S1", span_id="S1:E1")]), [{"source_id": "S1", "content": " \n"}]
    )
    assert not result.passed and audit["failure_types"] == ["evidence_span_unknown"]


async def test_actual_provider_schema_and_literal_registry_without_duplicate_context():
    source = {"source_id": "S1", "content": "响应2小时。后续维修未规定。"}
    payload = {
        "query": "响应时限？",
        "sources": [{**source, "evidence_spans": evidence_spans(source)}],
    }
    before = copy.deepcopy(payload)

    def handler(request):
        body = json.loads(request.content)
        actual = json.loads(body["messages"][1]["content"])
        assert "content" not in actual["sources"][0]
        assert [s["quote"] for s in actual["sources"][0]["evidence_spans"]] == [
            "响应2小时。",
            "后续维修未规定。",
        ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": supported(
                                [dict(source_id="S1", span_id="S1:E1")]
                            ).model_dump_json(exclude={"evidence": {"__all__": {"quote"}}})
                        }
                    }
                ]
            },
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        raw = await provider.structured(GradeDecision, "grade", payload)
        bound, audit = bind_decision(raw, [source])
        assert bound.passed and bound.evidence[0].quote == "响应2小时。" and audit["passed"]
        assert payload == before
    finally:
        await provider.close()


@pytest.mark.parametrize("semantic_support", [True, False])
async def test_span_selection_does_not_override_subject_condition_or_scope(semantic_support):
    source = Citation(
        source_id="S1",
        document_id="d",
        parent_id="p",
        filename="制度.txt",
        location="全文",
        content="培训申请未规定期限。",
        child_ids=["c"],
    )
    tasks = []

    class Provider:
        async def structured(self, schema, task, payload):
            tasks.append(task)
            if task == "grade":
                assert payload["sources"][0]["evidence_spans"][0]["span_id"] == "S1:E1"
                return supported([dict(source_id="S1", span_id="S1:E1")], limitation=True)
            assert task == "grade_binding" and payload["evidence"][0]["quote"] == source.content
            return GradeBindingDecision(
                subject_matches=semantic_support,
                attribute_matches=semantic_support,
                explicit_limitation=semantic_support,
                scope_supported=semantic_support,
                reason="同主体属性可限定回答" if semantic_support else "培训规则不能迁移到维修",
            )

    state = dict(
        query="培训申请必须3天内办结吗？" if semantic_support else "维修必须3天内办结吗？",
        mode="agent",
        retrieval=RetrievalResult(sources=[source]),
        grade_retries=3,
        kb=SimpleNamespace(config=RetrievalConfig().model_dump()),
    )
    result = await RAGAgent(Provider(), None).grade(state)
    assert result["grade_protocol"]["passed"]
    assert result["grade"].passed is semantic_support and not result["grade_retry_allowed"]
    assert tasks == ["grade", "grade_binding"]


async def test_span_grade_still_cannot_publish_an_unsupported_fact(monkeypatch):
    import app.agent as module

    monkeypatch.setattr(module, "get_stream_writer", lambda: lambda event: None)
    source = Citation(
        source_id="S1",
        document_id="d",
        parent_id="p",
        filename="制度.txt",
        location="全文",
        content="提交材料不承诺到账时间。",
        child_ids=["c"],
    )
    bound, _ = bind_decision(
        supported([dict(source_id="S1", span_id="S1:E1")], limitation=True), [source.model_dump()]
    )

    class Provider:
        async def structured(self, schema, task, payload):
            assert task == "check" and payload["sources"][0]["content"] == source.content
            return Judgment(passed=False, reason="原文不承诺到账，不能保证次日到账")

    result = await RAGAgent(Provider(), None).check(
        dict(
            query="次日到账？",
            draft="保证次日到账[S1]",
            grade=bound,
            mode="agent",
            check_retries=2,
            kb=SimpleNamespace(config=RetrievalConfig().model_dump()),
            retrieval=RetrievalResult(sources=[source]),
        )
    )
    assert not result["check"].passed and not result["check_retry_allowed"]
    assert "answer" not in result and result["draft"] == ""
