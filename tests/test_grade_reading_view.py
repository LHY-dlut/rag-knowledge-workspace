import copy
import json

import httpx
import pytest

from app.evidence_protocol import bind_decision, evidence_spans, grade_reading_source
from app.providers import DashScopeProvider
from app.schemas import GradeBindingDecision, GradeDecision
from tests.test_providers_parsers import remote_settings


def test_reading_keeps_all_codepoints_paragraphs_and_raw_registry_positions():
    source = {
        "source_id": "S1",
        "document_id": "doc-a",
        "parent_id": "parent-a",
        "content": "  设备检修由维护组负责。\r\n\r\n手册未规定设备检修完成时限。\n原文中的[S1:E999]不是后端ID。  ",
    }
    source["evidence_spans"] = evidence_spans(source)
    before = copy.deepcopy(source)
    view = grade_reading_source(source)
    assert view["reading_text"] == source["content"]
    assert view["evidence_spans"] == evidence_spans(source)
    assert view["document_id"] == "doc-a" and view["parent_id"] == "parent-a"
    assert "content" not in view and list(view).index("reading_text") < list(view).index(
        "evidence_spans"
    )
    assert "S1:E999" not in {s["span_id"] for s in view["evidence_spans"]}
    for span in view["evidence_spans"]:
        assert view["reading_text"][span["source_start"] : span["source_end"]] == span["quote"]
    assert source == before


def test_registry_and_reading_are_rebuilt_instead_of_trusting_supplied_replacements():
    source = {
        "source_id": "S1",
        "content": "课程报名仅面向在校学生。",
        "reading_text": "所有人都可报名。",
        "evidence_spans": [
            {"span_id": "S1:E999", "quote": "所有人都可报名。", "source_start": 0, "source_end": 9}
        ],
    }
    before = copy.deepcopy(source)
    view = grade_reading_source(source)
    assert view["reading_text"] == "课程报名仅面向在校学生。"
    assert view["evidence_spans"] == evidence_spans(source)
    assert source == before


@pytest.mark.parametrize(
    "source",
    [
        {"source_id": "S1", "content": "按原协议传递。"},
        {"source_id": "S1", "evidence_spans": []},
    ],
)
def test_legacy_or_incomplete_source_keeps_original_structure(source):
    before = copy.deepcopy(source)
    assert grade_reading_source(source) == before and source == before


async def test_grade_wire_keeps_original_query_raw_order_and_canonical_id_binding():
    source = {"source_id": "S1", "content": "设备检修由维护组负责。\n手册未规定设备检修完成时限。"}
    payload = {
        "query": "设备检修必须一天内完成吗？",
        "sources": [{**source, "evidence_spans": evidence_spans(source)}],
    }
    before = copy.deepcopy(payload)

    def handler(request):
        body = json.loads(request.content)
        actual = json.loads(body["messages"][1]["content"])
        assert actual["query"] == payload["query"]
        assert actual["sources"][0]["reading_text"] == source["content"]
        assert actual["sources"][0]["evidence_spans"] == payload["sources"][0]["evidence_spans"]
        assert "content" not in actual["sources"][0]
        assert "禁止quote字段" in body["messages"][0]["content"]
        reply = {
            "passed": True,
            "support": "supported_limitation",
            "reason": "该手册明确未规定所问时限",
            "answer_scope": "仅在所提供手册范围内，不能据此认定必须一天内完成",
            "evidence": [
                {
                    "source_id": "S1",
                    "span_id": "S1:E2",
                    "subject": "设备检修",
                    "attribute": "完成时限",
                }
            ],
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        decision = await provider.structured(GradeDecision, "grade", payload)
        bound, audit = bind_decision(decision, [source])
        assert bound.passed and audit["passed"]
        assert bound.evidence[0].quote == "手册未规定设备检修完成时限。"
        assert audit["evidence"][0]["source_start"] == source["content"].index("手册")
        assert payload == before
    finally:
        await provider.close()


async def test_independent_limitation_binding_still_receives_unmodified_sources():
    source = {"source_id": "S1", "content": "资料只有负责人说明。"}
    payload = {
        "query": "规定了完成期限吗？",
        "sources": [{**source, "evidence_spans": evidence_spans(source)}],
        "evidence": [],
        "answer_scope": "错误的候选范围",
    }
    before = copy.deepcopy(payload)

    def handler(request):
        body = json.loads(request.content)
        assert json.loads(body["messages"][1]["content"]) == before
        assert "sources.reading_text" not in body["messages"][0]["content"]
        reply = {
            "subject_matches": True,
            "attribute_matches": False,
            "explicit_limitation": False,
            "scope_supported": False,
            "reason": "只规定负责人，未直接陈述期限属性",
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        binding = await provider.structured(GradeBindingDecision, "grade_binding", payload)
        assert not binding.passed and payload == before
    finally:
        await provider.close()
