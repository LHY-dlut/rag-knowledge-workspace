import json

import httpx
import pytest
from pydantic import ValidationError

from app.check_protocol import answer_citation_manifest, bind_check
from app.check_scope import PREDICATES, literal_edit_view, validate_check_scope
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import CheckDecision, CheckScopeAssessment, CheckScopeDecision
from tests.test_check_scope import ANSWER, SOURCE, scope_reply, upstream
from tests.test_providers_parsers import remote_settings


def assessment(**changes):
    return {
        "checks": [
            {
                "answer_span_id": "A:E1",
                "source_ids": ["S1"],
                **{name: "preserved" for name in PREDICATES},
                "projection": {
                    "answer_modes": ["asserted"],
                    "answer_voices": ["fact"],
                    "source_modes": ["asserted"],
                    "source_voices": ["fact"],
                },
                **changes,
                "reason": "分类与对应语气先独立核对",
            }
        ]
    }


@pytest.mark.parametrize("name", PREDICATES)
@pytest.mark.parametrize("relation", ["changed", "undetermined"])
def test_backend_derives_every_negative_or_unknown_relation_as_false(name, relation):
    value = CheckScopeAssessment.model_validate(assessment(**{name: relation}))
    decision = value.to_decision()
    assert getattr(decision.checks[0], name) is False
    assert all(getattr(decision.checks[0], n) is True for n in PREDICATES if n != name)
    assert value.model_dump()["checks"][0][name] == relation


@pytest.mark.parametrize("bad", [True, "true", "supported", None])
def test_scope_wire_does_not_reaccept_old_boolean_or_unknown_enum(bad):
    with pytest.raises(ValidationError):
        CheckScopeAssessment.model_validate(assessment(modality_preserved=bad))


async def test_actual_provider_requests_enum_schema_and_derives_decision():
    def handle(request):
        body = json.loads(request.content)
        assert '"changed"' in body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(assessment(modality_preserved="changed"))}}
                ]
            },
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        decision = await provider.structured(CheckScopeDecision, "check_scope_binding", {})
        assert not decision.checks[0].modality_preserved
        assert decision.checks[0].subject_predicate_preserved
    finally:
        await provider.close()


def test_character_view_reconstructs_unmodified_unicode_text_and_exact_positions():
    answer = {"span_id": "A:E4", "quote": "预约已成为推动力。[S1]", "source_start": 31}
    evidence = {
        "source_id": "S1",
        "span_id": "S1:E5",
        "quote": "预约将成为推动力。",
        "source_start": 107,
    }
    view = literal_edit_view(answer, evidence)
    assert view["semantic_verdict"] is None
    assert "".join(x["source_text"] for x in view["edits"]) == evidence["quote"]
    assert "".join(x["answer_text"] for x in view["edits"]) == answer["quote"]
    for item in view["edits"]:
        assert (
            item["source_text"]
            == evidence["quote"][item["source_start"] - 107 : item["source_end"] - 107]
        )
        assert (
            item["answer_text"]
            == answer["quote"][item["answer_start"] - 31 : item["answer_end"] - 31]
        )
    assert any(x["source_text"] == "将" and x["answer_text"] == "已" for x in view["edits"])


async def test_separate_claim_requests_retain_whole_relevant_source_and_full_original_answer():
    sources = [
        {**SOURCE, "content": "预约将成为推动力。只有登记居民可预约。"},
        {**SOURCE, "source_id": "S2", "content": "设备已完成登记。目录暂未公开。"},
    ]
    answer = "预约将成为推动力。[S1]\n设备已完成登记。[S2]"
    span_ids = [item["answer_span_id"] for item in answer_citation_manifest(answer)]
    original = bind_check(
        CheckDecision(
            checks=[
                {
                    "answer_span_id": span_ids[0],
                    "verdict": "supported",
                    "reason": "构造位置校验",
                    "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                },
                {
                    "answer_span_id": span_ids[1],
                    "verdict": "supported",
                    "reason": "构造位置校验",
                    "evidence": [{"source_id": "S2", "span_id": "S2:E1"}],
                },
            ]
        ),
        answer,
        sources,
    )
    assert original.passed, original.reason
    calls = []

    class Provider:
        supports_check_scope_binding = True

        async def structured(self, schema, task, payload):
            calls.append(payload)
            assert payload["answer"] == answer and len(payload["answer_spans"]) == 1
            assert len(payload["sources"]) == len(payload["verified_literal_evidence"]) == 1
            assert payload["sources"][0] == sources[len(calls) - 1]
            assert payload["focus_answer_span_id"] == span_ids[len(calls) - 1]
            assert "literal_edit_view" not in payload
            assert [s["span_id"] for s in payload["answer_context_spans"]] == span_ids
            return scope_reply(payload, modality_preserved=len(calls) != 1)

    final, audit = await validate_check_scope(
        Provider(), original, answer, sources, "资料陈述了什么？"
    )
    assert not final.passed and len(calls) == audit["scope_model_requests"] == 2
    assert len(audit["checks"]) == 2 and len(audit["reading_views"]) == 2
    assert audit["literal_edits_only_in_audit_not_model_input"]
    assert all(
        v["semantic_verdict"] is None
        for r in audit["reading_views"]
        for v in r["literal_edit_view"]
    )
    assert final.checks == original.checks and original.passed


async def test_missing_upstream_claim_fails_before_any_new_paid_request():
    class Provider:
        supports_check_scope_binding = True

        async def structured(self, *args):
            pytest.fail("An incomplete upstream provenance record is rejected before dispatch")

    incomplete = upstream().model_copy(update={"checks": []})
    with pytest.raises(CheckProtocolError):
        await validate_check_scope(Provider(), incomplete, ANSWER, [SOURCE], "服务？")
