"""Synthetic grammar/coverage boundaries; Mock results are not semantic quality."""

import json

import httpx
import pytest

from app.atomic_scope import review_atomic_scope
from app.evidence_protocol import evidence_spans
from app.grounded_relations import GroundedRelations
from app.grounded_selection import parse_grounded_selection
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import AnswerRangePredicate, AnswerRangePredicates
from app.settings import Settings
from app.source_windows import SourceWindowParts
from tests.test_predicate_reading import FIELDS


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "missing_prefix", "extra_limitation", *FIELDS])
async def test_shared_attribution_keeps_full_coverage_without_waiving_any_veto(failure):
    source = {"source_id": "S1", "content": "工作人员发放抽签号码，离场时回收号码。"}
    answer = "照当天登记表，工作人员发放抽签号码，离场时回收号码。[S1]"
    focus = {"span_id": "A1", "source_start": 0, "source_end": len(answer), "quote": answer}

    class Provider:
        supports_check_predicate_parts = True
        supports_check_single_predicate_scope = True
        supports_check_source_predicate_parts = True
        supports_check_grounded_relations = True
        supports_check_explicit_predicate_focus = True
        supports_check_source_windows = True

        async def structured(self, schema, task, payload):
            if task == "check_answer_predicate_parts":
                parts = payload["answer_fragments"]
                assert len(parts) == 3
                prefix = [] if failure == "missing_prefix" else [parts[0]["fragment_id"]]
                predicates = [
                    AnswerRangePredicate(
                        predicate_id=name,
                        fragment_ids=prefix + [parts[index]["fragment_id"]],
                        mode="asserted",
                        voice="fact",
                        reason="Mock共享出处，真实位置完整覆盖",
                    )
                    for name, index in (("issue", 1), ("return", 2))
                ]
                if failure == "extra_limitation":
                    predicates.append(
                        AnswerRangePredicate(
                            predicate_id="fabricated",
                            fragment_ids=[p["fragment_id"] for p in parts],
                            mode="document_limitation",
                            voice="fact",
                            reason="Mock故意制造额外类别；后端必须保留否决，不能删掉它来通过",
                        )
                    )
                return AnswerRangePredicates(answer_span_id="A1", predicates=predicates)
            if task == "check_source_predicate_window":
                assert not {"query", "answer", "grade", "score"} & payload.keys()
                parts = payload["source_fragments"]
                assert len(parts) == 2
                return SourceWindowParts(
                    source_id="S1",
                    source_window_id=payload["focus_source_window"]["window_id"],
                    predicates=[
                        AnswerRangePredicate(
                            predicate_id=name,
                            fragment_ids=[parts[index]["fragment_id"]],
                            mode="asserted",
                            voice="fact",
                            reason="Mock独立原文关系",
                        )
                        for name, index in (("issue", 0), ("return", 1))
                    ],
                )
            assert task == "check_grounded_relations"
            registry = {
                p["source_predicate_id"].split(":")[-1]: p for p in payload["source_predicates"]
            }
            pid = payload["focus_predicate_id"]
            selected = registry["issue" if pid == "fabricated" else pid]
            value = {
                "answer_span_id": "A1",
                "checks": [
                    {
                        "predicate_id": pid,
                        "source_predicate_ids": [selected["source_predicate_id"]],
                        "context_source_predicate_ids": [],
                        **{k: "changed" if k == failure else "preserved" for k in FIELDS},
                        "reason": "Mock原始关系结果保持，不修补类别或否决",
                    }
                ],
            }
            return parse_grounded_selection(json.dumps(value), payload)

    args = (
        Provider(),
        answer,
        focus,
        [focus],
        [source],
        {"S1"},
        "登记流程？",
        [{"source_id": "S1", **p} for p in evidence_spans(source)],
    )
    if failure == "missing_prefix":
        with pytest.raises(CheckProtocolError, match="遗漏"):
            await review_atomic_scope(*args)
        return
    result, _ = await review_atomic_scope(*args)
    assert all(result[k] for k in FIELDS) is (failure is None)
    assert len(result["predicate_checks"]) == (3 if failure == "extra_limitation" else 2)
    if failure in FIELDS:
        assert result[failure] is False
    if failure is None:
        for check in result["predicate_checks"]:
            parts = check["answer_predicate"]["answer_parts"]
            assert parts[0]["quote"] == "照当天登记表，"
            assert all(answer[p["source_start"] : p["source_end"]] == p["quote"] for p in parts)


@pytest.mark.asyncio
async def test_comparison_prompt_requests_only_available_full_context_and_retains_current_focus():
    from tests.test_relation_view import inputs
    from tests.test_source_windows import selection

    registry, payload, _ = inputs()
    captured = []

    def respond(request):
        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        captured.append((body, data))
        value = selection(data["source_predicates"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})

    cfg = Settings(
        _env_file=None,
        check_source_window_projection=True,
        dashscope_chat_base_url="https://example.test/v1",
    )
    provider = DashScopeProvider(cfg, httpx.MockTransport(respond))
    try:
        decision = await provider.structured(GroundedRelations, "check_grounded_relations", payload)
    finally:
        await provider.close()
    body, data = captured[0]
    assert "focus_answer_reading_span" not in body["messages"][0]["content"]
    assert "answer_context_spans" in body["messages"][0]["content"]
    assert data["answer_context_spans"] == payload["answer_context_spans"]
    assert data["focus_answer_predicate"] == payload["focus_answer_predicate"]
    assert decision.checks[0].source_predicate_ids == [registry[0]["source_predicate_id"]]
    assert "query" not in data
