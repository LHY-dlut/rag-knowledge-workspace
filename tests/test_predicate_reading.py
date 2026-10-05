"""Mock projection granularity and unchanged veto guards; no semantic quality claim."""

import json

import pytest

from app.answer_ranges import answer_fragments
from app.atomic_scope import review_atomic_scope
from app.evidence_protocol import evidence_spans
from app.grounded_selection import parse_grounded_selection
from app.schemas import AnswerRangePredicate, AnswerRangePredicates
from app.source_windows import SourceWindowParts

FIELDS = (
    "subject_predicate_preserved",
    "modality_preserved",
    "temporal_scope_preserved",
    "conditions_preserved",
    "attribution_preserved",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "swapped_primary", "mixed_primary", *FIELDS])
async def test_distinct_proposition_and_its_property_preserve_exact_pairing_and_vetoes(failure):
    source = {"source_id": "S1", "content": "顾问宋岚预计登记系统将简化审核。这份意见属于预测。"}
    answer = "顾问宋岚预计登记系统将简化审核，这份意见属于预测。[S1]"
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
                parts = answer_fragments(focus, answer)
                return AnswerRangePredicates(
                    answer_span_id="A1",
                    predicates=[
                        AnswerRangePredicate(
                            predicate_id="forecast",
                            fragment_ids=[parts[0]["fragment_id"]],
                            mode="future",
                            voice="opinion",
                            reason="Mock预测关系",
                        ),
                        AnswerRangePredicate(
                            predicate_id="property",
                            fragment_ids=[parts[1]["fragment_id"]],
                            mode="asserted",
                            voice="fact",
                            reason="Mock实际报告预测性质",
                        ),
                    ],
                )
            if task == "check_source_predicate_window":
                assert "query" not in payload and "answer" not in payload
                parts = payload["source_fragments"]
                return SourceWindowParts(
                    source_id="S1",
                    source_window_id=payload["focus_source_window"]["window_id"],
                    predicates=[
                        AnswerRangePredicate(
                            predicate_id="forecast",
                            fragment_ids=[parts[0]["fragment_id"]],
                            mode="future",
                            voice="opinion",
                            reason="Mock独立来源预测",
                        ),
                        AnswerRangePredicate(
                            predicate_id="property",
                            fragment_ids=[parts[1]["fragment_id"]],
                            mode="asserted",
                            voice="fact",
                            reason="Mock独立来源性质",
                        ),
                    ],
                )
            assert task == "check_grounded_relations"
            ids = {
                p["source_predicate_id"].split(":")[-1]: p["source_predicate_id"]
                for p in payload["source_predicates"]
            }
            assert all("mode" not in p and "voice" not in p for p in payload["source_predicates"])
            pid = payload["focus_predicate_id"]
            selected = [ids[pid]]
            if failure == "swapped_primary":
                selected = [ids["property" if pid == "forecast" else "forecast"]]
            if failure == "mixed_primary":
                selected = list(ids.values())
            flags = {k: "changed" if k == failure else "preserved" for k in FIELDS}
            value = {
                "answer_span_id": "A1",
                "checks": [
                    {
                        "predicate_id": pid,
                        "source_predicate_ids": selected,
                        "context_source_predicate_ids": [],
                        **flags,
                        "reason": "Mock原始判断，不因文本限定或理由看似正确而覆盖changed",
                    }
                ],
            }
            return parse_grounded_selection(json.dumps(value), payload)

    result, _ = await review_atomic_scope(
        Provider(),
        answer,
        focus,
        [focus],
        [source],
        {"S1"},
        "意见与其性质？",
        [{"source_id": "S1", **s} for s in evidence_spans(source)],
    )
    assert all(result[k] for k in FIELDS) is (failure is None)
    assert len(result["predicate_checks"]) == 2
    if failure is None:
        byid = {p["predicate_id"]: p for p in result["predicate_checks"]}
        assert byid["forecast"]["source_mode"] == "future"
        assert byid["property"]["source_mode"] == "asserted"
    if failure in FIELDS:
        assert result[failure] is False
        assert all(
            p["reason"] == "Mock原始判断，不因文本限定或理由看似正确而覆盖changed"
            for p in result["predicate_checks"]
        )
