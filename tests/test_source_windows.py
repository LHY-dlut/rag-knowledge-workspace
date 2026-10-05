import json
from copy import deepcopy

import httpx
import pytest

from app.atomic_scope import review_atomic_scope
from app.evidence_protocol import evidence_spans
from app.grounded_relations import bind_grounded_relations
from app.grounded_selection import parse_grounded_selection
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import AnswerRangePredicate, AnswerRangePredicates
from app.settings import Settings
from app.source_windows import (
    SourceWindowParts,
    bind_source_window,
    predicate_refs,
    project_source_windows,
    source_windows,
    window_fragments,
)

SOURCE = {
    "source_id": "S1",
    "content": "顾问沈宁预计移动登记将减少排队。这是顾问对未来的判断。另有统计：去年服务120人。",
}


def decision(source=SOURCE):
    window = source_windows(source)[0]
    parts = window_fragments(source, window)
    return SourceWindowParts(
        source_id=source["source_id"],
        source_window_id=window["window_id"],
        predicates=[
            AnswerRangePredicate(
                predicate_id="forecast",
                fragment_ids=[p["fragment_id"] for p in parts[:2]],
                mode="future",
                voice="opinion",
                reason="同一顾问的预测及其性质说明",
            ),
            AnswerRangePredicate(
                predicate_id="past_count",
                fragment_ids=[p["fragment_id"] for p in parts[2:]],
                mode="asserted",
                voice="fact",
                reason="独立去年的服务统计",
            ),
        ],
    )


def bound():
    return bind_source_window(decision(), SOURCE, source_windows(SOURCE)[0])


def selection_payload(registry):
    return {
        "focus_answer_span_id": "A1",
        "focus_predicate_id": "P1",
        "source_predicates": registry,
        "sources": [SOURCE],
        "actual_cited_source_ids": ["S1"],
    }


def selection(registry):
    return {
        "answer_span_id": "A1",
        "checks": [
            {
                "predicate_id": "P1",
                "source_predicate_ids": [registry[0]["source_predicate_id"]],
                "context_source_predicate_ids": [],
                **{
                    k: "preserved"
                    for k in (
                        "subject_predicate_preserved",
                        "modality_preserved",
                        "temporal_scope_preserved",
                        "conditions_preserved",
                        "attribution_preserved",
                    )
                },
                "reason": "Mock关系判定仅测试位置绑定，不能证明语义",
            }
        ],
    }


def test_cross_sentence_qualifier_keeps_each_exact_original_position():
    rows = bound()
    assert rows[0]["mode"] == "future" and rows[0]["voice"] == "opinion"
    assert predicate_refs(rows[0]) == {("S1", "S1:E1"), ("S1", "S1:E2")}
    assert rows[1]["mode"] == "asserted"
    for row in rows:
        assert row["literal_binding"]["semantic_support_checked"] is False
        assert row["literal_binding"]["composite_quote_created"] is False
        for p in row["source_parts"]:
            assert SOURCE["content"][p["source_start"] : p["source_end"]] == p["quote"]


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_fragment",
        "missing_fact",
        "reordered",
        "wrong_source",
        "wrong_window",
        "modified_window",
    ],
)
def test_window_rejects_invalid_or_incomplete_projection(mutation):
    value = decision().model_dump()
    window = deepcopy(source_windows(SOURCE)[0])
    if mutation == "unknown_fragment":
        value["predicates"][0]["fragment_ids"][0] = "S2:E1:F1"
    if mutation == "missing_fact":
        value["predicates"].pop()
    if mutation == "reordered":
        value["predicates"][0]["fragment_ids"].reverse()
    if mutation == "wrong_source":
        value["source_id"] = "S2"
    if mutation == "wrong_window":
        value["source_window_id"] = "S1:W2"
    if mutation == "modified_window":
        window["quote"] += "不存在的条件"
    with pytest.raises(CheckProtocolError):
        bind_source_window(SourceWindowParts.model_validate(value), SOURCE, window)


def test_bounded_windows_cover_all_canonical_spans_without_skipping_long_text():
    source = {"source_id": "S1", "content": ("第一主体保持原规则。\n" * 100) + ("乙" * 1400)}
    windows = source_windows(source)
    expected = evidence_spans(source)
    assert [s for w in windows for s in w["evidence_spans"]] == expected
    assert all(
        w["source_end"] - w["source_start"] <= 800 and len(w["evidence_spans"]) <= 8
        for w in windows
    )
    assert all(
        source["content"][w["source_start"] : w["source_end"]] == w["quote"] for w in windows
    )


def test_backend_derives_all_cross_sentence_evidence_without_constructing_a_new_span():
    registry = bound()
    parsed = parse_grounded_selection(json.dumps(selection(registry)), selection_payload(registry))
    assert [(r.source_id, r.span_id) for r in parsed.checks[0].evidence] == [
        ("S1", "S1:E1"),
        ("S1", "S1:E2"),
    ]
    result = bind_grounded_relations(parsed, registry)
    assert result.checks[0].source_mode == "future"
    assert result.checks[0].source_voice == "opinion"


@pytest.mark.parametrize(
    "mutation",
    [
        "quote",
        "start_bool",
        "end",
        "span_id",
        "window_id",
        "refs_missing",
        "refs_extra",
        "uncited_source",
    ],
)
def test_selection_revalidates_real_window_positions(mutation):
    registry = deepcopy(bound())
    payload = selection_payload(registry)
    first = registry[0]
    if mutation == "quote":
        first["source_parts"][0]["quote"] = "伪造原文"
    if mutation == "start_bool":
        first["source_parts"][0]["source_start"] = False
    if mutation == "end":
        first["source_parts"][0]["source_end"] += 1
    if mutation == "span_id":
        first["source_parts"][0]["span_id"] = "S1:E3"
    if mutation == "window_id":
        first["source_window_id"] = "S1:W999"
    if mutation == "refs_missing":
        first["evidence_refs"].pop()
    if mutation == "refs_extra":
        first["evidence_refs"].append({"source_id": "S1", "span_id": "S1:E3"})
    if mutation == "uncited_source":
        payload["actual_cited_source_ids"] = []
    with pytest.raises(CheckProtocolError):
        parse_grounded_selection(json.dumps(selection(registry)), payload)


def test_legacy_explicit_window_evidence_cannot_omit_cross_sentence_support():
    registry = bound()
    value = selection(registry)
    value["checks"][0]["evidence"] = [{"source_id": "S1", "span_id": "S1:E1"}]
    parsed = parse_grounded_selection(json.dumps(value), selection_payload(registry))
    with pytest.raises(CheckProtocolError):
        bind_grounded_relations(parsed, registry)


@pytest.mark.asyncio
async def test_source_window_projection_never_receives_question_or_answer_and_caches_exact_source():
    class Projection:
        def __init__(self):
            self.inputs = []

        async def structured(self, schema, task, payload):
            assert schema is SourceWindowParts and task == "check_source_predicate_window"
            assert set(payload) == {
                "source_id",
                "source_text",
                "focus_source_window",
                "source_fragments",
            }
            self.inputs.append(payload)
            return decision()

    provider = Projection()
    cache = {}
    first, calls = await project_source_windows(provider, [SOURCE], cache)
    second, second_calls = await project_source_windows(provider, [SOURCE], cache)
    assert first == second and calls == 1 and second_calls == 0 and len(provider.inputs) == 1


@pytest.mark.asyncio
async def test_new_wire_uses_only_registered_fragments_separate_categories_and_all_coverage():
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": decision().model_dump_json()}}]}
        )

    cfg = Settings(
        _env_file=None,
        check_source_window_projection=True,
        dashscope_chat_base_url="https://example.test/v1",
    )
    provider = DashScopeProvider(cfg, httpx.MockTransport(respond))
    try:
        await project_source_windows(provider, [SOURCE], {})
    finally:
        await provider.close()
    system = requests[0]["messages"][0]["content"]
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["output_contract"]["expected_source_window_id"] == "S1:W1"
    assert payload["output_contract"]["mode_and_voice_are_separate_fields"]
    assert payload["output_contract"]["required_fragment_ids"] == [
        p["fragment_id"] for p in window_fragments(SOURCE, source_windows(SOURCE)[0])
    ]
    assert "独立读取source_text" in system
    assert not {"query", "answer", "system_results", "candidate_scores"} & payload.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "subject_predicate_preserved",
        "modality_preserved",
        "temporal_scope_preserved",
        "conditions_preserved",
        "attribution_preserved",
        "mixed_primary",
    ],
)
async def test_window_path_preserves_all_five_vetoes_and_exact_independent_category(failure):
    answer = "顾问沈宁预计移动登记将减少排队，属于未来判断。[S1]"
    focus = {"span_id": "A1", "source_start": 0, "source_end": len(answer), "quote": answer}

    class Fake:
        supports_check_predicate_parts = True
        supports_check_single_predicate_scope = True
        supports_check_source_predicate_parts = True
        supports_check_grounded_relations = True
        supports_check_explicit_predicate_focus = True
        supports_check_source_windows = True

        async def structured(self, schema, task, payload):
            if task == "check_answer_predicate_parts":
                return AnswerRangePredicates(
                    answer_span_id="A1",
                    predicates=[
                        AnswerRangePredicate(
                            predicate_id="P1",
                            fragment_ids=[p["fragment_id"] for p in payload["answer_fragments"]],
                            mode="future",
                            voice="opinion",
                            reason="Mock预测及性质说明",
                        )
                    ],
                )
            if task == "check_source_predicate_window":
                return decision()
            assert task == "check_grounded_relations"
            assert payload["focus_answer_reading_span"] == focus
            assert all("mode" not in p and "voice" not in p for p in payload["source_predicates"])
            value = selection(payload["source_predicates"])
            if failure == "mixed_primary":
                value["checks"][0]["source_predicate_ids"].append(
                    payload["source_predicates"][1]["source_predicate_id"]
                )
            elif failure:
                value["checks"][0][failure] = "changed"
            return parse_grounded_selection(json.dumps(value), payload)

    source_evidence = [{"source_id": "S1", **s} for s in evidence_spans(SOURCE)]
    result, _ = await review_atomic_scope(
        Fake(), answer, focus, [focus], [SOURCE], {"S1"}, "顾问的判断？", source_evidence
    )
    fields = (
        "subject_predicate_preserved",
        "modality_preserved",
        "temporal_scope_preserved",
        "conditions_preserved",
        "attribution_preserved",
    )
    assert all(result[k] for k in fields) is (failure is None)
    assert result["source_projection_has_question_or_answer"] is False
