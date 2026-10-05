import copy
import json

import httpx
import pytest

from app.answer_parts import bind_answer_parts
from app.answer_ranges import answer_fragments
from app.check_protocol import answer_spans
from app.check_scope import validate_check_scope
from app.grounded_relations import GroundedRelations
from app.output_contract import check_output_contract
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import AnswerRangePredicates
from app.source_parts import SourcePredicateParts
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_explicit_focus import ExplicitFocusProvider
from tests.test_predicate_ranges import selection
from tests.test_providers_parsers import remote_settings


@pytest.mark.parametrize(
    "text",
    [
        "依据本资料，审批后可借用。[S1]",
        "按2022年办法，当时可以申请。[S1]",
        "在西区试点内，审核后才可领取。[S1]",
    ],
)
def test_required_prefix_is_explicit_and_omission_still_fails(text):
    focus = answer_spans(text)[0]
    payload = {
        "answer": text,
        "answer_fragments": answer_fragments(focus, text),
        "focus_answer_span_id": focus["span_id"],
    }
    contract = check_output_contract(
        "check_answer_predicate_parts", payload, AnswerRangePredicates.model_json_schema()
    )
    ids = contract["required_fragment_ids"]
    assert ids == ["A:E1:F1", "A:E1:F2"]
    complete = AnswerRangePredicates(
        answer_span_id="A:E1", predicates=[selection(ids, mode="permission")]
    )
    assert len(bind_answer_parts(complete, focus, text)[0]["answer_parts"]) == 2
    incomplete = AnswerRangePredicates(
        answer_span_id="A:E1", predicates=[selection(ids[1:], mode="permission")]
    )
    with pytest.raises(CheckProtocolError, match="遗漏"):
        bind_answer_parts(incomplete, focus, text)


def test_contract_separates_axes_without_mutating_schema_or_input():
    focus = answer_spans(GOOD)[0]
    data = {
        "answer": GOOD,
        "answer_fragments": answer_fragments(focus, GOOD),
        "focus_answer_span_id": "A:E1",
    }
    schema = AnswerRangePredicates.model_json_schema()
    before = copy.deepcopy((data, schema))
    contract = check_output_contract("check_answer_predicate_parts", data, schema)
    assert (data, schema) == before
    assert (
        "opinion" in contract["voice_allowed_values"]
        and "opinion" not in contract["mode_allowed_values"]
    )
    assert contract["reason_target_characters"] == 80 and contract["reason_max_characters"] == 300
    assert contract["semantic_support_not_prejudged"]


def test_source_contract_uses_source_positions_and_contains_no_answer_or_question():
    source = {"source_id": "R7", "content": "说明：本区审核后可申请。"}
    from app.evidence_protocol import evidence_spans

    focus = evidence_spans(source)[0]
    data = {
        "source_id": "R7",
        "source_text": source["content"],
        "focus_source_span": focus,
        "source_fragments": answer_fragments(focus, source["content"]),
    }
    contract = check_output_contract(
        "check_source_predicate_parts", data, SourcePredicateParts.model_json_schema()
    )
    assert contract["expected_source_id"] == "R7" and contract["expected_source_span_id"] == "R7:E1"
    assert contract["required_fragment_ids"] == ["R7:E1:F1", "R7:E1:F2"]
    assert not {"answer", "query", "sources", "grade", "score"} & contract.keys()


async def run_wire(invalid=None):
    fixture = ExplicitFocusProvider()
    captured = []

    async def handle(request):
        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        captured.append((body, data))
        if "focus_source_span" in data:
            task, schema = "check_source_predicate_parts", SourcePredicateParts
        elif "source_predicates" in data:
            task, schema = "check_grounded_relations", GroundedRelations
        else:
            task, schema = "check_answer_predicate_parts", AnswerRangePredicates
        value = (await fixture.structured(schema, task, data)).model_dump()
        if invalid == "mode" and task == "check_answer_predicate_parts":
            value["predicates"][0]["mode"] = "opinion"
        if invalid == "reason" and task == "check_grounded_relations":
            value["checks"][0]["reason"] = "长" * 301
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(value, ensure_ascii=False)}}]}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
        return result, audit, captured
    finally:
        await provider.close()


async def test_actual_provider_exposes_unchanged_limits_and_IDs_for_all_three_tasks():
    result, audit, captured = await run_wire()
    assert result.passed and len(captured) == audit["scope_model_requests"] == 4
    for body, data in captured:
        contract = data["output_contract"]
        assert (
            contract["reason_max_characters"] == 300 and contract["semantic_support_not_prejudged"]
        )
        assert "协议完整不等于事实获支持" in body["messages"][0]["content"]
        if "source_predicates" in data:
            assert contract["expected_predicate_id"] == data["focus_predicate_id"]
            assert contract["checks_count"] == 1
            assert set(contract["relation_values"]) == {"preserved", "changed", "undetermined"}
        else:
            assert len(contract["required_fragment_ids"]) == 2
            assert "opinion" not in contract["mode_allowed_values"]


async def test_invalid_mode_is_rejected_without_response_repair():
    with pytest.raises(CheckProtocolError):
        await run_wire("mode")


async def test_overlong_reason_is_rejected_without_truncation():
    with pytest.raises(CheckProtocolError):
        await run_wire("reason")
