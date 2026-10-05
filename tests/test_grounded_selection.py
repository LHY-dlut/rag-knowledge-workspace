import json
from copy import deepcopy

import httpx
import pytest
from pydantic import ValidationError

from app.answer_ranges import answer_fragments
from app.check_scope import PREDICATES, validate_check_scope
from app.evidence_protocol import evidence_spans
from app.grounded_relations import GroundedRelations, bind_grounded_relations
from app.grounded_selection import parse_grounded_selection, selection_wire
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import AnswerRangePredicates
from app.source_parts import SourcePredicateParts, bind_source_parts
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_explicit_focus import ExplicitFocusProvider
from tests.test_providers_parsers import remote_settings


def payload_and_response():
    source = {
        "source_id": "S1",
        "content": "试行范围：完成审批后可以借用工具。该规则已在次年废止。",
    }
    parts = []
    for ordinal, focus in enumerate(evidence_spans(source)):
        value = SourcePredicateParts.model_validate(
            {
                "source_id": "S1",
                "source_span_id": focus["span_id"],
                "predicates": [
                    {
                        "predicate_id": "P1",
                        "fragment_ids": [
                            f["fragment_id"] for f in answer_fragments(focus, source["content"])
                        ],
                        "mode": "permission" if ordinal == 0 else "asserted",
                        "voice": "fact",
                        "reason": "对应原文关系",
                    }
                ],
            }
        )
        parts.extend(bind_source_parts(value, source, focus))
    response = {
        "answer_span_id": "A:E1",
        "checks": [
            {
                "predicate_id": "A:E1:P1",
                "source_predicate_ids": [parts[0]["source_predicate_id"]],
                "context_source_predicate_ids": [parts[1]["source_predicate_id"]],
                "reason": "仅核验历史许可，废止仍为必要范围上下文。",
                **{n: "preserved" for n in PREDICATES},
            }
        ],
    }
    payload = {
        "focus_answer_span_id": "A:E1",
        "focus_predicate_id": "A:E1:P1",
        "sources": [source],
        "source_predicates": parts,
        "actual_cited_source_ids": ["S1"],
    }
    return payload, response


def test_selected_ids_bind_exact_existing_positions_without_changing_inputs():
    payload, response = payload_and_response()
    frozen = deepcopy((payload, response))
    decision = parse_grounded_selection(json.dumps(response), payload)
    assert [r.span_id for r in decision.checks[0].evidence] == ["S1:E1", "S1:E2"]
    bound = bind_grounded_relations(decision, payload["source_predicates"])
    assert bound.checks[0].source_mode == "permission" and (payload, response) == frozen


@pytest.mark.parametrize("field", PREDICATES)
def test_position_binding_cannot_turn_changed_or_undetermined_into_supported(field):
    payload, response = payload_and_response()
    response["checks"][0][field] = "changed"
    bound = bind_grounded_relations(
        parse_grounded_selection(json.dumps(response), payload), payload["source_predicates"]
    )
    assert getattr(bound.checks[0], field) == "changed"


@pytest.mark.parametrize(
    "bad",
    [
        "unknown",
        "uncited",
        "offset",
        "quote",
        "fragment",
        "duplicate_registry",
        "duplicate_source",
        "missing_parts",
        "boolean_offset",
    ],
)
def test_invalid_registry_or_source_provenance_cannot_create_positions(bad):
    payload, response = payload_and_response()
    if bad == "unknown":
        response["checks"][0]["context_source_predicate_ids"] = ["unknown"]
    elif bad == "uncited":
        payload["actual_cited_source_ids"] = []
    elif bad == "duplicate_registry":
        payload["source_predicates"].append(deepcopy(payload["source_predicates"][0]))
    elif bad == "duplicate_source":
        payload["sources"].append(deepcopy(payload["sources"][0]))
    elif bad == "missing_parts":
        payload["source_predicates"][0]["source_parts"] = []
    else:
        part = payload["source_predicates"][0]["source_parts"][0]
        if bad == "offset":
            part["source_end"] += 1
        elif bad == "quote":
            part["quote"] = "其他主体的许可"
        elif bad == "boolean_offset":
            part["source_start"] = False
        else:
            part["fragment_id"] = "S1:E99:F1"
    with pytest.raises(CheckProtocolError):
        parse_grounded_selection(json.dumps(response), payload)


@pytest.mark.parametrize(
    "bad", ["empty_primary", "overlap", "unknown_extra", "wrong_focus", "long_reason"]
)
def test_new_selection_protocol_keeps_strict_shape_and_focus(bad):
    payload, response = payload_and_response()
    claim = response["checks"][0]
    if bad == "empty_primary":
        claim["source_predicate_ids"] = []
    elif bad == "overlap":
        claim["context_source_predicate_ids"] = list(claim["source_predicate_ids"])
    elif bad == "unknown_extra":
        claim["source_mode"] = "permission"
    elif bad == "wrong_focus":
        claim["predicate_id"] = "A:E9:P9"
    else:
        claim["reason"] = "长" * 301
    with pytest.raises((CheckProtocolError, ValidationError)):
        parse_grounded_selection(json.dumps(response), payload)


def test_invalid_legacy_explicit_evidence_is_never_filled_or_repaired():
    payload, response = payload_and_response()
    response["checks"][0]["evidence"] = [{"source_id": "S1", "span_id": "S1:E1"}]
    value = parse_grounded_selection(json.dumps(response), payload)
    assert [r.span_id for r in value.checks[0].evidence] == ["S1:E1"]
    with pytest.raises(CheckProtocolError):
        bind_grounded_relations(value, payload["source_predicates"])


def test_wire_only_removes_duplicate_position_entry_and_preserves_five_flags():
    before = GroundedRelations.model_json_schema()
    frozen = deepcopy(before)
    wire = selection_wire(before)
    fields = wire["$defs"]["GroundedRelation"]["properties"]
    assert "evidence" not in fields and before == frozen
    for field in PREDICATES:
        assert fields[field] == before["$defs"]["GroundedRelation"]["properties"][field]


@pytest.mark.parametrize("failed", [None, *PREDICATES])
async def test_actual_provider_and_complete_scope_pipeline_bind_selection_and_keep_vetoes(failed):
    fixture = ExplicitFocusProvider(failed=failed)
    bodies = []

    async def handle(request):
        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        bodies.append((body, data))
        if "focus_source_span" in data:
            task, schema = "check_source_predicate_parts", SourcePredicateParts
        elif "source_predicates" in data:
            task, schema = "check_grounded_relations", GroundedRelations
        else:
            task, schema = "check_answer_predicate_parts", AnswerRangePredicates
        value = (await fixture.structured(schema, task, data)).model_dump()
        if task == "check_grounded_relations":
            for claim in value["checks"]:
                claim.pop("evidence")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
        assert result.passed is (failed is None)
        if failed:
            assert not audit["checks"][0][failed]
        for body, data in bodies:
            assert "本次输出字段契约" in body["messages"][0]["content"]
            if "source_predicates" in data:
                assert data["output_contract"]["evidence_positions_bound_by_backend"]
    finally:
        await provider.close()
