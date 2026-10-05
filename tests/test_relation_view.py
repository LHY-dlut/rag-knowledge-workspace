"""Reversible ID transport must preserve all strict evidence and veto guards."""

import copy
import json

import httpx
import pytest

from app.grounded_relations import GroundedRelations, bind_grounded_relations
from app.grounded_selection import selection_wire
from app.providers import CheckProtocolError, DashScopeProvider
from app.relation_view import parse_relation_view, relation_wire_view
from app.settings import Settings
from tests.test_source_windows import bound, selection, selection_payload


def inputs():
    registry = bound()
    payload = selection_payload(registry)
    focus = {
        "predicate_id": "P1",
        "answer_parts": [
            {"fragment_id": "A1:F1", "source_start": 0, "source_end": 3, "quote": "预计。"}
        ],
    }
    payload.update(
        query="问题预设不能替代实际答案事实",
        grade={"passed": True},
        score=1,
        answer_context_spans=[
            {"span_id": "A1", "source_start": 0, "source_end": 3, "quote": "预计。"}
        ],
        focus_answer_predicate=focus,
    )
    document = selection_wire(GroundedRelations.model_json_schema())
    return registry, payload, document


def test_wire_preserves_raw_context_positions_and_does_not_expose_prior_decisions():
    registry, payload, document = inputs()
    before = copy.deepcopy((registry, payload, document))
    view, schema, aliases = relation_wire_view(payload, document)
    assert (registry, payload, document) == before
    assert not {"query", "grade", "score", "focus_answer_reading_span"} & view.keys()
    assert view["answer_context_spans"] == payload["answer_context_spans"]
    assert view["focus_answer_predicate"] == payload["focus_answer_predicate"]
    assert list(view)[-1] == "focus_answer_predicate"
    assert view["sources"] == payload["sources"]
    for actual, original in zip(view["source_predicates"], registry, strict=True):
        assert aliases[actual["source_predicate_id"]] == original["source_predicate_id"]
        assert actual["source_parts"] == original["source_parts"]
        assert not {"mode", "voice", "reason"} & actual.keys()
    definition = schema["$defs"]["GroundedRelation"]["properties"]
    assert definition["source_predicate_ids"]["items"]["enum"] == list(aliases)
    assert definition["context_source_predicate_ids"]["items"]["enum"] == list(aliases)


@pytest.mark.parametrize(
    "mutation", [None, "unknown", "boolean", "empty", "duplicate", "overlap", "explicit_evidence"]
)
def test_translation_does_not_repair_invalid_selection_or_a_changed_semantic_flag(mutation):
    registry, payload, document = inputs()
    view, _, aliases = relation_wire_view(payload, document)
    value = selection(view["source_predicates"])
    check = value["checks"][0]
    check["temporal_scope_preserved"] = "changed"
    alias = check["source_predicate_ids"][0]
    if mutation == "unknown":
        check["source_predicate_ids"] = ["R-unknown"]
    if mutation == "boolean":
        check["source_predicate_ids"] = [True]
    if mutation == "empty":
        check["source_predicate_ids"] = []
    if mutation == "duplicate":
        check["source_predicate_ids"] = [alias, alias]
    if mutation == "overlap":
        check["context_source_predicate_ids"] = [alias]
    if mutation == "explicit_evidence":
        check["evidence"] = [{"source_id": "S99", "span_id": "S99:E1"}]
    if mutation is not None:
        with pytest.raises((ValueError, CheckProtocolError)):
            parse_relation_view(json.dumps(value), aliases, payload)
        return
    parsed = parse_relation_view(json.dumps(value), aliases, payload)
    relation = bind_grounded_relations(parsed, registry).checks[0]
    assert relation.source_predicate_ids == [registry[0]["source_predicate_id"]]
    assert relation.temporal_scope_preserved == "changed"
    assert relation.temporal_scope_preserved == check["temporal_scope_preserved"]
    assert relation.source_mode == "future" and relation.source_voice == "opinion"


@pytest.mark.asyncio
async def test_actual_provider_uses_alias_contract_then_original_literal_registry():
    registry, payload, _ = inputs()
    payload["source_predicates"] = [
        {k: v for k, v in p.items() if k not in {"mode", "voice", "reason", "literal_binding"}}
        for p in registry
    ]
    captured = []

    def respond(request):
        body = json.loads(request.content)
        wire = json.loads(body["messages"][1]["content"])
        captured.append(wire)
        assert wire["output_contract"]["allowed_source_predicate_ids"] == [
            p["source_predicate_id"] for p in wire["source_predicates"]
        ]
        assert "query" not in wire and "grade" not in wire
        value = selection(wire["source_predicates"])
        value["checks"][0]["conditions_preserved"] = "changed"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})

    cfg = Settings(
        _env_file=None,
        check_source_window_projection=True,
        dashscope_chat_base_url="https://example.test/v1",
    )
    provider = DashScopeProvider(cfg, httpx.MockTransport(respond))
    try:
        result = await provider.structured(GroundedRelations, "check_grounded_relations", payload)
    finally:
        await provider.close()
    assert len(captured) == 1
    assert result.checks[0].conditions_preserved == "changed"
    assert result.checks[0].source_predicate_ids == [registry[0]["source_predicate_id"]]
    assert [(p.source_id, p.span_id) for p in result.checks[0].evidence] == [
        ("S1", "S1:E1"),
        ("S1", "S1:E2"),
    ]
