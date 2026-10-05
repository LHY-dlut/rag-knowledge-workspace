from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.check_scope import PREDICATES, validate_check_scope
from app.grounded_relations import GroundedRelations, bind_grounded_relations
from app.providers import CheckProtocolError
from app.source_parts import validate_source_pairings
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_grounded_relations import GroundedProvider


def registry():
    return [
        {
            "source_predicate_id": "permit",
            "source_id": "S1",
            "span_id": "S1:E1",
            "source_parts": [
                {"source_start": 8, "source_end": 22, "quote": "试行期间符合条件者可以借用。"}
            ],
            "mode": "permission",
            "voice": "fact",
        },
        {
            "source_predicate_id": "repeal",
            "source_id": "S1",
            "span_id": "S1:E2",
            "source_parts": [
                {"source_start": 22, "source_end": 32, "quote": "该规则已于次年废止。"}
            ],
            "mode": "asserted",
            "voice": "fact",
        },
    ]


def decision(**changes):
    claim = {
        "predicate_id": "A:E1:P1",
        "source_predicate_ids": ["permit"],
        "context_source_predicate_ids": ["repeal"],
        "evidence": [
            {"source_id": "S1", "span_id": "S1:E1"},
            {"source_id": "S1", "span_id": "S1:E2"},
        ],
        "reason": "历史许可及其废止范围均须保留。",
        **{field: "preserved" for field in PREDICATES},
        **changes,
    }
    return GroundedRelations.model_validate({"answer_span_id": "A:E1", "checks": [claim]})


def test_different_context_category_never_reclassifies_primary_fact():
    raw = registry()
    frozen = deepcopy(raw)
    bound = bind_grounded_relations(decision(), raw)
    claim = bound.checks[0]
    assert claim.source_mode == "permission" and claim.source_voice == "fact"
    audits = validate_source_pairings(bound.checks, raw)
    assert audits[0]["positioned_source_predicates"] == [raw[0]]
    assert audits[0]["positioned_scope_context_predicates"] == [raw[1]]
    assert len(claim.evidence) == 2 and raw == frozen


@pytest.mark.parametrize("field", PREDICATES)
def test_context_does_not_suppress_any_semantic_veto(field):
    bound = bind_grounded_relations(decision(**{field: "changed"}), registry())
    validate_source_pairings(bound.checks, registry())
    assert getattr(bound.checks[0], field) == "changed"


@pytest.mark.parametrize(
    "changes",
    [
        {"source_predicate_ids": []},
        {"context_source_predicate_ids": ["permit"]},
        {"context_source_predicate_ids": ["repeal", "repeal"]},
    ],
)
def test_context_cannot_replace_primary_fact_or_overlap_it(changes):
    with pytest.raises(ValidationError):
        decision(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"context_source_predicate_ids": ["unknown"]},
        {"evidence": [{"source_id": "S1", "span_id": "S1:E1"}]},
        {"context_source_predicate_ids": []},
        {
            "evidence": [
                {"source_id": "S1", "span_id": "S1:E1"},
                {"source_id": "S9", "span_id": "S9:E2"},
            ]
        },
    ],
)
def test_unknown_unbound_or_omitted_context_position_fails_closed(changes):
    with pytest.raises(CheckProtocolError):
        bind_grounded_relations(decision(**changes), registry())


def test_duplicate_registry_cannot_overwrite_source():
    parts = registry()
    with pytest.raises(CheckProtocolError):
        bind_grounded_relations(decision(), parts + [{**parts[0], "mode": "asserted"}])


def test_mixed_primary_categories_remain_undetermined():
    claim = decision(source_predicate_ids=["permit", "repeal"], context_source_predicate_ids=[])
    bound = bind_grounded_relations(claim, registry())
    assert bound.checks[0].source_mode == "undetermined"
    validate_source_pairings(bound.checks, registry())
    assert bound.checks[0].modality_preserved == "changed"


class ContextProvider(GroundedProvider):
    async def structured(self, schema, task, payload):
        value = await super().structured(schema, task, payload)
        if task == "check_grounded_relations":
            primary = set(value.checks[0].source_predicate_ids)
            value.checks[0].context_source_predicate_ids = [
                p["source_predicate_id"]
                for p in payload["source_predicates"]
                if p["source_predicate_id"] not in primary
            ]
        return value


async def test_scope_pipeline_records_both_roles_without_waiving_category_match():
    result, audit = await validate_check_scope(
        ContextProvider(), original(), GOOD, [SOURCE], "问题"
    )
    assert result.passed
    checks = audit["checks"][0]
    assert [p["source_mode"] for p in checks["predicate_checks"]] == ["planned", "asserted"]
    assert all(p["positioned_scope_context_predicates"] for p in checks["source_pairing_checks"])


@pytest.mark.parametrize("failed", PREDICATES)
async def test_scope_pipeline_context_cannot_waive_negative_relation(failed):
    result, audit = await validate_check_scope(
        ContextProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and not audit["checks"][0][failed]


@pytest.mark.parametrize("field", ["mode", "voice"])
async def test_scope_context_cannot_supply_a_convenient_primary_classification(field):
    provider = ContextProvider(source_mode_swap=field == "mode", source_voice_swap=field == "voice")
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
    assert not result.passed
    flag = "mode_matches_this_predicate" if field == "mode" else "voice_matches_this_predicate"
    assert not audit["checks"][0]["predicate_checks"][0][flag]
