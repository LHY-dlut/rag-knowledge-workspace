import json

import httpx
import pytest
from pydantic import ValidationError

from app.check_protocol import bind_check
from app.check_scope import PREDICATES, validate_check_scope
from app.grounded_relations import GroundedRelations
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import AnswerRangePredicates, CheckDecision
from app.source_parts import SourcePredicateParts
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_providers_parsers import remote_settings
from tests.test_source_pairs import PairedProvider


class GroundedProvider(PairedProvider):
    supports_check_grounded_relations = True

    def __init__(self, *args, source_mode_swap=False, source_voice_swap=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_mode_swap = source_mode_swap
        self.source_voice_swap = source_voice_swap

    async def structured(self, schema, task, payload):
        if task != "check_grounded_relations":
            value = await super().structured(schema, task, payload)
            if task == "check_source_predicate_parts":
                if self.source_mode_swap:
                    value.predicates[0].mode = "asserted"
                if self.source_voice_swap:
                    value.predicates[0].voice = "opinion"
            return value
        self.calls.append((task, payload))
        assert schema is GroundedRelations
        assert all(not {"mode", "voice", "reason"} & p.keys() for p in payload["source_predicates"])
        pid = payload["focus_predicate_id"]
        selected = payload["source_predicates"][int(pid[1:]) - 1]
        row = {
            "predicate_id": pid,
            "source_predicate_ids": [selected["source_predicate_id"]],
            "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
            "reason": "当前事实与原文对应",
            **{n: "changed" if n == self.failed else "preserved" for n in PREDICATES},
        }
        if self.invalid == "mixed_source_modes":
            row["source_predicate_ids"] = [
                p["source_predicate_id"] for p in payload["source_predicates"]
            ]
        if self.invalid == "foreign_pair":
            row["source_predicate_ids"] = ["S9:E77:P0"]
        if self.invalid == "foreign_evidence":
            row["evidence"] = [{"source_id": "S9", "span_id": "S9:E1"}]
        if self.invalid == "foreign_predicate":
            row["predicate_id"] = "P999"
        if self.invalid == "category_field":
            row["source_mode"] = "asserted"
        if self.invalid == "timeout_relation":
            raise TimeoutError("受控关系核验超时")
        return schema.model_validate({"answer_span_id": "A:E1", "checks": [row]})


async def test_source_categories_are_only_independent_and_never_requested_again():
    provider = GroundedProvider()
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "私有问题")
    assert result.passed and len(provider.calls) == audit["scope_model_requests"] == 4
    claim = audit["checks"][0]
    assert claim["source_category_origin"] == "independent_literal_source_predicates_only"
    assert [p["source_mode"] for p in claim["predicate_checks"]] == ["planned", "asserted"]
    assert all("mode" not in p for _, data in provider.calls[2:] for p in data["answer_predicates"])
    assert all("mode" not in p for _, data in provider.calls[2:] for p in data["source_predicates"])


@pytest.mark.parametrize("failed", PREDICATES)
async def test_every_negative_relation_remains_a_veto(failed):
    result, audit = await validate_check_scope(
        GroundedProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and not audit["checks"][0][failed]


@pytest.mark.parametrize("field", ["mode", "voice"])
async def test_known_but_different_independent_categories_still_reject(field):
    provider = GroundedProvider(
        source_mode_swap=field == "mode", source_voice_swap=field == "voice"
    )
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
    assert not result.passed
    flag = "mode_matches_this_predicate" if field == "mode" else "voice_matches_this_predicate"
    assert not audit["checks"][0]["predicate_checks"][0][flag]


@pytest.mark.parametrize(
    "invalid", ["mixed_source_modes", "undetermined_mode", "undetermined_voice"]
)
async def test_mixed_or_unknown_source_categories_cannot_select_a_convenient_answer_mode(invalid):
    result, audit = await validate_check_scope(
        GroundedProvider(invalid=invalid), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and not audit["passed"]
    if invalid == "mixed_source_modes":
        assert all(
            p["source_mode"] == "undetermined" for p in audit["checks"][0]["predicate_checks"]
        )


@pytest.mark.parametrize(
    "invalid",
    ["foreign_pair", "foreign_evidence", "foreign_predicate", "category_field", "timeout_relation"],
)
async def test_invalid_ids_category_override_or_timeout_never_publish_success(invalid):
    with pytest.raises((CheckProtocolError, ValidationError, TimeoutError)):
        await validate_check_scope(
            GroundedProvider(invalid=invalid), original(), GOOD, [SOURCE], "问题"
        )


async def test_uncited_source_cannot_enter_independent_projection_or_support_pool():
    provider = GroundedProvider()
    unused = {"source_id": "S2", "content": "另一个主体的资料。"}
    result, _ = await validate_check_scope(provider, original(), GOOD, [SOURCE, unused], "问题")
    assert result.passed
    for task, data in provider.calls:
        if task == "check_source_predicate_parts":
            assert data["source_id"] == "S1"
        if task == "check_grounded_relations":
            assert {p["source_id"] for p in data["source_predicates"]} == {"S1"}


async def test_duplicate_source_ids_with_different_text_cannot_overwrite_evidence():
    with pytest.raises(CheckProtocolError):
        await validate_check_scope(
            GroundedProvider(),
            original(),
            GOOD,
            [SOURCE, {**SOURCE, "content": "其他文档。"}],
            "问题",
        )


async def test_rejected_upstream_never_dispatches_any_independent_or_relation_calls():
    provider = GroundedProvider()
    judgment = original().model_copy(update={"passed": False})
    result, audit = await validate_check_scope(provider, judgment, GOOD, [SOURCE], "问题")
    assert result is judgment and audit is None and provider.calls == []


class LaterSpanProvider:
    supports_check_scope_binding = True
    supports_check_atomic_scope = True
    supports_check_predicate_parts = True
    supports_check_single_predicate_scope = True
    supports_check_source_predicate_parts = True
    supports_check_grounded_relations = True
    supports_check_context_without_foreign_ids = True

    def __init__(self):
        self.calls = []

    async def structured(self, schema, task, data):
        self.calls.append((task, data))
        if task == "check_answer_predicate_parts":
            return schema.model_validate(
                {
                    "answer_span_id": "A:E1",
                    "predicates": [
                        {
                            "predicate_id": f"P{i + 1}",
                            "fragment_ids": [p["fragment_id"]],
                            "mode": "asserted",
                            "voice": "fact",
                            "reason": "真实答案事实",
                        }
                        for i, p in enumerate(data["answer_fragments"])
                    ],
                }
            )
        if task == "check_source_predicate_parts":
            assert not {"answer", "query"} & data.keys()
            return schema.model_validate(
                {
                    "source_id": "S1",
                    "source_span_id": data["focus_source_span"]["span_id"],
                    "predicates": [
                        {
                            "predicate_id": "R1",
                            "fragment_ids": [p["fragment_id"] for p in data["source_fragments"]],
                            "mode": "asserted",
                            "voice": "fact",
                            "reason": "独立原文事实",
                        }
                    ],
                }
            )
        assert task == "check_grounded_relations"
        span_id = "S1:E1" if data["focus_predicate_id"] == "P1" else "S1:E2"
        selected = next(p for p in data["source_predicates"] if p["span_id"] == span_id)
        return schema.model_validate(
            {
                "answer_span_id": "A:E1",
                "checks": [
                    {
                        "predicate_id": data["focus_predicate_id"],
                        "source_predicate_ids": [selected["source_predicate_id"]],
                        "evidence": [{"source_id": "S1", "span_id": span_id}],
                        "reason": "对应实际主体，未借其他主体事实",
                        **{n: "preserved" for n in PREDICATES},
                    }
                ],
            }
        )


async def test_later_real_source_span_is_available_even_if_basic_check_selected_only_first():
    source = {"source_id": "S1", "content": "设备甲已登记。设备乙已归还。"}
    answer = "设备甲已登记，设备乙已归还。[S1]"
    before = bind_check(
        CheckDecision(
            checks=[
                {
                    "answer_span_id": "A:E1",
                    "verdict": "supported",
                    "reason": "基础check只选首句，额外核验必须可见后句",
                    "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                }
            ]
        ),
        answer,
        [source],
    )
    provider = LaterSpanProvider()
    result, audit = await validate_check_scope(provider, before, answer, [source], "设备记录？")
    assert result.passed and audit["scope_model_requests"] == len(provider.calls) == 5
    assert [
        d["focus_source_span"]["span_id"]
        for t, d in provider.calls
        if t == "check_source_predicate_parts"
    ] == ["S1:E1", "S1:E2"]
    second = audit["checks"][0]["predicate_checks"][1]
    assert second["literal_evidence"][0]["source_start"] == len("设备甲已登记。")
    assert second["literal_evidence"][0]["original_quote"] == "设备乙已归还。"
    assert second["source_predicate_ids"] == ["S1:S1:E2:R1"]


def test_duplicate_relations_refs_and_extra_category_fields_are_strictly_invalid():
    row = {
        "predicate_id": "P1",
        "source_predicate_ids": ["S1:S1:E1:P1"],
        "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
        "reason": "原文关系",
        **{n: "preserved" for n in PREDICATES},
    }
    for checks in (
        [row, row],
        [{**row, "source_mode": "asserted"}],
        [{**row, "source_predicate_ids": row["source_predicate_ids"] * 2}],
        [{**row, "evidence": row["evidence"] * 2}],
    ):
        with pytest.raises(ValidationError):
            GroundedRelations.model_validate({"answer_span_id": "A:E1", "checks": checks})


async def test_actual_provider_relation_only_schema_and_cache_do_not_leak_categories():
    fixture = GroundedProvider()
    bodies = []

    async def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        data = json.loads(body["messages"][1]["content"])
        system = body["messages"][0]["content"].split("JSON 对象：", 1)[1]
        wire, _ = json.JSONDecoder().raw_decode(system)
        if "focus_source_span" in data:
            value = await fixture.structured(
                SourcePredicateParts, "check_source_predicate_parts", data
            )
        elif "source_predicates" in data:
            fields = wire["$defs"]["GroundedRelation"]["properties"]
            assert "source_mode" not in fields and "source_voice" not in fields
            assert fields["predicate_id"]["enum"] == [data["focus_predicate_id"]]
            assert fields["source_predicate_ids"]["items"]["enum"] == [
                p["source_predicate_id"] for p in data["source_predicates"]
            ]
            value = await fixture.structured(GroundedRelations, "check_grounded_relations", data)
        else:
            value = await fixture.structured(
                AnswerRangePredicates, "check_answer_predicate_parts", data
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": value.model_dump_json()}}]}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
        assert result.passed and audit["scope_model_requests"] == len(bodies) == 4
        assert all(body["response_format"] == {"type": "json_object"} for body in bodies)
    finally:
        await provider.close()
