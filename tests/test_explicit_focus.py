import json

import httpx
import pytest

from app.check_protocol import answer_spans
from app.check_scope import PREDICATES, validate_check_scope
from app.grounded_relations import GroundedRelations
from app.providers import DashScopeProvider
from app.schemas import AnswerRangePredicates
from app.source_parts import SourcePredicateParts
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_grounded_relations import GroundedProvider
from tests.test_providers_parsers import remote_settings


class ExplicitFocusProvider(GroundedProvider):
    supports_check_explicit_predicate_focus = True

    async def structured(self, schema, task, payload):
        if task == "check_grounded_relations":
            assert list(payload)[0] == "focus_answer_predicate"
            assert "answer" not in payload and "answer_predicates" not in payload
            assert (
                payload["focus_answer_predicate"]["predicate_id"] == payload["focus_predicate_id"]
            )
            assert not {"mode", "voice", "reason"} & payload["focus_answer_predicate"].keys()
            assert payload["answer_context_spans"] == answer_spans(GOOD)
        return await super().structured(schema, task, payload)


async def test_explicit_current_parts_are_first_and_all_original_context_is_retained():
    provider = ExplicitFocusProvider()
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
    assert result.passed and len(provider.calls) == audit["scope_model_requests"] == 4
    source_calls = [data for task, data in provider.calls if task == "check_grounded_relations"]
    assert len(source_calls) == 2
    assert [data["focus_predicate_id"] for data in source_calls] == ["P1", "P2"]
    assert (
        source_calls[0]["focus_answer_predicate"]["answer_parts"][0]["quote"] == "培训拟下周进行，"
    )
    assert source_calls[1]["focus_answer_predicate"]["answer_parts"][0]["quote"] == "值班已开始。"
    for data in source_calls:
        for part in data["focus_answer_predicate"]["answer_parts"]:
            assert GOOD[part["source_start"] : part["source_end"]] == part["quote"]
        assert data["sources"][0]["content"] == SOURCE["content"]
        assert data["actual_cited_source_ids"] == ["S1"]


@pytest.mark.parametrize("failed", PREDICATES)
async def test_focus_does_not_upgrade_any_negative_relationship(failed):
    result, audit = await validate_check_scope(
        ExplicitFocusProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and not audit["checks"][0][failed]


async def test_actual_provider_wire_preserves_independent_taxonomy_and_current_focus():
    fixture = ExplicitFocusProvider()
    bodies = []

    async def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        data = json.loads(body["messages"][1]["content"])
        if "focus_source_span" in data:
            task, schema = "check_source_predicate_parts", SourcePredicateParts
        elif "source_predicates" in data:
            task, schema = "check_grounded_relations", GroundedRelations
        else:
            task, schema = "check_answer_predicate_parts", AnswerRangePredicates
        value = await fixture.structured(schema, task, data)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": value.model_dump_json()}}]}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
        assert result.passed and len(bodies) == audit["scope_model_requests"] == 4
        independent = [
            b["messages"][0]["content"]
            for b in bodies
            if "source_predicates" not in json.loads(b["messages"][1]["content"])
        ]
        assert len(independent) == 2
        for instruction in independent:
            assert "专门类别取决于实际谓词关系，不取决于宾语的主题领域" in instruction
            assert "仍分类被转述的内容性关系" in instruction
        assert all(b["response_format"] == {"type": "json_object"} for b in bodies)
    finally:
        await provider.close()


async def test_independent_category_mismatch_is_not_repaired_by_better_focus():
    result, audit = await validate_check_scope(
        ExplicitFocusProvider(source_mode_swap=True), original(), GOOD, [SOURCE], "问题"
    )
    assert (
        not result.passed
        and not audit["checks"][0]["predicate_checks"][0]["mode_matches_this_predicate"]
    )
