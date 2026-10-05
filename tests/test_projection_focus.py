import json

import httpx
import pytest

from app.check_protocol import answer_spans
from app.check_scope import PREDICATES, validate_check_scope
from app.grounded_relations import GroundedRelations
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import AnswerRangePredicates
from app.source_parts import SourcePredicateParts
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_explicit_focus import ExplicitFocusProvider
from tests.test_providers_parsers import remote_settings


class ProjectionFocusProvider(ExplicitFocusProvider):
    supports_check_explicit_projection_focus = True

    async def structured(self, schema, task, payload):
        if task == "check_answer_predicate_parts":
            assert list(payload)[0] == "focus_answer_span"
            assert payload["focus_answer_span"] == payload["answer_spans"][0]
            assert payload["answer"] == GOOD  # Internal exact coverage only.
            assert not {"query", "sources", "previous_judgment"} & payload.keys()
        return await super().structured(schema, task, payload)


async def test_projection_focus_is_explicit_without_sources_or_prior_conclusions():
    provider = ProjectionFocusProvider()
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
    assert result.passed and len(provider.calls) == audit["scope_model_requests"] == 4


@pytest.mark.parametrize("failed", PREDICATES)
async def test_projection_focus_retains_all_five_semantic_vetoes(failed):
    result, audit = await validate_check_scope(
        ProjectionFocusProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and not audit["checks"][0][failed]


async def test_real_wire_has_focus_and_context_once_with_current_ID_coverage():
    fixture = ExplicitFocusProvider()
    captured = []

    async def handle(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        captured.append((body, payload))
        if "focus_source_span" in payload:
            task, schema = "check_source_predicate_parts", SourcePredicateParts
        elif "source_predicates" in payload:
            task, schema = "check_grounded_relations", GroundedRelations
        else:
            task, schema = "check_answer_predicate_parts", AnswerRangePredicates
        value = await fixture.structured(schema, task, payload)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": value.model_dump_json()}}]}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
        assert result.passed and len(captured) == audit["scope_model_requests"] == 4
        body, data = captured[0]
        assert list(data)[0] == "focus_answer_span" and "answer" not in data
        assert data["focus_answer_span"] == answer_spans(GOOD)[0]
        assert data["answer_context_spans"] == [
            {k: v for k, v in span.items() if k != "span_id"} for span in answer_spans(GOOD)
        ]
        focus = data["focus_answer_span"]
        assert GOOD[focus["source_start"] : focus["source_end"]] == focus["quote"]
        assert "[S1]" not in focus["quote"]
        for part in data["answer_fragments"]:
            assert GOOD[part["source_start"] : part["source_end"]] == part["quote"]
        instruction = body["messages"][0]["content"]
        assert "仅在其他上下文片段出现的独立事实不得产生当前谓词" in instruction
        assert "不得用当前fragment代替其他片段的谓词位置" in instruction
        assert '"enum": ["A:E1:F1", "A:E1:F2"]' in instruction
        assert '"const": "A:E1:F1"' in instruction and '"const": "A:E1:F2"' in instruction
    finally:
        await provider.close()


async def test_foreign_fragment_remains_protocol_failure_before_any_source_review():
    class Foreign(ProjectionFocusProvider):
        async def structured(self, schema, task, payload):
            value = await super().structured(schema, task, payload)
            if task == "check_answer_predicate_parts":
                value.predicates[0].fragment_ids = ["A:E2:F1"]
            return value

    provider = Foreign()
    with pytest.raises(CheckProtocolError, match="未知或其他片段ID"):
        await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
    assert [task for task, _ in provider.calls] == ["check_answer_predicate_parts"]


async def test_projection_cannot_repair_independent_source_category_mismatch():
    result, audit = await validate_check_scope(
        ProjectionFocusProvider(source_mode_swap=True), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed
    assert not audit["checks"][0]["predicate_checks"][0]["mode_matches_this_predicate"]
