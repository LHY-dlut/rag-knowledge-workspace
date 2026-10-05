from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agent import RAGAgent, validate_grade_support
from app.providers import DashScopeProvider, DemoProvider
from app.schemas import GradeAnswerBindingDecision, GradeDecision, RetrievalConfig, RetrievalResult
from tests.test_providers_parsers import remote_settings

SOURCE = {
    "source_id": "S1",
    "document_id": "doc-a",
    "parent_id": "p-a",
    "filename": "a.txt",
    "location": "段1",
    "content": "甲馆提供预约借书。",
    "child_ids": ["c-a"],
}


def positive():
    return GradeDecision(
        passed=True,
        support="supported_answer",
        reason="待独立绑定",
        answer_scope="甲馆提供预约借书。",
        evidence=[{"source_id": "S1", "span_id": "S1:E1", "subject": "甲馆", "attribute": "服务"}],
    )


def binding(**changes):
    return GradeAnswerBindingDecision(
        subject_attribute_pairs_supported=True,
        question_requirements_covered=True,
        conditions_preserved=True,
        scope_supported=True,
        reason="核对实际问句与原文",
        **changes,
    )


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "subject_attribute_pairs_supported",
        "question_requirements_covered",
        "conditions_preserved",
        "scope_supported",
    ],
)
async def test_every_semantic_predicate_controls_final_supported_grade(missing):
    values = {
        name: name != missing
        for name in (
            "subject_attribute_pairs_supported",
            "question_requirements_covered",
            "conditions_preserved",
            "scope_supported",
        )
    }

    class Provider:
        supports_answer_binding = True

        async def structured(self, schema, task, payload):
            assert schema is GradeAnswerBindingDecision and task == "grade_answer_binding"
            assert "reason" not in payload and "passed" not in payload
            assert payload["evidence"][0]["quote"] == SOURCE["content"]
            assert payload["query"] == "甲馆有哪些服务？"
            return GradeAnswerBindingDecision(**values, reason="命题绑定及必要子问覆盖")

    result, checked = await validate_grade_support(
        Provider(), positive(), [SOURCE], "甲馆有哪些服务？"
    )
    assert result.passed == (missing is None) and checked.passed == (missing is None)
    if missing is not None:
        assert result.support == "insufficient" and not result.evidence
    assert RAGAgent._summary({"grade_binding": checked})["grade_binding"][
        missing or "scope_supported"
    ] == (missing is None)


@pytest.mark.parametrize("bad", ["true", 1, None])
def test_predicates_do_not_coerce_non_boolean_values_or_allow_passed_override(bad):
    values = binding().model_dump(exclude={"passed"})
    with pytest.raises(ValidationError):
        GradeAnswerBindingDecision.model_validate({**values, "question_requirements_covered": bad})
    with pytest.raises(ValidationError):
        GradeAnswerBindingDecision.model_validate({**values, "passed": True})


async def test_unknown_evidence_fails_before_any_semantic_model_call():
    class Provider:
        supports_answer_binding = True

        async def structured(self, *args):
            pytest.fail("Invalid source ID cannot reach a semantic binding call")

    decision = positive()
    decision.evidence[0].source_id = "S999"
    result, checked = await validate_grade_support(
        Provider(), decision, [SOURCE], "甲馆有哪些服务？"
    )
    assert not result.passed and checked is None


async def test_missing_necessary_subquestion_keeps_existing_retry_then_fallback_limit():
    class Provider:
        supports_answer_binding = True

        async def structured(self, schema, task, payload):
            if task == "grade":
                return positive()
            assert task == "grade_answer_binding"
            return GradeAnswerBindingDecision(
                subject_attribute_pairs_supported=True,
                question_requirements_covered=False,
                conditions_preserved=True,
                scope_supported=True,
                reason="问题要求两方，原文只支持一方",
            )

    agent = RAGAgent(Provider(), None)
    state = {
        "query": "甲馆和乙馆各有哪些服务？",
        "retrieval": RetrievalResult(sources=[SOURCE]),
        "mode": "agent",
        "kb": SimpleNamespace(config=RetrievalConfig().model_dump()),
        "grade_retries": 2,
    }
    first = await agent.grade(state)
    assert not first["grade"].passed and first["grade_retries"] == 3
    assert agent._after_grade({**state, **first}) == "rewrite"
    last = await agent.grade({**state, **first})
    assert not last["grade_retry_allowed"] and agent._after_grade({**state, **last}) == "fallback"


async def test_real_provider_always_advertises_binding_and_demo_remains_labelled_simulation():
    real = DashScopeProvider(remote_settings())
    try:
        assert real.supports_answer_binding is True
        assert DemoProvider(remote_settings()).supports_answer_binding is False
    finally:
        await real.close()
