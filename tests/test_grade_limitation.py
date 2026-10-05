import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from app.agent import RAGAgent, bind_grade_evidence, validate_grade_support
from app.providers import DashScopeProvider
from app.schemas import (
    GradeBindingDecision,
    GradeDecision,
    Judgment,
    RetrievalConfig,
    RetrievalResult,
)
from tests.test_providers_parsers import remote_settings


async def test_generation_does_not_treat_freeform_gate_explanation_as_answer_evidence(
    monkeypatch,
):
    import app.agent as module

    quote = "实验设备由保障组负责。资料未规定维修完成期限。"
    grade = GradeDecision(
        passed=True,
        support="supported_answer",
        reason="未核对的解释：另有审批材料清单",
        answer_scope="新增要求：须提供领导签字证明，否则无法受理",
        evidence=[
            {
                "source_id": "S1",
                "quote": quote,
                "subject": "实验设备",
                "attribute": "维修",
            }
        ],
    )
    calls, events = [], []

    class Provider:
        async def stream_answer(self, query, sources, feedback, prompt):
            calls.append((query, sources, json.loads(feedback), prompt))
            yield "根据资料，维修由保障组负责[S1]。"

    async def no_template(**kwargs):
        return None

    monkeypatch.setattr(module.PromptTemplate, "get_or_none", no_template)
    monkeypatch.setattr(module, "get_stream_writer", lambda: events.append)
    state = {
        "owner_id": "isolated-development-owner",
        "query": "维修由谁负责？",
        "grade": grade,
        "retrieval": SimpleNamespace(
            sources=[SimpleNamespace(model_dump=lambda: {"source_id": "S1", "content": quote})]
        ),
        "feedback": "之前草稿的事实需要对应原文引用",
    }
    result = await RAGAgent(Provider(), None).generate(state)
    _, sources, feedback, _ = calls[0]
    assert sources == [{"source_id": "S1", "content": quote}]
    assessment = feedback["evidence_assessment"]
    assert assessment["support"] == "supported_answer"
    assert assessment["evidence"][0]["quote"] == quote
    assert "领导签字" not in json.dumps(assessment, ensure_ascii=False)
    assert "审批材料清单" not in json.dumps(assessment, ensure_ascii=False)
    assert feedback["revision_feedback"] == state["feedback"]
    assert result["draft"] == "根据资料，维修由保障组负责[S1]。"
    assert events[0]["event"] == "draft_start"
    assert events[1]["data"]["verified"] is False


def supported(quote="该管理办法未规定培训申请的办理期限。", source_id="S1"):
    return GradeDecision(
        passed=True,
        support="supported_limitation",
        reason="同一主体与所问期限属性有明确否定证据",
        answer_scope="仅根据所提供管理办法，不能认定培训申请必须三天办结。",
        evidence=[
            {
                "source_id": source_id,
                "quote": quote,
                "subject": "培训申请",
                "attribute": "办理期限",
            }
        ],
    )


def test_limitation_requires_evidence_scope_and_consistent_classification():
    assert supported().passed
    for change in [
        {"evidence": []},
        {"answer_scope": " "},
        {"passed": False},
        {"support": "insufficient"},
    ]:
        with pytest.raises(ValidationError):
            GradeDecision.model_validate({**supported().model_dump(), **change})


@pytest.mark.parametrize(
    "source_id,quote",
    [
        ("S999", "该管理办法未规定培训申请的办理期限。"),
        ("S1", "现实中不存在任何办理期限。"),
        ("S1", " "),
    ],
)
def test_invalid_evidence_binding_fails_closed(source_id, quote):
    sources = [{"source_id": "S1", "content": "该管理办法未规定培训申请的办理期限。"}]
    decision = bind_grade_evidence(supported(quote, source_id), sources)
    assert not decision.passed and decision.support == "insufficient"


def test_exact_bound_evidence_retains_the_limitation_scope():
    original = supported()
    bound = bind_grade_evidence(
        original, [{"source_id": "S1", "content": original.evidence[0].quote}]
    )
    assert bound == original


@pytest.mark.parametrize("semantic_match", [True, False])
async def test_limitation_semantic_binding_controls_final_grade(semantic_match):
    decision = supported()
    source = [{"source_id": "S1", "content": decision.evidence[0].quote}]

    class Provider:
        async def structured(self, schema, task, payload):
            assert schema is GradeBindingDecision and task == "grade_binding"
            assert payload["evidence"] == [item.model_dump() for item in decision.evidence]
            assert "reason" not in payload
            return GradeBindingDecision(
                subject_matches=True,
                attribute_matches=semantic_match,
                explicit_limitation=True,
                scope_supported=semantic_match,
                reason="主体属性匹配" if semantic_match else "属性不同，不能迁移否定",
            )

    final, binding = await validate_grade_support(Provider(), decision, source, "培训办理期限？")
    assert final.passed == semantic_match and binding.passed == semantic_match
    assert final.support == ("supported_limitation" if semantic_match else "insufficient")


def test_binding_cannot_override_a_missing_attribute_with_passed_true():
    values = dict(
        subject_matches=True,
        attribute_matches=False,
        explicit_limitation=False,
        scope_supported=False,
        reason="上游声称passed=true但属性不匹配",
    )
    assert not GradeBindingDecision(**values).passed
    with pytest.raises(ValidationError):
        GradeBindingDecision.model_validate({**values, "passed": True})


async def test_empty_context_never_calls_model_and_keeps_retry_limit():
    class ForbiddenProvider:
        async def structured(self, *args):
            pytest.fail("Empty evidence must not call grade")

    agent = RAGAgent(ForbiddenProvider(), None)
    state = {
        "query": "培训办理期限？",
        "retrieval": RetrievalResult(),
        "mode": "agent",
        "kb": SimpleNamespace(config=RetrievalConfig().model_dump()),
        "grade_retries": 3,
    }
    update = await agent.grade(state)
    assert update["grade"].support == "insufficient"
    assert not update["grade_retry_allowed"]
    assert agent._after_grade({**state, **update}) == "fallback"


async def test_supported_limitation_reaches_generation_with_scope(monkeypatch):
    import app.agent as module

    received = {}
    events = []

    class Provider:
        async def stream_answer(self, query, sources, feedback, prompt):
            received.update(json.loads(feedback))
            yield "根据所提供管理办法，不能认定培训申请必须三天办结[S1]。"

    async def no_template(**kwargs):
        return None

    monkeypatch.setattr(module, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(module.PromptTemplate, "get_or_none", no_template)
    agent = RAGAgent(Provider(), None)
    grade = supported()
    state = {
        "query": "培训申请必须三天内办结吗？",
        "owner_id": "test",
        "grade": grade,
        "retrieval": RetrievalResult(),
        "feedback": "上次核验要求收紧范围",
    }
    result = await agent.generate(state)
    assert received["evidence_assessment"] == grade.model_dump()
    assert received["revision_feedback"] == state["feedback"]
    assert "[S1]" in result["draft"] and events[0]["event"] == "draft_start"


async def test_check_rejection_never_publishes_expanded_claim(monkeypatch):
    import app.agent as module

    captured = {}
    events = []

    class Provider:
        async def structured(self, schema, task, payload):
            captured.update(payload)
            return Judgment(passed=False, reason="把所提供资料边界扩大为现实中所有制度")

    monkeypatch.setattr(module, "get_stream_writer", lambda: events.append)
    agent = RAGAgent(Provider(), None)
    from app.schemas import Citation

    source = Citation(
        source_id="S1",
        document_id="d",
        parent_id="p",
        filename="training.txt",
        location="段1",
        content=supported().evidence[0].quote,
        child_ids=["c"],
    )
    result = await agent.check(
        {
            "query": "培训申请是否有办理期限？",
            "draft": "现实中培训申请不存在任何期限[S1]。",
            "grade": supported(),
            "retrieval": RetrievalResult(sources=[source]),
            "mode": "agent",
            "check_retries": 2,
            "kb": SimpleNamespace(config=RetrievalConfig().model_dump()),
        }
    )
    assert not result["check"].passed and "answer" not in result and result["draft"] == ""
    assert not result["check_retry_allowed"]
    assert captured["evidence_assessment"]["support"] == "supported_limitation"
    assert events[0]["event"] == "draft_reset"


async def test_real_provider_contract_preserves_grade_and_check_boundaries():
    requests = []

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        result = (
            supported().model_dump()
            if len(requests) == 1
            else {
                "checks": [
                    {
                        "answer_span_id": "A:E1",
                        "verdict": "unsupported",
                        "evidence": [],
                        "reason": "无原文支持",
                    }
                ],
            }
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        grade = await provider.structured(
            GradeDecision, "grade", {"query": "办理期限？", "sources": []}
        )
        assert grade.support == "supported_limitation"
        check = await provider.structured(Judgment, "check", {"answer": "扩大断言", "sources": []})
        assert not check.passed
    finally:
        await provider.close()
    assert "主体" in requests[0]["messages"][0]["content"]
    assert "supported_limitation" in requests[0]["messages"][0]["content"]
    assert "现实中不存在任何" in requests[1]["messages"][0]["content"]
