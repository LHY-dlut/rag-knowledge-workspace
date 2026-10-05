import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent import RAGAgent, bind_grade_evidence, citation_check, validate_grade_support
from app.evidence_protocol import bind_decision, match_quote, prepare_citations
from app.schemas import (
    Citation,
    GradeBindingDecision,
    GradeDecision,
    Judgment,
    RetrievalConfig,
    RetrievalResult,
)


def decision(quote, sid="S1", support="supported_answer"):
    return GradeDecision(
        passed=True,
        support=support,
        reason="原文有据",
        answer_scope="仅根据所给制度",
        evidence=[dict(source_id=sid, quote=quote, subject="设备维修", attribute="时限")],
    )


@pytest.mark.parametrize(
    "answer,passed,error",
    [
        ("有据[S1]", True, None),
        ("有据[ S1 ][S1]", True, None),
        ("有据［\nS1\t］。", True, None),
        ("有据[S1][ S99 ]", False, "citation_id_unknown"),
        ("有据（S1）", False, "citation_format_invalid"),
        ("有据[S1]另据[S99", False, "citation_format_invalid"),
        ("有据[S1]另据[s99]", False, "citation_format_invalid"),
        ("有据[S1]另据[S1,S99]", False, "citation_format_invalid"),
        ("没有证据", False, "citation_format_invalid"),
    ],
)
def test_all_markers_checked_before_rendering(answer, passed, error):
    rendered, audit = prepare_citations(answer, [{"source_id": "S1"}])
    assert audit["passed"] is passed
    assert citation_check(answer, [{"source_id": "S1"}]).passed is passed
    if error:
        assert error in audit["failure_types"] and rendered == answer
    else:
        assert "[S1]" in rendered and not any(c in rendered for c in ["［", "］"])
    assert audit["raw_answer"] == answer


@pytest.mark.parametrize(
    "text,quote,passed",
    [
        ("甲：不予批准。", "甲：不予批准。", True),
        ("甲：\r\n  不予批准。", "甲： 不予批准。", True),
        ("甲：\t不予批准。", "甲：\n不予批准。", True),
        ("甲：不予批准。", "甲:不予批准。", False),
        ("甲：先申请；后审批。", "甲：先申请……后审批。", False),
        ("金额1 00元", "金额100元", False),
        ("报道说明：收入达9亿元\xa0...", "收入达9亿元...", True),
        ("原文称…后续内容", "原文称 \n…后续内容", True),
        ("收入达9亿元，持续增长。", "收入达9亿元...", False),
        ("数值9 . 1", "数值9.1", False),
        ("三个点 . . .", "三个点...", False),
        ("not able", "notable", False),
        ("甲\n乙 丙 甲\t乙", "甲 乙", False),
        ("有证据。", "不存在的事实", False),
        ("", "有证据", False),
    ],
)
def test_continuous_quote_and_reversible_whitespace_mapping(text, quote, passed):
    result = match_quote(quote, {"source_id": "S1", "content": text})
    assert result["passed"] is passed
    if passed:
        assert text[result["source_start"] : result["source_end"]] == result["original_quote"]
        assert all(0 <= s < e <= len(text) for s, e in result["normalization_map"])
        assert result["model_quote"] == quote


def test_cross_child_quote_works_only_inside_actual_continuous_parent():
    sources = [{"source_id": "S1", "content": "先申请。后审批。", "child_ids": ["c1", "c2"]}]
    assert bind_grade_evidence(decision("先申请。后审批。"), sources).passed
    split = [{"source_id": "S1", "content": "先申请。"}, {"source_id": "S2", "content": "后审批。"}]
    assert not bind_grade_evidence(decision("先申请。后审批。"), split).passed
    two = decision("先申请。")
    two.evidence.append(decision("后审批。", "S2").evidence[0])
    assert bind_grade_evidence(two, split).passed


async def test_exact_quote_does_not_override_subject_attribute_failure():
    class Provider:
        async def structured(self, schema, task, payload):
            assert task == "grade_binding"
            return GradeBindingDecision(
                subject_matches=False,
                attribute_matches=False,
                explicit_limitation=True,
                scope_supported=False,
                reason="报销不能迁移到维修",
            )

    d = decision("报销应在7天内办理。", support="supported_limitation")
    result, binding = await validate_grade_support(
        Provider(),
        d,
        [{"source_id": "S1", "content": d.evidence[0].quote}],
        "维修必须7天内完成吗？",
    )
    assert not result.passed and not binding.passed


@pytest.mark.parametrize("semantic_pass", [True, False])
async def test_rendered_answer_still_requires_independent_check(monkeypatch, semantic_pass):
    import app.agent as module

    monkeypatch.setattr(module, "get_stream_writer", lambda: lambda event: None)

    class Provider:
        async def structured(self, schema, task, payload):
            assert task == "check" and payload["answer"] == "次日到账[S1]"
            return Judgment(
                passed=semantic_pass, reason="证据有据" if semantic_pass else "仅材料不支持到账"
            )

    src = Citation(
        source_id="S1",
        document_id="d",
        parent_id="p",
        filename="p.txt",
        location="原文",
        content="需提交发票和审批单。",
        child_ids=["c"],
    )
    state = dict(
        draft="次日到账[ S1 ]",
        query="能保证次日到账吗？",
        mode="agent",
        retrieval=RetrievalResult(sources=[src]),
        grade=decision(src.content),
        kb=SimpleNamespace(config=RetrievalConfig().model_dump()),
        check_retries=2,
    )
    result = await RAGAgent(Provider(), None).check(state)
    assert result["check"].passed is semantic_pass
    if semantic_pass:
        assert result["answer"] == "次日到账[S1]"
    else:
        assert "answer" not in result and not result["check_retry_allowed"]


async def test_bad_quote_feedback_reaches_next_grade_without_extra_retry(monkeypatch):
    seen = []

    class Provider:
        async def structured(self, schema, task, payload):
            seen.append(payload)
            return decision("甲……丙") if len(seen) == 1 else decision("甲乙丙")

    src = Citation(
        source_id="S1",
        document_id="d",
        parent_id="p",
        filename="p.txt",
        location="原文",
        content="甲乙丙",
        child_ids=["c"],
    )
    agent = RAGAgent(Provider(), None)
    state = dict(
        query="时限？",
        mode="agent",
        retrieval=RetrievalResult(sources=[src]),
        kb=SimpleNamespace(config=RetrievalConfig().model_dump()),
        grade_retries=2,
    )
    first = await agent.grade(state)
    assert not first["grade"].passed and first["grade_retries"] == 3
    second = await agent.grade({**state, **first})
    assert second["grade"].passed and "protocol_feedback" in seen[1]
    assert second["protocol_feedback"] == ""


@pytest.mark.parametrize(
    "quote,sid,error",
    [
        ("需经理签字...", "S1", "evidence_text_mismatch"),
        ("需主管批准。", "S1", "evidence_text_mismatch"),
        ("需经理签字。", "S99", "citation_id_unknown"),
    ],
)
def test_structured_retry_reports_actual_bad_evidence_without_fixing_it(quote, sid, error):
    original = decision(quote, sid)
    before = original.model_dump()
    bound, audit = bind_decision(original, [{"source_id": "S1", "content": "需经理签字。"}])
    assert not bound.passed and original.model_dump() == before
    assert audit["retry_feedback"]["invalid_evidence"] == [
        {"source_id": sid, "model_quote": quote, "failure_type": error}
    ]
    assert "不要添加原文不存在" in audit["retry_feedback"]["instruction"]
    assert "original_quote" not in audit["evidence"][0]


async def test_corrected_quote_still_cannot_override_semantic_binding_failure():
    calls = []

    class Provider:
        async def structured(self, schema, task, payload):
            calls.append(task)
            if task == "grade_binding":
                return GradeBindingDecision(
                    subject_matches=False,
                    attribute_matches=True,
                    explicit_limitation=True,
                    scope_supported=False,
                    reason="采购规则不能迁移为维修规则",
                )
            if "protocol_feedback" not in payload:
                return decision("采购未规定处理时限...", support="supported_limitation")
            bad = payload["protocol_feedback"]["invalid_evidence"]
            assert bad[0]["model_quote"] == "采购未规定处理时限..."
            return decision("采购未规定处理时限。", support="supported_limitation")

    src = Citation(
        source_id="S1",
        document_id="d",
        parent_id="p",
        filename="rules.txt",
        location="原文",
        content="采购未规定处理时限。",
        child_ids=["c"],
    )
    state = dict(
        query="维修是否没有时限？",
        mode="agent",
        retrieval=RetrievalResult(sources=[src]),
        kb=SimpleNamespace(config=RetrievalConfig().model_dump()),
        grade_retries=2,
    )
    agent = RAGAgent(Provider(), None)
    first = await agent.grade(state)
    second = await agent.grade({**state, **first})
    assert not first["grade"].passed and first["grade_retry_allowed"]
    assert second["grade_protocol"]["passed"] and not second["grade"].passed
    assert not second["grade_retry_allowed"] and second["grade_binding"].subject_matches is False
    assert calls == ["grade", "grade", "grade_binding"]


async def test_detailed_protocol_feedback_does_not_extend_exhausted_retry_limit():
    class Provider:
        async def structured(self, schema, task, payload):
            return decision("原文不存在的摘录...")

    src = Citation(
        source_id="S1",
        document_id="d",
        parent_id="p",
        filename="r.txt",
        location="原文",
        content="明确规定先申请。",
        child_ids=["c"],
    )
    result = await RAGAgent(Provider(), None).grade(
        dict(
            query="需要申请吗？",
            mode="agent",
            retrieval=RetrievalResult(sources=[src]),
            kb=SimpleNamespace(config=RetrievalConfig().model_dump()),
            grade_retries=3,
        )
    )
    assert not result["grade"].passed and not result["grade_retry_allowed"]
    assert "grade_retries" not in result
    assert result["protocol_feedback"]["invalid_evidence"][0]["source_id"] == "S1"


@pytest.mark.parametrize("index", range(5))
async def test_original_five_frozen_boundary_protocol_only_mock(index):
    """Reuse human labels as a Mock oracle; not a fresh real-model semantic evaluation."""
    path = Path(__file__).parents[1] / "artifacts/human_eval_20261002_v1/confirmed_dataset.json"
    data = path.read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == "4e81e85c37758da209fe660b0bc0ada1567b3e608f6ffc594e458af978eaa944"
    )
    case = json.loads(data)["cases"][index]
    quote = case["provenance"][0]["quote"] if case["provenance"] else "报销需发票和审批单。"
    src = Citation(
        source_id="S1",
        document_id="d",
        parent_id="p",
        filename="p.txt",
        location="原冻结原文",
        content=quote,
        child_ids=["c"],
    )

    class Provider:
        async def structured(self, schema, task, payload):
            if task == "grade_binding":
                return GradeBindingDecision(
                    subject_matches=True,
                    attribute_matches=True,
                    explicit_limitation=True,
                    scope_supported=True,
                    reason="冻结限定证据",
                )
            assert task == "grade" and payload["query"] == case["question"]
            if not case["answerable"]:
                return GradeDecision(passed=False, support="insufficient", reason="所问事实无依据")
            return decision(
                quote, support="supported_limitation" if index == 3 else "supported_answer"
            )

    state = dict(
        query=case["question"],
        history=case["history"],
        mode="agent",
        retrieval=RetrievalResult(sources=[src]),
        grade_retries=3,
        kb=SimpleNamespace(config=RetrievalConfig().model_dump()),
    )
    result = await RAGAgent(Provider(), None).grade(state)
    assert result["grade"].passed is case["answerable"]
    if not case["answerable"]:
        assert case["relevant_child_ids"] == [] and not result["grade_retry_allowed"]
    assert path.read_bytes() == data
