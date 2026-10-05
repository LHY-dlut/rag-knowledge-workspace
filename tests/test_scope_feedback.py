import copy

import pytest

from app.check_scope import PREDICATES, validate_check_scope
from app.scope_feedback import FIELDS, atomic_scope_feedback
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_grounded_relations import GroundedProvider


@pytest.mark.parametrize("failed", PREDICATES)
async def test_each_negative_relation_is_named_and_raw_model_reason_is_retained(failed):
    result, audit = await validate_check_scope(
        GroundedProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and FIELDS[failed] in result.reason
    assert "模型关系核验未通过" in result.reason
    assert all(p["reason"] == "当前事实与原文对应" for p in audit["checks"][0]["predicate_checks"])


@pytest.mark.parametrize("axis", ["mode", "voice"])
async def test_backend_category_veto_cannot_be_explained_only_by_positive_model_reason(axis):
    provider = GroundedProvider(**{f"source_{axis}_swap": True})
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
    assert not result.passed and "类别不一致或待定" in result.reason
    assert "不代表后端通过" in result.reason
    assert "当前事实与原文对应" not in result.reason
    rows = audit["checks"][0]["predicate_checks"]
    assert rows[0]["reason"] == "当前事实与原文对应"
    assert not rows[0]["modality_preserved" if axis == "mode" else "attribution_preserved"]


async def test_feedback_does_not_mutate_any_verdict_binding_or_raw_reason():
    result, audit = await validate_check_scope(
        GroundedProvider(source_mode_swap=True, source_voice_swap=True),
        original(),
        GOOD,
        [SOURCE],
        "问题",
    )
    rows = audit["checks"][0]["predicate_checks"]
    before = copy.deepcopy(rows)
    text = atomic_scope_feedback(rows)
    assert rows == before and not result.passed
    assert "答案=planned，原文=asserted" in text
    assert "答案=fact，原文=opinion" in text
    assert len(text) <= 300


async def test_accepted_answer_feedback_and_evidence_remain_unchanged():
    result, audit = await validate_check_scope(
        GroundedProvider(), original(), GOOD, [SOURCE], "问题"
    )
    assert result.passed
    assert audit["checks"][0]["reason"] == "所有逐谓词关系及独立语气归属均保留"
    assert all(all(row[n] for n in PREDICATES) for row in audit["checks"])
