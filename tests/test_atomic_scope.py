import json

import httpx
import pytest
from pydantic import ValidationError

from app.atomic_scope import bind_answer_predicates
from app.check_protocol import answer_spans, bind_check
from app.check_scope import PREDICATES, validate_check_scope
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import (
    AnswerPredicates,
    CheckDecision,
    CheckScopeProjection,
    PredicateScopeDecision,
)
from tests.test_providers_parsers import remote_settings

SOURCE = {
    "source_id": "S1",
    "content": "培训拟下周进行，值班已开始。",
    "document_id": "doc-local",
    "parent_id": "p-local",
}
GOOD = "培训拟下周进行，值班已开始。[S1]"
BAD = "培训已进行，值班拟下周开始。[S1]"


def original(answer=GOOD):
    return bind_check(
        CheckDecision(
            checks=[
                {
                    "answer_span_id": "A:E1",
                    "verdict": "supported",
                    "reason": "构造上游支持，检查后续不能遗漏语气差异",
                    "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                }
            ]
        ),
        answer,
        [SOURCE],
    )


class AtomicProvider:
    supports_check_scope_binding = True
    supports_check_atomic_scope = True

    def __init__(self, answer=GOOD, failed=None, mode=None, voice=None, invalid=None):
        self.answer, self.failed, self.mode, self.voice, self.invalid = (
            answer,
            failed,
            mode,
            voice,
            invalid,
        )
        self.calls = []

    async def structured(self, schema, task, payload):
        self.calls.append((task, payload))
        if task == "check_answer_predicates":
            assert schema is AnswerPredicates
            assert set(payload) == {
                "answer",
                "focus_answer_span_id",
                "answer_spans",
                "answer_context_spans",
            }
            clauses = self.answer.split("[S1]")[0].split("，")
            modes = ["planned", "asserted"] if self.answer == GOOD else ["asserted", "planned"]
            result = {
                "answer_span_id": "A:E1",
                "predicates": [
                    {
                        "predicate_id": "P1",
                        "quote": clauses[0] + "，",
                        "mode": self.mode or modes[0],
                        "voice": self.voice or "fact",
                        "reason": "仅依据当前答案第一关系",
                    },
                    {
                        "predicate_id": "P2",
                        "quote": clauses[1],
                        "mode": modes[1],
                        "voice": "fact",
                        "reason": "仅依据当前答案第二关系",
                    },
                ],
            }
            if self.invalid == "quote_not_in_answer":
                result["predicates"][0]["quote"] = "原文有而实际答案没有的关系"
            if self.invalid == "missing_answer":
                result["predicates"].pop()
            if self.invalid == "wrong_answer_span":
                result["answer_span_id"] = "A:E999"
            return schema.model_validate(result)
        assert schema is PredicateScopeDecision and task == "check_predicate_scope"
        assert all(
            set(p) == {"predicate_id", "original_quote", "source_start", "source_end"}
            for p in payload["answer_predicates"]
        )
        result = {"answer_span_id": "A:E1", "checks": []}
        for i, p in enumerate(payload["answer_predicates"]):
            row = {
                "predicate_id": p["predicate_id"],
                "source_mode": self.mode or ["planned", "asserted"][i],
                "source_voice": "fact",
                "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                "reason": "分别核对原文中对应关系",
                **{n: "preserved" for n in PREDICATES},
            }
            if self.failed:
                row[self.failed] = "changed"
            if self.invalid == "unknown_source":
                row["evidence"][0]["source_id"] = "S999"
            if self.invalid == "unknown_source_span":
                row["evidence"][0]["span_id"] = "S1:E999"
            result["checks"].append(row)
        if self.invalid == "missing_predicate":
            result["checks"].pop()
        if self.invalid == "wrong_predicate":
            result["checks"][0]["predicate_id"] = "P999"
        return schema.model_validate(result)


async def test_matching_literal_predicates_keep_support_and_original_positions():
    provider = AtomicProvider()
    upstream = original()
    final, audit = await validate_check_scope(
        provider, upstream, GOOD, [SOURCE], "私有问题不传投影"
    )
    assert final.passed and upstream.passed
    assert final.evidence_protocol == upstream.evidence_protocol
    assert len(provider.calls) == audit["scope_model_requests"] == 2
    assert "sources" not in provider.calls[0][1] and "query" not in provider.calls[0][1]
    assert "私有问题不传投影" not in json.dumps(provider.calls[0][1], ensure_ascii=False)
    assert audit["per_predicate_scope_binding"] and not audit["can_upgrade_rejection"]
    details = audit["checks"][0]["predicate_checks"]
    for detail in details:
        bound = detail["answer_predicate"]
        assert GOOD[bound["source_start"] : bound["source_end"]] == bound["original_quote"]
        assert detail["literal_evidence"][0]["document_id"] == "doc-local"


async def test_equal_global_tag_sets_do_not_accept_swapped_predicate_modalities():
    whole = CheckScopeProjection(
        answer_modes=["asserted", "planned"],
        source_modes=["planned", "asserted"],
        answer_voices=["fact"],
        source_voices=["fact"],
    )
    assert whole.preserves_modes() and whole.preserves_voices()
    provider = AtomicProvider(answer=BAD)
    upstream = original(BAD)
    final, audit = await validate_check_scope(provider, upstream, BAD, [SOURCE], "同一组事实")
    assert upstream.passed and not final.passed
    assert all(not d["mode_matches_this_predicate"] for d in audit["checks"][0]["predicate_checks"])
    assert not audit["can_upgrade_rejection"]


@pytest.mark.parametrize("failed", PREDICATES)
async def test_no_atomic_relation_negative_can_be_upgraded(failed):
    final, audit = await validate_check_scope(
        AtomicProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not final.passed and not audit["checks"][0][failed]


@pytest.mark.parametrize(
    "invalid",
    [
        "quote_not_in_answer",
        "missing_answer",
        "wrong_answer_span",
        "unknown_source",
        "unknown_source_span",
        "missing_predicate",
        "wrong_predicate",
    ],
)
async def test_atomic_binding_and_coverage_fail_closed(invalid):
    with pytest.raises(CheckProtocolError):
        await validate_check_scope(
            AtomicProvider(invalid=invalid), original(), GOOD, [SOURCE], "问题"
        )


async def test_unknown_mode_and_wrong_voice_are_not_a_matching_success():
    for provider in (AtomicProvider(mode="undetermined"), AtomicProvider(voice="opinion")):
        final, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
        assert not final.passed
        if provider.voice:
            assert not audit["checks"][0]["attribution_preserved"]
        else:
            assert not audit["checks"][0]["modality_preserved"]


async def test_rejected_upstream_does_not_dispatch_atomic_checks():
    provider = AtomicProvider()
    upstream = original().model_copy(update={"passed": False})
    final, audit = await validate_check_scope(provider, upstream, GOOD, [SOURCE], "问题")
    assert final is upstream and audit is None and provider.calls == []


@pytest.mark.parametrize("change", ["duplicate", "bool_mode", "long_reason", "empty_quote"])
def test_invalid_answer_predicate_wire_has_no_silent_repair(change):
    row = {
        "predicate_id": "P1",
        "quote": "培训",
        "mode": "planned",
        "voice": "fact",
        "reason": "限定",
    }
    rows = [row]
    if change == "duplicate":
        rows.append(dict(row))
    elif change == "bool_mode":
        row["mode"] = True
    elif change == "long_reason":
        row["reason"] = "证" * 301
    else:
        row["quote"] = ""
    with pytest.raises(ValidationError):
        AnswerPredicates(answer_span_id="A:E1", predicates=rows)


def test_duplicate_scope_references_and_ids_are_rejected():
    row = {
        "predicate_id": "P1",
        "source_mode": "planned",
        "source_voice": "fact",
        "reason": "匹配",
        "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
        **{n: "preserved" for n in PREDICATES},
    }
    with pytest.raises(ValidationError):
        PredicateScopeDecision(answer_span_id="A:E1", checks=[row, row])
    row["evidence"].append(dict(row["evidence"][0]))
    with pytest.raises(ValidationError):
        PredicateScopeDecision(answer_span_id="A:E1", checks=[row])


def test_missing_shared_condition_cannot_hide_outside_literal_quote_coverage():
    answer = "通过资质审核后才能申请。[S1]"
    decision = AnswerPredicates(
        answer_span_id="A:E1",
        predicates=[
            {
                "predicate_id": "P1",
                "quote": "才能申请。",
                "mode": "permission",
                "voice": "fact",
                "reason": "省略前提的无效投影",
            }
        ],
    )
    with pytest.raises(CheckProtocolError, match="遗漏"):
        bind_answer_predicates(decision, answer_spans(answer)[0], answer)


async def test_actual_provider_uses_strict_atomic_schema_and_keeps_bad_json_failed():
    captured = []

    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer_span_id":"A:E1","predicates":[]}'}}]
            },
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        assert provider.supports_check_atomic_scope
        with pytest.raises(CheckProtocolError):
            await provider.structured(AnswerPredicates, "check_answer_predicates", {"answer": GOOD})
        assert '"AnswerPredicates"' in captured[0]["messages"][0]["content"]
        assert json.loads(captured[0]["messages"][1]["content"]) == {"answer": GOOD}
    finally:
        await provider.close()


def test_atomic_literal_binding_positions_are_relative_to_unchanged_full_answer():
    answer = " \t" + GOOD
    focus = answer_spans(answer)[0]
    assert focus["source_start"] == 2
    decision = AnswerPredicates(
        answer_span_id=focus["span_id"],
        predicates=[
            {
                "predicate_id": "P1",
                "quote": GOOD.split("[S1]")[0],
                "mode": "planned",
                "voice": "fact",
                "reason": "偏移回归用的真实连续片段",
            }
        ],
    )
    row = bind_answer_predicates(decision, focus, answer)[0]
    binding = row["literal_binding"]
    assert binding["source_start"] == row["source_start"] == 2
    assert binding["source_end"] == row["source_end"]
    assert binding["normalization_map"][0] == (2, 3)
    assert answer[binding["source_start"] : binding["source_end"]] == row["original_quote"]


def test_repeated_literal_predicate_cannot_silently_choose_first_occurrence():
    answer = "培训培训。[S1]"
    decision = AnswerPredicates(
        answer_span_id="A:E1",
        predicates=[
            {
                "predicate_id": "P1",
                "quote": "培训",
                "mode": "asserted",
                "voice": "fact",
                "reason": "重复短语位置不唯一",
            }
        ],
    )
    with pytest.raises(CheckProtocolError, match="歧义"):
        bind_answer_predicates(decision, answer_spans(answer)[0], answer)
