import json

import httpx
import pytest
from pydantic import ValidationError

from app.answer_ranges import answer_fragments, bind_answer_ranges
from app.check_protocol import answer_spans
from app.check_scope import PREDICATES, validate_check_scope
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import AnswerRangePredicates
from tests.test_atomic_scope import BAD, GOOD, SOURCE, AtomicProvider, original
from tests.test_providers_parsers import remote_settings


def selection(ids, predicate_id="P1", mode="asserted", voice="fact"):
    return {
        "predicate_id": predicate_id,
        "fragment_ids": ids,
        "mode": mode,
        "voice": voice,
        "reason": "仅选择注册的真实范围",
    }


class RangeProvider(AtomicProvider):
    supports_check_predicate_ranges = True

    async def structured(self, schema, task, payload):
        if task != "check_answer_predicate_ranges":
            return await super().structured(schema, task, payload)
        self.calls.append((task, payload))
        assert schema is AnswerRangePredicates
        assert set(payload) == {
            "answer",
            "focus_answer_span_id",
            "answer_spans",
            "answer_context_spans",
            "answer_fragments",
        }
        assert [f["fragment_id"] for f in payload["answer_fragments"]] == ["A:E1:F1", "A:E1:F2"]
        modes = ["planned", "asserted"] if self.answer == GOOD else ["asserted", "planned"]
        rows = [
            selection(["A:E1:F1"], "P1", self.mode or modes[0], self.voice or "fact"),
            selection(["A:E1:F2"], "P2", modes[1]),
        ]
        if self.invalid == "missing_answer":
            rows.pop(0)
        if self.invalid == "unknown_fragment":
            rows[0]["fragment_ids"] = ["A:E2:F1"]
        return schema.model_validate({"answer_span_id": "A:E1", "predicates": rows})


def test_fragment_registry_keeps_original_punctuation_unicode_and_absolute_offsets():
    answer = " \t资料范围：甲😀，金额1,234元。[S1]"
    focus = answer_spans(answer)[0]
    fragments = answer_fragments(focus, answer)
    assert [f["quote"] for f in fragments] == ["资料范围：", "甲😀，", "金额1,", "234元。"]
    assert fragments[0]["source_start"] == 2
    assert "".join(f["quote"] for f in fragments) == focus["quote"]
    assert all(answer[f["source_start"] : f["source_end"]] == f["quote"] for f in fragments)
    decision = AnswerRangePredicates(
        answer_span_id=focus["span_id"],
        predicates=[selection([f["fragment_id"] for f in fragments])],
    )
    bound = bind_answer_ranges(decision, focus, answer)[0]
    assert bound["original_quote"] == "资料范围：甲😀，金额1,234元。"
    assert bound["literal_binding"]["normalization_map"][0] == (2, 3)
    assert not bound["literal_binding"]["semantic_support_checked"]
    assert "model_quote" not in bound["literal_binding"]


@pytest.mark.parametrize(
    "ids", [["A:E1:F999"], ["A:E2:F1"], ["A:E1:F2", "A:E1:F1"], ["A:E1:F1", "A:E1:F3"], ["A:E1:F2"]]
)
def test_unknown_reordered_noncontiguous_or_missing_qualifier_ranges_fail(ids):
    answer = "根据本资料，甲未规定期限，不能据此认定必须四天完成。[S1]"
    focus = answer_spans(answer)[0]
    decision = AnswerRangePredicates(
        answer_span_id=focus["span_id"], predicates=[selection(ids, mode="document_limitation")]
    )
    with pytest.raises(CheckProtocolError):
        bind_answer_ranges(decision, focus, answer)


def test_explicit_overlapping_contiguous_ranges_keep_shared_document_limitation():
    answer = "根据本资料，甲未规定期限，不能据此认定必须四天完成。[S1]"
    focus = answer_spans(answer)[0]
    ids = [f["fragment_id"] for f in answer_fragments(focus, answer)]
    decision = AnswerRangePredicates(
        answer_span_id="A:E1",
        predicates=[
            selection(ids[:2], "P1", "document_limitation"),
            selection(ids, "P2", "document_limitation"),
        ],
    )
    rows = bind_answer_ranges(decision, focus, answer)
    assert rows[0]["original_quote"] == "根据本资料，甲未规定期限，"
    assert rows[1]["original_quote"] == focus["quote"]
    assert all(row["original_quote"].startswith("根据本资料") for row in rows)


def test_repeated_phrases_are_disambiguated_only_by_explicit_server_ids():
    answer = "培训，培训，已完成。[S1]"
    focus = answer_spans(answer)[0]
    decision = AnswerRangePredicates(
        answer_span_id="A:E1",
        predicates=[
            selection(["A:E1:F1"], "P1"),
            selection(["A:E1:F2"], "P2"),
            selection(["A:E1:F3"], "P3"),
        ],
    )
    rows = bind_answer_ranges(decision, focus, answer)
    assert rows[0]["original_quote"] == rows[1]["original_quote"] == "培训，"
    assert rows[0]["source_start"] == 0 and rows[1]["source_start"] == 3


@pytest.mark.parametrize(
    "change",
    ["duplicate_fragment", "duplicate_predicate", "empty", "boolean", "quote_field", "long_reason"],
)
def test_malformed_range_wire_is_not_repaired(change):
    row = selection(["A:E1:F1"])
    rows = [row]
    if change == "duplicate_fragment":
        row["fragment_ids"].append("A:E1:F1")
    elif change == "duplicate_predicate":
        rows.append(dict(row))
    elif change == "empty":
        row["fragment_ids"] = []
    elif change == "boolean":
        row["fragment_ids"] = [True]
    elif change == "quote_field":
        row["quote"] = "模型不得手写替代摘录"
    else:
        row["reason"] = "证" * 301
    with pytest.raises(ValidationError):
        AnswerRangePredicates(answer_span_id="A:E1", predicates=rows)


def test_stale_server_registry_cannot_select_text_from_a_different_answer():
    focus = {**answer_spans(GOOD)[0], "quote": "不同原文"}
    with pytest.raises(CheckProtocolError, match="不一致"):
        answer_fragments(focus, GOOD)


async def test_real_range_path_keeps_matching_support_and_hides_independent_categories():
    provider = RangeProvider()
    upstream = original()
    final, audit = await validate_check_scope(
        provider, upstream, GOOD, [SOURCE], "问题不传独立读取"
    )
    assert final.passed and final.evidence_protocol == upstream.evidence_protocol
    assert len(provider.calls) == audit["scope_model_requests"] == 2
    assert "query" not in provider.calls[0][1] and "sources" not in provider.calls[0][1]
    assert "问题不传独立读取" not in json.dumps(provider.calls[0][1], ensure_ascii=False)
    assert all(
        "mode" not in p and "voice" not in p and "reason" not in p
        for p in provider.calls[1][1]["answer_predicates"]
    )
    assert all(
        d["answer_predicate"]["literal_binding"]["match_mode"] == "server_fragment_range"
        for d in audit["checks"][0]["predicate_checks"]
    )


async def test_range_path_rejects_swapped_modalities_even_when_global_tags_match():
    provider = RangeProvider(answer=BAD)
    final, audit = await validate_check_scope(
        provider, original(BAD), BAD, [SOURCE], "相同事实集合"
    )
    assert not final.passed and all(
        not d["mode_matches_this_predicate"] for d in audit["checks"][0]["predicate_checks"]
    )


@pytest.mark.parametrize("failed", PREDICATES)
async def test_range_ids_do_not_upgrade_any_semantic_negative(failed):
    final, audit = await validate_check_scope(
        RangeProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not final.passed and not audit["checks"][0][failed]


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_answer",
        "unknown_fragment",
        "unknown_source",
        "unknown_source_span",
        "wrong_predicate",
    ],
)
async def test_range_path_identity_and_coverage_fail_closed(invalid):
    provider = RangeProvider(invalid=invalid)
    with pytest.raises(CheckProtocolError):
        await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
    if invalid in ("missing_answer", "unknown_fragment"):
        assert len(provider.calls) == 1


async def test_rejected_upstream_stops_before_range_dispatch():
    provider = RangeProvider()
    before = original().model_copy(update={"passed": False})
    result, audit = await validate_check_scope(provider, before, GOOD, [SOURCE], "问题")
    assert result is before and audit is None and provider.calls == []


async def test_provider_range_schema_is_mandatory_and_invalid_json_stays_a_failure():
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
        assert provider.supports_check_predicate_ranges
        with pytest.raises(CheckProtocolError):
            await provider.structured(
                AnswerRangePredicates, "check_answer_predicate_ranges", {"answer": GOOD}
            )
        assert '"AnswerRangePredicates"' in captured[0]["messages"][0]["content"]
        assert json.loads(captured[0]["messages"][1]["content"]) == {"answer": GOOD}
    finally:
        await provider.close()
