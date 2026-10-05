import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app.answer_parts import bind_answer_parts, constrain_parts_wire
from app.answer_ranges import answer_fragments
from app.check_protocol import answer_spans
from app.check_scope import PREDICATES, validate_check_scope
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import AnswerRangePredicates, PredicateScopeDecision
from tests.test_atomic_scope import BAD, GOOD, SOURCE, AtomicProvider, original
from tests.test_predicate_ranges import selection
from tests.test_providers_parsers import remote_settings


def payload_for(answer):
    focus = answer_spans(answer)[0]
    return {
        "answer": answer,
        "focus_answer_span_id": focus["span_id"],
        "answer_fragments": answer_fragments(focus, answer),
    }


def bind(answer, rows, span="A:E1"):
    return bind_answer_parts(
        AnswerRangePredicates(answer_span_id=span, predicates=rows),
        answer_spans(answer)[0],
        answer,
    )


def test_noncontiguous_shared_scope_stays_separate_with_exact_original_positions():
    answer = " \t按本资料，甲未规定期限，不能据此认定须四天完成。[S1]"
    rows = bind(
        answer,
        [selection(["A:E1:F1", "A:E1:F2"], "P1"), selection(["A:E1:F1", "A:E1:F3"], "P2")],
    )
    second = rows[1]
    assert "original_quote" not in second and "source_start" not in second
    assert [p["fragment_id"] for p in second["answer_parts"]] == ["A:E1:F1", "A:E1:F3"]
    assert second["answer_parts"][0]["source_end"] < second["answer_parts"][1]["source_start"]
    assert not second["literal_binding"]["composite_quote_created"]
    for row in rows:
        for part in row["literal_binding"]["parts"]:
            assert answer[part["source_start"] : part["source_end"]] == part["quote"]
            assert part["normalization_map"][0] == (part["source_start"], part["source_start"] + 1)
        assert not row["literal_binding"]["semantic_support_checked"]


@pytest.mark.parametrize(
    "ids", [["A:E2"], ["A:E2:F1"], ["A:E1:F999"], ["A:E1:F2", "A:E1:F1"], ["A:E1:F2"]]
)
def test_unknown_context_ids_order_and_missing_scope_stay_failures(ids):
    with pytest.raises(CheckProtocolError):
        bind("范围限定，只有通过审批才能领取。[S1]", [selection(ids)])


@pytest.mark.parametrize("change", ["duplicate_id", "duplicate_predicate", "quote_field", "empty"])
def test_parts_schema_never_repairs_invalid_model_selections(change):
    row = selection(["A:E1:F1"])
    rows = [row]
    if change == "duplicate_id":
        row["fragment_ids"].append("A:E1:F1")
    elif change == "duplicate_predicate":
        rows.append(dict(row))
    elif change == "quote_field":
        row["quote"] = "制造的新引文"
    else:
        row["fragment_ids"] = []
    with pytest.raises(ValidationError):
        bind(GOOD, rows)


def test_only_citation_or_punctuation_does_not_count_as_a_predicate():
    with pytest.raises(CheckProtocolError, match="标点或引用"):
        bind("，[S1]，内容。[S1]", [selection(["A:E1:F1"], "P1"), selection(["A:E1:F2"], "P2")])


def test_wire_coverage_includes_scope_and_numeric_parts_without_mutating_base_schema():
    before = AnswerRangePredicates.model_json_schema()
    payload = payload_for("资料限定：金额1,234元。[S1]")
    wire = constrain_parts_wire(AnswerRangePredicates.model_json_schema(), payload)
    ids = [p["fragment_id"] for p in payload["answer_fragments"]]
    assert (
        wire["$defs"]["AnswerRangePredicate"]["properties"]["fragment_ids"]["items"]["enum"] == ids
    )
    required = [
        r["properties"]["predicates"]["contains"]["properties"]["fragment_ids"]["contains"]["const"]
        for r in wire["allOf"]
    ]
    assert required == ids
    assert wire["properties"]["answer_span_id"]["enum"] == ["A:E1"]
    assert AnswerRangePredicates.model_json_schema() == before


class PartsProvider(AtomicProvider):
    supports_check_predicate_parts = True

    async def structured(self, schema, task, payload):
        self.calls.append((task, payload))
        if self.invalid == "timeout":
            raise TimeoutError("受控超时")
        if task == "check_answer_predicate_parts":
            assert schema is AnswerRangePredicates
            assert "query" not in payload and "sources" not in payload
            modes = ["planned", "asserted"] if self.answer == GOOD else ["asserted", "planned"]
            rows = [
                selection(
                    [f"A:E1:F{i + 1}"], f"P{i + 1}", self.mode or modes[i], self.voice or "fact"
                )
                for i in range(2)
            ]
            if self.invalid == "missing_answer":
                rows.pop(0)
            if self.invalid == "unknown_fragment":
                rows[0]["fragment_ids"] = ["A:E2"]
            return schema.model_validate({"answer_span_id": "A:E1", "predicates": rows})
        assert schema is PredicateScopeDecision and task == "check_predicate_parts_scope"
        assert all(set(p) == {"predicate_id", "answer_parts"} for p in payload["answer_predicates"])
        rows = []
        for i, part in enumerate(payload["answer_predicates"]):
            row = {
                "predicate_id": part["predicate_id"],
                "source_mode": "planned" if i == 0 else "asserted",
                "source_voice": "fact",
                "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                "reason": "逐项依据真实原文",
                **{name: "preserved" for name in PREDICATES},
            }
            if self.failed:
                row[self.failed] = "changed"
            if self.invalid == "source":
                row["evidence"][0]["span_id"] = "S1:E999"
            rows.append(row)
        return schema.model_validate({"answer_span_id": "A:E1", "checks": rows})


async def test_real_parts_path_hides_categories_and_keeps_full_unmodified_answer():
    provider = PartsProvider()
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "私有问题")
    assert result.passed and audit["scope_model_requests"] == 2
    assert "私有问题" not in json.dumps(provider.calls[0][1], ensure_ascii=False)
    source_input = provider.calls[1][1]
    assert source_input["answer"] == GOOD
    assert source_input["sources"][0]["content"] == SOURCE["content"]
    assert all("mode" not in p and "voice" not in p for p in source_input["answer_predicates"])
    assert all(
        d["answer_predicate"]["literal_binding"]["match_mode"] == "server_positioned_parts"
        for d in audit["checks"][0]["predicate_checks"]
    )


@pytest.mark.parametrize("failed", PREDICATES)
async def test_position_ids_cannot_upgrade_any_of_the_five_semantic_failures(failed):
    result, audit = await validate_check_scope(
        PartsProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and not audit["checks"][0][failed]


async def test_swapped_subject_modalities_and_unknown_or_wrong_voice_stay_rejected():
    for provider in (
        PartsProvider(answer=BAD),
        PartsProvider(mode="undetermined"),
        PartsProvider(voice="opinion"),
    ):
        answer = provider.answer
        result, _ = await validate_check_scope(provider, original(answer), answer, [SOURCE], "问题")
        assert not result.passed


@pytest.mark.parametrize("invalid", ["missing_answer", "unknown_fragment", "source", "timeout"])
async def test_protocol_failure_or_timeout_does_not_publish_a_supported_answer(invalid):
    with pytest.raises((CheckProtocolError, TimeoutError)):
        await validate_check_scope(
            PartsProvider(invalid=invalid), original(), GOOD, [SOURCE], "问题"
        )


async def test_upstream_rejection_stops_before_part_selection():
    provider = PartsProvider()
    before = original().model_copy(update={"passed": False})
    result, audit = await validate_check_scope(provider, before, GOOD, [SOURCE], "问题")
    assert result is before and audit is None and provider.calls == []


async def test_actual_provider_request_schema_is_isolated_between_concurrent_focuses():
    captured = []

    def handle(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        system = body["messages"][0]["content"].split("JSON 对象：", 1)[1]
        wire, _ = json.JSONDecoder().raw_decode(system)
        captured.append((payload, wire))
        response = {
            "answer_span_id": payload["focus_answer_span_id"],
            "predicates": [selection([p["fragment_id"] for p in payload["answer_fragments"]])],
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(response)}}]}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    first = payload_for("第一条件，内容。[S1]")
    answer = "首句。[S1]第二条件，内容。[S1]"
    second_focus = answer_spans(answer)[1]
    second = {
        "answer": answer,
        "focus_answer_span_id": second_focus["span_id"],
        "answer_fragments": answer_fragments(second_focus, answer),
    }
    try:
        assert provider.supports_check_predicate_parts
        await asyncio.gather(
            *[
                provider.structured(AnswerRangePredicates, "check_answer_predicate_parts", p)
                for p in (first, second)
            ]
        )
        assert len(captured) == 2
        for payload, wire in captured:
            assert wire["properties"]["answer_span_id"]["enum"] == [payload["focus_answer_span_id"]]
            assert wire["$defs"]["AnswerRangePredicate"]["properties"]["fragment_ids"]["items"][
                "enum"
            ] == [p["fragment_id"] for p in payload["answer_fragments"]]
        assert (
            "enum" not in AnswerRangePredicates.model_json_schema()["properties"]["answer_span_id"]
        )
    finally:
        await provider.close()
