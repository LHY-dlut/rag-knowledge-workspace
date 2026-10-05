import json

import httpx
import pytest

from app.answer_ranges import answer_fragments
from app.check_protocol import answer_spans
from app.check_scope import PREDICATES, validate_check_scope
from app.evidence_protocol import evidence_spans
from app.providers import CheckProtocolError, DashScopeProvider
from app.source_parts import (
    PairedPredicateScopeDecision,
    SourcePredicateParts,
    bind_source_parts,
    validate_source_pairings,
)
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_predicate_focus import FocusProvider
from tests.test_providers_parsers import remote_settings


def source_value(source=SOURCE, focus=None):
    focus = focus or evidence_spans(source)[0]
    parts = answer_fragments(focus, source["content"])
    return {
        "source_id": source["source_id"],
        "source_span_id": focus["span_id"],
        "predicates": [
            {
                "predicate_id": f"R{i + 1}",
                "fragment_ids": [p["fragment_id"]],
                "mode": "planned" if i == 0 else "asserted",
                "voice": "fact",
                "reason": "读取当前原文关系",
            }
            for i, p in enumerate(parts)
        ],
    }


class PairedProvider(FocusProvider):
    supports_check_source_predicate_parts = True

    async def structured(self, schema, task, payload):
        if task == "check_source_predicate_parts":
            self.calls.append((task, payload))
            assert schema is SourcePredicateParts
            assert "answer" not in payload and "query" not in payload
            focus = payload["focus_source_span"]
            assert (
                payload["source_text"][focus["source_start"] : focus["source_end"]]
                == focus["quote"]
            )
            value = source_value(
                {"source_id": payload["source_id"], "content": payload["source_text"]}, focus
            )
            if self.invalid == "foreign_source":
                value["source_id"] = "S9"
            if self.invalid == "foreign_fragment":
                value["predicates"][0]["fragment_ids"] = ["S1:E999:F1"]
            if self.invalid == "missing_part":
                value["predicates"] = value["predicates"][:1]
            if self.invalid == "timeout_source":
                raise TimeoutError("受控来源投影超时")
            if self.invalid in {"undetermined_mode", "undetermined_voice"}:
                field = "mode" if self.invalid == "undetermined_mode" else "voice"
                value["predicates"][1][field] = "undetermined"
            return schema.model_validate(value)
        if task != "check_paired_predicate_scope":
            return await super().structured(schema, task, payload)
        self.calls.append((task, payload))
        assert schema is PairedPredicateScopeDecision
        assert all("mode" not in p and "voice" not in p for p in payload["source_predicates"])
        assert all("reason" not in p for p in payload["source_predicates"])
        pid = payload["focus_predicate_id"]
        index = int(pid[1:]) - 1
        selected = payload["source_predicates"][index]
        row = {
            "predicate_id": pid,
            "source_mode": "planned" if index == 0 else "asserted",
            "source_voice": "fact",
            "source_predicate_ids": [selected["source_predicate_id"]],
            "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
            "reason": "只比较当前已定位关系",
            **{n: "changed" if n == self.failed else "preserved" for n in PREDICATES},
        }
        if self.invalid == "mode_swap":
            row["source_mode"] = "asserted" if index == 0 else "planned"
        if self.invalid == "voice_swap":
            row["source_voice"] = "opinion"
        if self.invalid == "foreign_pair":
            row["source_predicate_ids"] = ["S9:E99:RP9"]
        return schema.model_validate({"answer_span_id": "A:E1", "checks": [row]})


async def test_independent_source_is_positioned_and_hidden_from_comparative_categories():
    provider = PairedProvider()
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "私有问题")
    assert result.passed and audit["scope_model_requests"] == len(provider.calls) == 4
    assert [t for t, _ in provider.calls] == [
        "check_answer_predicate_parts",
        "check_source_predicate_parts",
        "check_paired_predicate_scope",
        "check_paired_predicate_scope",
    ]
    independent = provider.calls[1][1]
    assert not {"answer", "query", "answer_predicates", "answer_context_spans"} & independent.keys()
    claims = audit["checks"][0]
    assert claims["source_projection_has_question_or_answer"] is False
    for part in claims["independent_source_predicates"]:
        assert part["literal_binding"]["source_id"] == "S1"
        assert part["literal_binding"]["offset_unit"] == "unchanged full source Unicode codepoint"
        for raw in part["source_parts"]:
            assert SOURCE["content"][raw["source_start"] : raw["source_end"]] == raw["quote"]


@pytest.mark.parametrize("failed", PREDICATES)
async def test_new_pairing_cannot_override_any_negative_relation(failed):
    result, audit = await validate_check_scope(
        PairedProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and not audit["checks"][0][failed]


@pytest.mark.parametrize("invalid", ["mode_swap", "voice_swap"])
async def test_independent_source_mode_and_voice_are_additional_vetoes(invalid):
    result, audit = await validate_check_scope(
        PairedProvider(invalid=invalid), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed
    name = (
        "source_mode_matches_independent_parts"
        if invalid == "mode_swap"
        else "source_voice_matches_independent_parts"
    )
    assert all(not p[name] for p in audit["checks"][0]["source_pairing_checks"])


@pytest.mark.parametrize(
    "invalid",
    ["foreign_source", "foreign_fragment", "missing_part", "foreign_pair", "timeout_source"],
)
async def test_missing_or_foreign_literal_parts_and_timeout_fail_closed(invalid):
    with pytest.raises((CheckProtocolError, TimeoutError)):
        await validate_check_scope(
            PairedProvider(invalid=invalid), original(), GOOD, [SOURCE], "问题"
        )


def test_repeated_text_binds_the_requested_occurrence_with_unchanged_coordinates():
    source = {"source_id": "S1", "content": "窗口服务仍在筹备。窗口服务仍在筹备。"}
    focus = evidence_spans(source)[1]
    value = source_value(source, focus)
    bound = bind_source_parts(SourcePredicateParts.model_validate(value), source, focus)
    part = bound[0]["source_parts"][0]
    assert part["source_start"] == len("窗口服务仍在筹备。")
    assert source["content"][part["source_start"] : part["source_end"]] == part["quote"]


def test_pair_cannot_use_a_predicate_from_a_different_literal_evidence_span():
    focus = evidence_spans(SOURCE)[0]
    registry = bind_source_parts(SourcePredicateParts.model_validate(source_value()), SOURCE, focus)
    claim = PairedPredicateScopeDecision.model_validate(
        {
            "answer_span_id": "A:E1",
            "checks": [
                {
                    "predicate_id": "P1",
                    "source_mode": "planned",
                    "source_voice": "fact",
                    "source_predicate_ids": [registry[0]["source_predicate_id"]],
                    "evidence": [{"source_id": "S1", "span_id": "S1:E2"}],
                    "reason": "错误位置",
                    **{n: "preserved" for n in PREDICATES},
                }
            ],
        }
    )
    with pytest.raises(CheckProtocolError):
        validate_source_pairings(claim.checks, registry)


async def test_request_local_source_cache_never_crosses_document_content():
    from app.atomic_scope import review_atomic_scope

    provider = PairedProvider()
    cache = {}
    focus = answer_spans(GOOD)[0]
    refs = [{"source_id": "S1", **evidence_spans(SOURCE)[0]}]
    args = (provider, GOOD, focus, [focus], [SOURCE], {"S1"}, "问题")
    first, _ = await review_atomic_scope(*args, source_evidence=refs, source_projection_cache=cache)
    second, _ = await review_atomic_scope(
        *args, source_evidence=refs, source_projection_cache=cache
    )
    assert first["scope_model_requests"] == 4 and second["scope_model_requests"] == 3
    assert sum(t == "check_source_predicate_parts" for t, _ in provider.calls) == 1
    assert all(key[1] == SOURCE["content"] for key in cache)
    changed = {**SOURCE, "content": "培训拟下周进行，值班已开展。"}
    changed_refs = [{"source_id": "S1", **evidence_spans(changed)[0]}]
    third, _ = await review_atomic_scope(
        provider,
        GOOD,
        focus,
        [focus],
        [changed],
        {"S1"},
        "问题",
        source_evidence=changed_refs,
        source_projection_cache=cache,
    )
    assert third["scope_model_requests"] == 4
    assert sum(t == "check_source_predicate_parts" for t, _ in provider.calls) == 2
    assert len(cache) == 2 and {key[1] for key in cache} == {SOURCE["content"], changed["content"]}


@pytest.mark.parametrize("invalid", ["undetermined_mode", "undetermined_voice"])
async def test_unknown_independent_source_classification_never_accepts(invalid):
    result, audit = await validate_check_scope(
        PairedProvider(invalid=invalid), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed
    assert not all(all(p[n] for n in PREDICATES) for p in audit["checks"])


async def test_paired_protocol_cannot_run_without_upstream_literal_evidence():
    from app.atomic_scope import review_atomic_scope

    focus = answer_spans(GOOD)[0]
    provider = PairedProvider()
    with pytest.raises(CheckProtocolError):
        await review_atomic_scope(provider, GOOD, focus, [focus], [SOURCE], {"S1"}, "问题")
    assert len(provider.calls) == 1


async def test_upstream_rejection_does_not_project_sources_or_change_history():
    provider = PairedProvider()
    judgment = original().model_copy(update={"passed": False})
    result, audit = await validate_check_scope(provider, judgment, GOOD, [SOURCE], "问题")
    assert result is judgment and audit is None and provider.calls == []


async def test_actual_provider_schemas_ids_and_full_paired_flow():
    fixture = PairedProvider()
    bodies = []

    async def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        payload = json.loads(body["messages"][1]["content"])
        system = body["messages"][0]["content"].split("JSON 对象：", 1)[1]
        wire, _ = json.JSONDecoder().raw_decode(system)
        if "focus_source_span" in payload:
            assert wire["properties"]["source_id"]["enum"] == ["S1"]
            value = await fixture.structured(
                SourcePredicateParts, "check_source_predicate_parts", payload
            )
        elif "source_predicates" in payload:
            assert wire["$defs"]["PairedPredicateScopeAssessment"]["properties"][
                "source_predicate_ids"
            ]["items"]["enum"] == [p["source_predicate_id"] for p in payload["source_predicates"]]
            value = await fixture.structured(
                PairedPredicateScopeDecision, "check_paired_predicate_scope", payload
            )
        else:
            from app.schemas import AnswerRangePredicates

            value = await fixture.structured(
                AnswerRangePredicates, "check_answer_predicate_parts", payload
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": value.model_dump_json()}}]}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    # Preserve the prior paired protocol fixture and its original assertions.
    # Relation-only wire behavior has a separate actual-provider regression.
    provider.supports_check_grounded_relations = False
    try:
        result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "问题")
        assert result.passed and len(bodies) == audit["scope_model_requests"] == 4
        assert all(b["response_format"] == {"type": "json_object"} for b in bodies)
    finally:
        await provider.close()
