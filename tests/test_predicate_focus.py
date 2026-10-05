import json

import httpx
import pytest

from app.check_protocol import answer_spans
from app.check_scope import PREDICATES, validate_check_scope
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import PredicateScopeDecision
from tests.test_atomic_scope import GOOD, SOURCE, original
from tests.test_predicate_parts import PartsProvider
from tests.test_providers_parsers import remote_settings


class FocusProvider(PartsProvider):
    supports_check_single_predicate_scope = True
    supports_check_context_without_foreign_ids = True

    async def structured(self, schema, task, payload):
        if task != "check_single_predicate_parts_scope":
            return await super().structured(schema, task, payload)
        self.calls.append((task, payload))
        assert schema is PredicateScopeDecision and len(payload["answer_predicates"]) == 1
        assert payload["focus_predicate_id"] == payload["answer_predicates"][0]["predicate_id"]
        pid = payload["focus_predicate_id"]
        row = {
            "predicate_id": pid,
            "source_mode": "planned" if pid == "P1" else "asserted",
            "source_voice": "fact",
            "reason": "仅核验当前关系",
            "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
            **{n: "changed" if n == self.failed else "preserved" for n in PREDICATES},
        }
        if self.invalid == "mode":
            row["source_mode"] = "asserted" if pid == "P1" else "planned"
        if self.invalid == "identity":
            row["predicate_id"] = "P999"
        if self.invalid == "source":
            row["evidence"][0]["span_id"] = "S1:E999"
        if self.invalid == "timeout":
            raise TimeoutError("受控单谓词超时")
        rows = [row]
        if self.invalid == "extra":
            rows.append({**row, "predicate_id": "P999"})
        return schema.model_validate({"answer_span_id": "A:E1", "checks": rows})


async def test_each_source_call_has_one_predicate_and_trace_counts_real_requests():
    provider = FocusProvider()
    result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "私有问题")
    assert result.passed and len(provider.calls) == audit["scope_model_requests"] == 3
    source_calls = provider.calls[1:]
    assert [payload["focus_predicate_id"] for _, payload in source_calls] == ["P1", "P2"]
    assert all(len(payload["answer_predicates"]) == 1 for _, payload in source_calls)
    assert all("mode" not in payload["answer_predicates"][0] for _, payload in source_calls)
    independent = provider.calls[0][1]
    assert "sources" not in independent and "query" not in independent
    assert all("span_id" not in s for s in independent["answer_context_spans"])
    assert independent["answer"] == GOOD
    assert independent["answer_context_spans"] == [
        {k: v for k, v in s.items() if k != "span_id"} for s in answer_spans(GOOD)
    ]


@pytest.mark.parametrize("failed", PREDICATES)
async def test_single_source_request_never_upgrades_a_semantic_negative(failed):
    result, audit = await validate_check_scope(
        FocusProvider(failed=failed), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed and not audit["checks"][0][failed]


async def test_cross_predicate_mode_swap_remains_rejected():
    result, audit = await validate_check_scope(
        FocusProvider(invalid="mode"), original(), GOOD, [SOURCE], "问题"
    )
    assert not result.passed
    assert all(not d["mode_matches_this_predicate"] for d in audit["checks"][0]["predicate_checks"])


@pytest.mark.parametrize("invalid", ["identity", "extra", "source", "timeout"])
async def test_bad_single_result_or_timeout_cannot_publish_a_late_success(invalid):
    with pytest.raises((CheckProtocolError, TimeoutError)):
        await validate_check_scope(
            FocusProvider(invalid=invalid), original(), GOOD, [SOURCE], "问题"
        )


async def test_rejected_upstream_never_dispatches_single_source_checks():
    provider = FocusProvider()
    before = original().model_copy(update={"passed": False})
    result, audit = await validate_check_scope(provider, before, GOOD, [SOURCE], "问题")
    assert result is before and audit is None and provider.calls == []


async def test_actual_provider_single_wire_has_only_current_ids_without_classification_leak():
    captured = []

    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        payload = json.loads(body["messages"][1]["content"])
        if "answer_predicates" not in payload:
            value = {
                "answer_span_id": "A:E1",
                "predicates": [
                    {
                        "predicate_id": "P1",
                        "fragment_ids": ["A:E1:F1"],
                        "mode": "planned",
                        "voice": "fact",
                        "reason": "第一关系",
                    },
                    {
                        "predicate_id": "P2",
                        "fragment_ids": ["A:E1:F2"],
                        "mode": "asserted",
                        "voice": "fact",
                        "reason": "第二关系",
                    },
                ],
            }
        else:
            pid = payload["focus_predicate_id"]
            system = body["messages"][0]["content"].split("JSON 对象：", 1)[1]
            wire, _ = json.JSONDecoder().raw_decode(system)
            assert wire["$defs"]["PredicateScopeAssessment"]["properties"]["predicate_id"][
                "enum"
            ] == [pid]
            assert (
                wire["properties"]["checks"]["minItems"]
                == wire["properties"]["checks"]["maxItems"]
                == 1
            )
            value = {
                "answer_span_id": "A:E1",
                "checks": [
                    {
                        "predicate_id": pid,
                        "source_mode": "planned" if pid == "P1" else "asserted",
                        "source_voice": "fact",
                        "reason": "对应原文",
                        "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                        **{n: "preserved" for n in PREDICATES},
                    }
                ],
            }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    # Keep this compatibility-path fixture and its original assertions. The
    # new source-pair path is exercised through the actual provider separately.
    provider.supports_check_source_predicate_parts = False
    try:
        result, audit = await validate_check_scope(provider, original(), GOOD, [SOURCE], "私有问题")
        assert result.passed and audit["scope_model_requests"] == len(captured) == 3
        independent = json.loads(captured[0]["messages"][1]["content"])
        assert independent["answer_context_spans"][0]["quote"] == answer_spans(GOOD)[0]["quote"]
        assert "span_id" not in independent["answer_context_spans"][0]
        assert all(b["response_format"] == {"type": "json_object"} for b in captured)
        assert (
            "enum" not in PredicateScopeDecision.model_json_schema()["properties"]["answer_span_id"]
        )
    finally:
        await provider.close()
