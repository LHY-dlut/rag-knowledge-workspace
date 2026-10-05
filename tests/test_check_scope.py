import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from app.agent import RAGAgent
from app.check_protocol import answer_citation_manifest, bind_check
from app.check_scope import PREDICATES, validate_check_scope
from app.providers import CheckProtocolError, DashScopeProvider, DemoProvider
from app.schemas import (
    CheckDecision,
    CheckScopeDecision,
    GradeDecision,
    Judgment,
    RetrievalConfig,
    RetrievalResult,
)
from tests.test_providers_parsers import remote_settings

SOURCE = {
    "source_id": "S1",
    "content": "林禾预计，预约将成为服务发展的主要推动力。",
    "document_id": "doc-a",
    "parent_id": "p-a",
    "filename": "a.txt",
    "location": "段1",
    "child_ids": ["c-a"],
}
ANSWER = "预约已成为服务发展的主要推动力。[S1]"


def upstream(answer=ANSWER):
    return bind_check(
        CheckDecision(
            checks=[
                {
                    "answer_span_id": "A:E1",
                    "verdict": "supported",
                    "reason": "构造原check遗漏模态的回归触发",
                    "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                }
            ]
        ),
        answer,
        [SOURCE],
    )


def scope_reply(payload, **changes):
    checks = []
    for item in payload["answer_citation_manifest"]:
        values = {name: True for name in PREDICATES}
        values.update(changes)
        checks.append(
            {
                "answer_span_id": item["answer_span_id"],
                "source_ids": item["cited_source_ids"],
                **values,
                "reason": "逐谓词核对原文语气与答案语气",
            }
        )
    return CheckScopeDecision(checks=checks)


@pytest.mark.parametrize("failed", [None, *PREDICATES])
async def test_each_scope_predicate_can_only_restrict_a_previously_accepted_check(failed):
    class Provider:
        supports_check_scope_binding = True

        async def structured(self, schema, task, payload):
            assert task == "check_scope_binding" and schema is CheckScopeDecision
            assert "prior_reason" not in payload and "passed" not in payload
            assert (
                payload["answer"] == ANSWER
                and payload["sources"][0]["content"] == SOURCE["content"]
            )
            assert (
                payload["verified_literal_evidence"][0]["literal_evidence"][0]["quote"]
                == SOURCE["content"]
            )
            return scope_reply(payload, **({failed: False} if failed else {}))

    original = upstream()
    result, audit = await validate_check_scope(
        Provider(), original, ANSWER, [SOURCE], "资料如何描述服务？"
    )
    assert result.passed is (failed is None) and audit["passed"] is (failed is None)
    assert original.passed and not audit["can_upgrade_rejection"]
    assert (
        result.checks == original.checks and result.evidence_protocol == original.evidence_protocol
    )


@pytest.mark.parametrize("bad", ["true", 1, None])
def test_scope_predicates_are_strict_and_cannot_take_model_passed_override(bad):
    reply = scope_reply({"answer_citation_manifest": answer_citation_manifest(ANSWER)}).model_dump()
    reply["checks"][0]["modality_preserved"] = bad
    with pytest.raises(ValidationError):
        CheckScopeDecision.model_validate(reply)
    reply["checks"][0]["modality_preserved"] = True
    with pytest.raises(ValidationError):
        CheckScopeDecision.model_validate({**reply, "passed": True})


@pytest.mark.parametrize(
    "invalid",
    ["unknown_span", "duplicate_span", "wrong_source", "duplicate_source", "missing_span"],
)
async def test_scope_coverage_and_actual_citation_identity_remain_fail_closed(invalid):
    class Provider:
        supports_check_scope_binding = True

        async def structured(self, schema, task, payload):
            reply = scope_reply(payload).model_dump()
            if invalid == "unknown_span":
                reply["checks"][0]["answer_span_id"] = "A:E999"
            elif invalid == "duplicate_span":
                reply["checks"].append(reply["checks"][0].copy())
            elif invalid == "wrong_source":
                reply["checks"][0]["source_ids"] = ["S999"]
            elif invalid == "duplicate_source":
                reply["checks"][0]["source_ids"] = ["S1", "S1"]
            else:
                # In the focused protocol each request asks for one span. Repeat
                # the first ID to reproduce omission of the actual second span.
                reply["checks"][0]["answer_span_id"] = "A:E1"
            return schema.model_validate(reply)

    answer = ANSWER if invalid != "missing_span" else ANSWER + "\n另一事实待核对。[S1]"
    decision = upstream()
    if invalid == "missing_span":
        decision = bind_check(
            CheckDecision(
                checks=[
                    {
                        "answer_span_id": "A:E1",
                        "verdict": "supported",
                        "reason": "上游构造",
                        "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                    },
                    {
                        "answer_span_id": answer_citation_manifest(answer)[1]["answer_span_id"],
                        "verdict": "supported",
                        "reason": "上游构造",
                        "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                    },
                ]
            ),
            answer,
            [SOURCE],
        )
    assert decision.passed
    with pytest.raises(CheckProtocolError):
        await validate_check_scope(Provider(), decision, answer, [SOURCE], "服务？")


async def test_previously_rejected_or_unverified_demo_never_gets_upgraded_or_additional_model_call():
    class Provider:
        supports_check_scope_binding = True

        async def structured(self, *args):
            pytest.fail("Rejected first check cannot be retried or upgraded by scope review")

    original = Judgment(passed=False, reason="没有事实支持")
    result, audit = await validate_check_scope(Provider(), original, ANSWER, [SOURCE], "服务？")
    assert result is original and audit is None
    demo = DemoProvider(remote_settings())
    result, audit = await validate_check_scope(demo, upstream(), ANSWER, [SOURCE], "服务？")
    assert result.passed and audit is None and not demo.supports_check_scope_binding
    real = DashScopeProvider(remote_settings())
    try:
        assert real.supports_check_scope_binding
    finally:
        await real.close()


async def test_scope_requires_provenance_checked_judgment_before_any_new_call():
    class Provider:
        supports_check_scope_binding = True

        async def structured(self, *args):
            pytest.fail("A bare positive boolean has no checked citation provenance")

    with pytest.raises(CheckProtocolError):
        await validate_check_scope(
            Provider(), Judgment(passed=True, reason="无位置审计"), ANSWER, [SOURCE], "服务？"
        )


@pytest.mark.parametrize("bad", ["non_boolean", "overall_override"])
async def test_scope_response_protocol_failure_stays_distinct_from_network_failure(bad):
    reply = scope_reply({"answer_citation_manifest": answer_citation_manifest(ANSWER)}).model_dump()
    if bad == "non_boolean":
        reply["checks"][0]["modality_preserved"] = "true"
    else:
        reply["passed"] = True
    provider = DashScopeProvider(
        remote_settings(),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(reply)}}]}
            )
        ),
    )
    try:
        with pytest.raises(CheckProtocolError):
            await provider.structured(CheckScopeDecision, "check_scope_binding", {"answer": ANSWER})
    finally:
        await provider.close()


async def test_scope_failure_resets_draft_and_obeys_two_extra_generation_retries(monkeypatch):
    events = []
    monkeypatch.setattr("app.agent.get_stream_writer", lambda: events.append)

    class Provider:
        supports_check_scope_binding = True

        async def structured(self, schema, task, payload):
            if task == "check":
                return upstream(payload["answer"])
            return scope_reply(payload, modality_preserved=False)

    agent = RAGAgent(Provider(), None)
    state = {
        "query": "资料如何描述服务？",
        "draft": ANSWER,
        "mode": "agent",
        "check_retries": 0,
        "retrieval": RetrievalResult(sources=[SOURCE]),
        "kb": SimpleNamespace(config=RetrievalConfig().model_dump()),
        "grade": GradeDecision(
            passed=True,
            support="supported_answer",
            reason="候选范围待生成",
            answer_scope="保留林禾预计的未来语气",
            evidence=[
                {"source_id": "S1", "span_id": "S1:E1", "subject": "林禾", "attribute": "预测"}
            ],
        ),
    }
    for attempt in range(3):
        update = await agent.check({**state, "draft": ANSWER})
        assert not update["check"].passed and update["draft"] == "" and "answer" not in update
        assert not update["citation_protocol"]["scope_review"]["passed"]
        assert agent._after_check({**state, **update}) == (
            "generate" if attempt < 2 else "fallback"
        )
        state.update(update)
    assert (
        state["check_retries"] == 2
        and len(events) == 3
        and all(e["event"] == "draft_reset" for e in events)
    )
