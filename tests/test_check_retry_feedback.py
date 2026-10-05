import json

import pytest
from pydantic import ValidationError

from app.check_protocol import bind_check, check_retry_feedback
from app.schemas import CheckDecision, Judgment
from tests.conftest import account, knowledge_base, parse_events, uploaded


def decision(evidence, verdict="supported"):
    return CheckDecision.model_validate(
        {
            "checks": [
                {
                    "answer_span_id": "A:E1",
                    "verdict": verdict,
                    "evidence": evidence,
                    "reason": "分别核对来源；位置匹配不代表事实正确",
                }
            ]
        }
    )


@pytest.mark.parametrize(
    "evidence,expected_missing",
    [
        ([{"source_id": "S1", "span_id": "S1:E1"}], ["S2"]),
        ([{"source_id": "S2", "span_id": "S2:E1"}], ["S1"]),
        ([], ["S1", "S2"]),
        (
            [{"source_id": "S1", "span_id": "S1:E1"}, {"source_id": "S2", "span_id": "S2:E9"}],
            ["S2"],
        ),
    ],
)
def test_missing_actual_source_bindings_are_explained_without_accepting_or_editing(
    evidence, expected_missing
):
    answer = "甲站有6个班组负责值守[S1][S2]。"
    sources = [
        {"source_id": "S1", "content": "甲站有6个班组负责值守。"},
        {"source_id": "S2", "content": "甲站有6个班组负责值守。"},
    ]
    source_copy = json.loads(json.dumps(sources))
    judged = bind_check(
        decision(evidence, "supported" if evidence else "unsupported"), answer, sources
    )
    original = judged.model_dump()
    assert not judged.passed
    feedback = check_retry_feedback(judged)
    details = json.loads(feedback[feedback.index('[{"answer_span_id"') :])
    assert details == [{"answer_span_id": "A:E1", "unverified_cited_source_ids": expected_missing}]
    assert judged.model_dump() == original and sources == source_copy
    assert "这不表示事实已通过" in feedback and "不得制造" in feedback


def test_semantic_rejection_is_preserved_despite_complete_source_bindings():
    answer = "甲站保证每天零故障[S1]。"
    sources = [{"source_id": "S1", "content": "甲站有6个班组负责值守。"}]
    judged = bind_check(
        decision([{"source_id": "S1", "span_id": "S1:E1"}], "unsupported"), answer, sources
    )
    assert not judged.passed
    assert check_retry_feedback(judged) == judged.reason
    assert judged.checks[0].verdict == "unsupported"


def test_nonprotocol_failure_does_not_manufacture_references_or_missing_ids():
    judged = Judgment(passed=False, reason="缺少对应事实，不能保证次日到账")
    assert check_retry_feedback(judged) == judged.reason and not judged.passed


def test_supported_claim_with_no_evidence_still_fails_original_schema():
    with pytest.raises(ValidationError, match="requires literal evidence"):
        decision([])


async def test_missing_evidence_feedback_reaches_regeneration_with_same_retry_limit(
    client, monkeypatch
):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    provider = client.app.state.provider
    structured, stream = provider.structured, provider.stream_answer
    checks, feedback = [], []

    async def missing_then_valid(schema, task, payload):
        if task == "check":
            checks.append(payload["answer"])
            if len(checks) == 1:
                return bind_check(
                    decision([], "unsupported"), payload["answer"], payload["sources"]
                )
        return await structured(schema, task, payload)

    async def capture_revision(query, sources, revision, prompt):
        feedback.append(revision)
        async for token in stream(query, sources, revision, prompt):
            yield token

    monkeypatch.setattr(provider, "structured", missing_then_valid)
    monkeypatch.setattr(provider, "stream_answer", capture_revision)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "出差报销需要哪些材料？", "mode": "agent"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and not done["rejected"]
    assert done["check_retries"] == 1 and len(checks) == len(feedback) == 2
    revision = json.loads(feedback[1])["revision_feedback"]
    assert '"unverified_cited_source_ids": ["S1"]' in revision
    assert any(name == "draft_reset" for name, _ in events)
