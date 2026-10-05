import json

import httpx
import pytest

from app.providers import DashScopeProvider, GradeProtocolError, ProviderError
from app.schemas import GradeDecision
from tests.conftest import account, knowledge_base, parse_events, uploaded
from tests.test_providers_parsers import remote_settings


@pytest.mark.parametrize("invalid", ["not_json", "contradictory_classification", "no_evidence"])
async def test_invalid_grade_is_strictly_rejected_without_coercing_support(invalid):
    result = {
        "passed": False,
        "support": "supported_limitation",
        "reason": "资料明确写未规定",
        "answer_scope": "本资料未规定时限",
        "evidence": [
            {"source_id": "S1", "span_id": "S1:E1", "subject": "维修", "attribute": "时限"}
        ],
    }
    if invalid == "no_evidence":
        result.update(passed=True, evidence=[])
    raw = "not JSON" if invalid == "not_json" else json.dumps(result, ensure_ascii=False)
    provider = DashScopeProvider(
        remote_settings(),
        httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})
        ),
    )
    try:
        with pytest.raises(GradeProtocolError) as error:
            await provider.structured(
                GradeDecision,
                "grade",
                {
                    "query": "维修时限？",
                    "sources": [{"source_id": "S1", "content": "维修规程未规定完成时限。"}],
                },
            )
        assert isinstance(error.value, ProviderError)
    finally:
        await provider.close()


@pytest.mark.parametrize("mode,attempts", [("agent", 4), ("rag", 1)])
async def test_invalid_grade_never_generates_and_respects_existing_retry_limit(
    client, monkeypatch, mode, attempts
):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    original = client.app.state.provider.structured
    payloads = []
    generated = []

    async def invalid_grade(schema, task, payload):
        if task == "grade":
            payloads.append(payload)
            raise GradeProtocolError("received contradictory grade")
        return await original(schema, task, payload)

    async def forbidden_generate(*args, **kwargs):
        generated.append(True)
        yield "不应发布"

    monkeypatch.setattr(client.app.state.provider, "structured", invalid_grade)
    monkeypatch.setattr(client.app.state.provider, "stream_answer", forbidden_generate)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "报销材料有哪些？", "mode": mode},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and done["rejected"]
    assert done["grade_retries"] == attempts - 1 and len(payloads) == attempts and not generated
    assert all(
        p["protocol_feedback"]["failure_type"] == "grade_response_schema_invalid"
        for p in payloads[1:]
    )
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    grades = [s["output_summary"] for s in trace["steps"] if s["node"] == "grade"]
    assert len(grades) == attempts
    assert all(
        not s["grade"]["passed"]
        and s["grade_protocol"]["failure_types"] == ["grade_response_schema_invalid"]
        for s in grades
    )
    assert not any(n == "draft_token" for n, _ in events)
    assert "".join(d["text"] for n, d in events if n == "token") == done["answer"]
    assert "不应发布" not in done["answer"]


async def test_transient_invalid_grade_requires_fresh_valid_evidence_before_generation(
    client, monkeypatch
):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    original = client.app.state.provider.structured
    attempts = []

    async def once_invalid(schema, task, payload):
        if task == "grade":
            attempts.append(payload)
            if len(attempts) == 1:
                raise GradeProtocolError("bad protocol")
        return await original(schema, task, payload)

    monkeypatch.setattr(client.app.state.provider, "structured", once_invalid)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "出差报销需要哪些材料？", "mode": "agent"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and not done["rejected"]
    assert done["grade_retries"] == 1 and len(attempts) == 2
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    grades = [s["output_summary"] for s in trace["steps"] if s["node"] == "grade"]
    assert [g["grade"]["passed"] for g in grades] == [False, True]
    assert grades[1]["grade"]["evidence"] and grades[1]["feedback"] == ""


async def test_grade_network_failure_is_not_relabelled_as_protocol_failure(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    original = client.app.state.provider.structured

    async def network_error(schema, task, payload):
        if task == "grade":
            raise ProviderError("HTTP 502")
        return await original(schema, task, payload)

    monkeypatch.setattr(client.app.state.provider, "structured", network_error)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "报销材料有哪些？", "mode": "agent"},
    )
    events = parse_events(response)
    assert events[-1][0] == "error" and not any(n == "done" for n, _ in events)
