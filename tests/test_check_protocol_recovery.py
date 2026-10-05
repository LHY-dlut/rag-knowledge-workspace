import json

import httpx
import pytest

from app.providers import CheckProtocolError, DashScopeProvider, ProviderError
from app.schemas import Judgment
from tests.conftest import account, knowledge_base, parse_events, uploaded
from tests.test_check_evidence_binding import ANSWER, SOURCE, decision
from tests.test_providers_parsers import remote_settings


@pytest.mark.parametrize("invalid", ["not JSON", "unknown_verdict", "supported_without_evidence"])
async def test_strict_check_schema_error_remains_a_provider_error(invalid):
    data = decision()
    if invalid == "unknown_verdict":
        data["checks"][0]["verdict"] = "non_factual"
    if invalid == "supported_without_evidence":
        data["checks"][0]["evidence"] = []
    raw = invalid if invalid == "not JSON" else json.dumps(data)

    def response(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(response))
    try:
        with pytest.raises(CheckProtocolError) as raised:
            await provider.structured(
                Judgment, "check", {"query": "开会条件？", "answer": ANSWER, "sources": [SOURCE]}
            )
        assert isinstance(raised.value, ProviderError)
    finally:
        await provider.close()


@pytest.mark.parametrize("mode,checks", [("agent", 3), ("rag", 1)])
async def test_invalid_checks_reset_drafts_and_respect_existing_retry_limit(
    client, monkeypatch, mode, checks
):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    original = client.app.state.provider.structured
    attempts = []

    async def invalid_check(schema, task, payload):
        if task == "check":
            attempts.append(payload["answer"])
            raise CheckProtocolError("received malformed check")
        return await original(schema, task, payload)

    monkeypatch.setattr(client.app.state.provider, "structured", invalid_check)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "出差报销需要哪些材料？", "mode": mode},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and done["rejected"]
    assert done["check_retries"] == checks - 1 and len(attempts) == checks
    assert not any(name == "token" and data["text"] in attempts for name, data in events)
    assert all(draft not in done["answer"] for draft in attempts)
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    check_steps = [s for s in trace["steps"] if s["node"] == "check"]
    assert len(check_steps) == checks
    assert all(s["output_summary"]["check"]["passed"] is False for s in check_steps)
    assert all(
        s["output_summary"]["citation_protocol"]["check_response"]["failure_type"]
        == "check_response_schema_invalid"
        for s in check_steps
    )
    conversation_id = next(d["conversation_id"] for n, d in events if n == "meta")
    messages = (
        await client.get(f"/api/conversations/{conversation_id}/messages", headers=headers)
    ).json()["data"]
    assert all(draft not in message["content"] for draft in attempts for message in messages)


async def test_transient_invalid_check_requires_a_new_valid_check_before_publication(
    client, monkeypatch
):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    original = client.app.state.provider.structured
    attempts = []

    async def once_invalid(schema, task, payload):
        if task == "check":
            attempts.append(payload["answer"])
            if len(attempts) == 1:
                raise CheckProtocolError("bad protocol")
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
    assert done["check_retries"] == 1 and len(attempts) == 2
    names = [n for n, _ in events]
    assert names.index("draft_reset") < names.index("token")
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    steps = [s for s in trace["steps"] if s["node"] == "check"]
    assert [s["output_summary"]["check"]["passed"] for s in steps] == [False, True]


async def test_network_provider_error_is_not_relabelled_as_check_protocol_failure(
    client, monkeypatch
):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    original = client.app.state.provider.structured

    async def unavailable(schema, task, payload):
        if task == "check":
            raise ProviderError("HTTP 502")
        return await original(schema, task, payload)

    monkeypatch.setattr(client.app.state.provider, "structured", unavailable)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "出差报销需要哪些材料？", "mode": "agent"},
    )
    events = parse_events(response)
    assert events[-1][0] == "error"
    assert not any(n == "done" for n, _ in events)
