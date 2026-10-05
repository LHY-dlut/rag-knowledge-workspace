import json

import httpx
import pytest

from app.models import Document, IngestJob, KnowledgeBase, ParentChunk
from app.providers import DashScopeProvider, ProviderError
from tests.conftest import account, knowledge_base, parse_events, uploaded
from tests.test_providers_parsers import remote_settings


async def test_cross_user_all_read_write_delete_search_trace_endpoints(client):
    first = await account(client, "owner")
    kb = await knowledge_base(client, first)
    doc = await uploaded(client, first, kb)
    parent = await ParentChunk.filter(doc_id=doc).first()
    job = await IngestJob.filter(doc_id=doc).first()
    result = await client.post(
        "/api/chat/stream", headers=first, json={"kb_id": kb, "query": "报销申请多久提交？"}
    )
    events = parse_events(result)
    run, conversation = events[0][1]["run_id"], events[0][1]["conversation_id"]
    second = await account(client, "attacker")
    requests = [
        ("GET", f"/api/knowledge-bases/{kb}/documents", {}),
        ("PUT", f"/api/knowledge-bases/{kb}/config", {"json": {}}),
        ("GET", f"/api/documents/{doc}/chunks", {}),
        ("DELETE", f"/api/documents/{doc}", {}),
        ("POST", f"/api/documents/{doc}/reindex", {}),
        ("PUT", f"/api/documents/{doc}/file", {"files": {"file": ("attack.txt", b"attack")}}),
        ("PUT", f"/api/chunks/{parent.id}", {"json": {"content": "attack"}}),
        ("GET", f"/api/jobs/{job.id}", {}),
        ("POST", "/api/retrieval/debug", {"json": {"kb_id": kb, "query": "secret"}}),
        ("GET", f"/api/runs/{run}", {}),
        ("GET", f"/api/runs/{run}/events", {}),
        ("POST", f"/api/runs/{run}/cancel", {}),
        ("GET", f"/api/conversations/{conversation}/messages", {}),
        ("GET", f"/api/conversations?kb_id={kb}", {}),
    ]
    for method, url, kwargs in requests:
        response = await client.request(method, url, headers=second, **kwargs)
        assert response.status_code == 404, (method, url, response.text)
    assert (await Document.get(id=doc)).status == "ready"
    assert "attack" not in (await ParentChunk.get(id=parent.id)).content
    assert (await KnowledgeBase.get(id=kb)).owner_id != "attacker"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_rerank_failure_never_fabricates_scores(status):
    calls = []

    async def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(status, json={"error": "unavailable"})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError):
            await provider.rerank("question", ["a", "b"])
        assert len(calls) == 3
    finally:
        await provider.close()


async def test_rerank_threshold_and_original_scores(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers, {"rerank": True, "rerank_threshold": 0.5})
    await uploaded(client, headers, kb)

    async def ranking(query, docs):
        return [(i, 0.4321) for i in range(len(docs))]

    monkeypatch.setattr(client.app.state.provider, "rerank", ranking)
    response = await client.post(
        "/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销"}
    )
    assert response.json()["data"]["sources"] == []
    await client.put(
        f"/api/knowledge-bases/{kb}/config",
        headers=headers,
        json={"rerank": True, "rerank_threshold": 0.4},
    )
    response = await client.post(
        "/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销"}
    )
    assert response.json()["data"]["candidates"][0]["rerank_score"] == 0.4321
