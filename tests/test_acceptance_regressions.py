"""Acceptance regressions: strict provider data and stale executors."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.models import Document, IngestJob
from app.providers import DashScopeProvider, ProviderError
from app.vector_models import ChunkVector, validate_vector
from tests.conftest import account, knowledge_base, uploaded
from tests.test_providers_parsers import remote_settings


@pytest.mark.parametrize("value", [True, False, "1", None, 1e308])
def test_embedding_rejects_invalid_scalars(value):
    with pytest.raises(ValueError):
        validate_vector([value] + [1.0] * 1023)


@pytest.mark.parametrize("index", [False, 0.0, "0"])
async def test_embedding_requires_integer_index(index):
    async def handle(request):
        return httpx.Response(
            200, json={"output": {"embeddings": [{"text_index": index, "embedding": [1.0] * 1024}]}}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError):
            await provider.embed(["sample"])
    finally:
        await provider.close()


@pytest.mark.parametrize(
    "results",
    [
        [{"index": False, "relevance_score": 0.3}],
        [{"index": 0, "relevance_score": True}],
        [{"index": 0.0, "relevance_score": 0.3}],
        [{"index": 0, "relevance_score": "0.3"}],
        [],
    ],
)
async def test_rerank_rejects_malformed_or_incomplete(results):
    async def handle(request):
        return httpx.Response(200, json={"output": {"results": results}})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        with pytest.raises(ProviderError):
            await provider.rerank("question", ["document"])
    finally:
        await provider.close()


async def test_rerank_batch_mapping_preserves_raw_scores():
    calls = []

    async def handle(request):
        documents = json.loads(request.content)["input"]["documents"]
        calls.append(documents)
        assert len(documents) <= 100
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": i, "relevance_score": int(d) / 300}
                        for i, d in reversed(list(enumerate(documents)))
                    ]
                }
            },
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        result = await provider.rerank("question", [str(i) for i in range(205)])
        assert len(calls) == 3
        assert result == [(i, i / 300) for i in reversed(range(205))]
    finally:
        await provider.close()


async def test_expired_ingest_executor_cannot_overwrite_new_index(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    doc_id = await uploaded(client, headers, kb)
    response = await client.post(f"/api/documents/{doc_id}/reindex", headers=headers)
    job_id = response.json()["data"]["job_id"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = client.app.state.provider.embed
    calls = 0

    async def barrier(texts, text_type="document"):
        nonlocal calls
        calls += 1
        result = await original(texts, text_type)
        if calls == 1:
            entered.set()
            await release.wait()
            return [[9.0] + [0.0] * 1023 for _ in result]
        return result

    monkeypatch.setattr(client.app.state.provider, "embed", barrier)
    old = asyncio.create_task(client.app.state.worker.tick())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await IngestJob.filter(id=job_id).update(
            lease_until=datetime.now(UTC) - timedelta(seconds=1)
        )
        from app.ingestion import IngestionWorker

        replacement = IngestionWorker(client.app.state.worker.service, client.app.state.settings)
        assert await replacement.tick()
        before = await ChunkVector.filter(doc_id=doc_id).values("id", "embedding")
        revision = (await Document.get(id=doc_id)).index_revision
        release.set()
        await old
        assert await ChunkVector.filter(doc_id=doc_id).values("id", "embedding") == before
        assert (await Document.get(id=doc_id)).index_revision == revision
        assert (await Document.get(id=doc_id)).status == "ready"
    finally:
        release.set()
        await asyncio.gather(old, return_exceptions=True)


async def test_chat_submission_is_durable_and_cancel_is_explicit(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        "/api/chat/runs", headers=headers, json={"kb_id": kb, "query": "报销"}
    )
    assert response.status_code == 202
    run = response.json()["data"]
    other = await account(client, "other")
    assert (await client.get(f"/api/runs/{run['run_id']}/events", headers=other)).status_code == 404
    assert (
        await client.post(f"/api/runs/{run['run_id']}/cancel", headers=other)
    ).status_code == 404
    assert (
        await client.post(f"/api/runs/{run['run_id']}/cancel", headers=headers)
    ).status_code == 200
    assert (await client.get(f"/api/runs/{run['run_id']}", headers=headers)).json()["data"][
        "status"
    ] == "cancelled"
