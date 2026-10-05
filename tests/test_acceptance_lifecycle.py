import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.chunking import ParsedUnit, parent_child_chunks
from app.models import AgentRun, ChatMessage, Conversation, Document, IngestJob, RunEvent
from app.providers import ProviderError
from app.schemas import RetrievalConfig
from app.vector_models import ChunkVector
from tests.conftest import account, knowledge_base, parse_events, uploaded


@pytest.mark.parametrize("strategy", ["recursive", "recursive_short", "parent_child"])
@pytest.mark.parametrize("mode", ["general", "laws", "qa"])
def test_structural_modes_cover_offsets_and_budget(strategy, mode):
    text = (
        "说明\n第一条 安全规则。\n第二条 "
        + "内容。" * 240
        + "\n问：如何办理？\n答：持证申请。\nQ2:条件？\nA:审核通过。"
    )
    config = RetrievalConfig(chunk_strategy=strategy, structure_mode=mode)
    parents = parent_child_chunks("doc", [ParsedUnit(text, {"page": 3})], config)
    coverage = set()
    for parent in parents:
        start, end = parent.metadata["start"], parent.metadata["end"]
        assert text[start:end] == parent.content
        coverage.update(range(start, end))
        assert parent.metadata["page"] == 3 and parent.metadata["structure_mode"] == mode
        for _, content, a, b in parent.children:
            assert content == parent.content[a:b]
    assert coverage == set(range(len(text)))
    if mode == "laws":
        assert any(p.content.startswith("第二条") for p in parents)
    if mode == "qa":
        assert any(p.content.startswith("问：") and "答：" in p.content for p in parents)


async def submit(client, headers, kb, **kwargs):
    response = await client.post(
        "/api/chat/runs",
        headers=headers,
        json={"kb_id": kb, "query": "报销申请多久提交？", **kwargs},
    )
    assert response.status_code == 202, response.text
    return response.json()["data"]


async def wait_terminal(run_id):
    for _ in range(300):
        run = await AgentRun.get(id=run_id)
        if run.status in {"completed", "failed", "cancelled"}:
            return run
        await asyncio.sleep(0.02)
    raise AssertionError("Run never reached terminal state")


async def test_same_conversation_limit_cancel_no_late_history(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    original = client.app.state.provider.stream_answer
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        async for token in original(*args, **kwargs):
            yield token

    monkeypatch.setattr(client.app.state.provider, "stream_answer", delayed)
    data = await submit(client, headers, kb)
    await asyncio.wait_for(entered.wait(), 5)
    duplicate = await client.post(
        "/api/chat/runs",
        headers=headers,
        json={"kb_id": kb, "query": "再问", "conversation_id": data["conversation_id"]},
    )
    assert duplicate.status_code == 409
    await client.post(f"/api/runs/{data['run_id']}/cancel", headers=headers)
    release.set()
    await asyncio.sleep(0.2)
    assert (await AgentRun.get(id=data["run_id"])).status == "cancelled"
    assert not await ChatMessage.filter(run_id=data["run_id"], accepted=True).exists()
    assert not await RunEvent.filter(run_id=data["run_id"], name__in=["token", "done"]).exists()
    assert (await Conversation.get(id=data["conversation_id"])).active_run_id == ""


async def test_replay_sequence_resume_and_terminal_compensation(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    data = await submit(client, headers, kb)
    run = await wait_terminal(data["run_id"])
    assert run.status == "completed"
    response = await client.get(f"/api/runs/{run.id}/events", headers=headers)
    events = parse_events(response)
    assert events[0][0] == "meta" and events[-1][0] == "done"
    ids = [
        int(line.rsplit(":", 1)[1]) for line in response.text.splitlines() if line.startswith("id:")
    ]
    assert ids == list(range(1, run.event_sequence + 1))
    resumed = await client.get(
        f"/api/runs/{run.id}/events", headers={**headers, "Last-Event-ID": f"{run.id}:3"}
    )
    assert parse_events(resumed) == events[3:]
    assert (
        await client.get(f"/api/runs/{run.id}/events?after={run.event_sequence}", headers=headers)
    ).text == ""
    assert (await client.get(f"/api/runs/{run.id}", headers=headers)).json()["data"][
        "answer"
    ] == events[-1][1]["answer"]
    assert (
        await client.get(f"/api/runs/{run.id}/events?after=99999", headers=headers)
    ).status_code == 400


@pytest.mark.parametrize("kind", ["lease", "conversation", "timeout"])
async def test_lost_lease_and_timeout_never_publish(client, monkeypatch, kind):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    original = client.app.state.provider.stream_answer

    async def invalidated(*args, **kwargs):
        if kind == "timeout":
            raise TimeoutError()
        if kind == "lease":
            await AgentRun.filter(status="running").update(
                lease_until=datetime.now(UTC) - timedelta(seconds=1)
            )
        else:
            await Conversation.all().update(
                active_run_until=datetime.now(UTC) - timedelta(seconds=1)
            )
        async for token in original(*args, **kwargs):
            yield token

    monkeypatch.setattr(client.app.state.provider, "stream_answer", invalidated)
    run = await wait_terminal((await submit(client, headers, kb))["run_id"])
    assert run.status == "failed" and not run.answer
    assert not await ChatMessage.filter(run_id=run.id, accepted=True).exists()
    assert not await RunEvent.filter(run_id=run.id, name="token").exists()


async def test_business_publication_failure_hides_committed_vectors_and_retry(client, monkeypatch):
    from app.models import ParentChunk

    headers = await account(client)
    kb = await knowledge_base(client, headers)
    original = ParentChunk.bulk_create

    async def fail(*args, **kwargs):
        raise RuntimeError("Injected business publication failure")

    response = await client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=headers,
        files={"file": ("a.txt", "报销7天内提交".encode())},
    )
    doc = response.json()["data"]["document"]["id"]
    monkeypatch.setattr(ParentChunk, "bulk_create", fail)
    await client.app.state.worker.tick()
    assert await ChunkVector.filter(doc_id=doc).exists()
    assert (await Document.get(id=doc)).status == "failed"
    result = await client.post(
        "/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销"}
    )
    assert result.json()["data"]["sources"] == []
    monkeypatch.setattr(ParentChunk, "bulk_create", original)
    await client.post(f"/api/documents/{doc}/reindex", headers=headers)
    await client.app.state.worker.tick()
    assert (await Document.get(id=doc)).status == "ready"


async def test_two_ingest_workers_claim_once_and_heartbeat(client, monkeypatch):
    from app.ingestion import IngestionWorker

    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=headers,
        files={"file": ("a.txt", "报销7天内提交".encode())},
    )
    job_id = response.json()["data"]["job_id"]
    entered, release = asyncio.Event(), asyncio.Event()
    original = client.app.state.provider.embed
    calls = 0

    async def delayed(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(client.app.state.provider, "embed", delayed)
    worker = client.app.state.worker
    other = IngestionWorker(worker.service, worker.settings)
    task = asyncio.create_task(worker.tick())
    await asyncio.wait_for(entered.wait(), 5)
    assert not await other.tick()
    assert calls == 1
    # Speed only the test heartbeat interval; production settings remain bounded.
    worker.settings.worker_lease_seconds = 3
    heartbeat = asyncio.create_task(worker._heartbeat(job_id))
    await asyncio.sleep(1.1)
    current = await IngestJob.get(id=job_id)
    assert current.lease_until > datetime.now(UTC)
    # Restore the normal lease through the actual heartbeat before cancelling
    # this accelerated helper. Restoring settings alone leaves a 3s DB lease,
    # which legitimately expires during unrelated cold jieba initialization.
    worker.settings.worker_lease_seconds = 600
    deadline = asyncio.get_running_loop().time() + 5
    while True:
        current = await IngestJob.get(id=job_id)
        if current.lease_until > datetime.now(UTC) + timedelta(seconds=300):
            break
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.05)
    heartbeat.cancel()
    await asyncio.gather(heartbeat, return_exceptions=True)
    release.set()
    await task
    assert (await IngestJob.get(id=job_id)).status == "done"


async def test_evaluation_missing_judge_and_independent_labels(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    doc_id = await uploaded(client, headers, kb)
    child = await ChunkVector.filter(doc_id=doc_id).first()
    case = {
        "question": "报销申请多久提交？",
        "reference_answer": "7天内提交",
        "reference_facts": ["7天内提交报销申请"],
    }
    body = {"kb_id": kb, "cases": [case]}
    result = (await client.post("/api/evaluations", headers=headers, json=body)).json()["data"]
    assert result["metrics"]["retrieval_f1"] is None
    case.update(relevant_child_ids=[child.id], annotation_method="human")
    result = (await client.post("/api/evaluations", headers=headers, json=body)).json()["data"]
    assert result["metrics"]["retrieval_recall"] == 1
    original = client.app.state.provider.structured

    async def failed_judge(schema, task, payload):
        if task == "evaluation":
            raise ProviderError("Judge unavailable")
        return await original(schema, task, payload)

    monkeypatch.setattr(client.app.state.provider, "structured", failed_judge)
    result = (await client.post("/api/evaluations", headers=headers, json=body)).json()["data"]
    assert result["status"] == "partial_missing"
    assert all(
        result["metrics"][k] is None
        for k in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
    )
    assert result["results"][0]["status"] == "missing"
