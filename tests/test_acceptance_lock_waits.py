"""Real MySQL row locks: waiting past lease expiry cannot resurrect an owner."""

import asyncio
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import asyncmy
import pytest

from app.models import AgentRun, ChatMessage, Conversation, IngestJob, RunEvent
from tests.conftest import account, knowledge_base, uploaded


async def lock_row(settings, model, row_id):
    url = urlparse(settings.business_db_url)
    conn = await asyncmy.connect(
        host=url.hostname,
        port=url.port,
        user=url.username,
        password=url.password,
        db=url.path.lstrip("/"),
        autocommit=False,
    )
    async with conn.cursor() as cursor:
        await cursor.execute(
            f"SELECT id FROM `{model._meta.db_table}` WHERE id=%s FOR UPDATE", (row_id,)
        )
    return conn


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["ingest", "chat"])
async def test_lock_wait_cannot_renew_expired_lease(dual_client, kind):
    client = dual_client
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    doc = await uploaded(client, headers, kb)
    expires = datetime.now(UTC) + timedelta(seconds=1.5)
    execution = None
    if kind == "ingest":
        worker = client.app.state.worker
        row = await IngestJob.filter(doc_id=doc).first()
        row.status, row.lease_owner, row.lease_until = "processing", worker.id, expires
        await row.save()
        model = IngestJob
    else:
        worker = client.app.state.chat_worker
        convo = await Conversation.create(
            owner_id=(await IngestJob.filter(doc_id=doc).first()).owner_id,
            kb_id=kb,
            title="lock wait",
        )
        row = await AgentRun.create(
            owner_id=convo.owner_id,
            kb_id=kb,
            conversation_id=convo.id,
            query="test",
            status="running",
            lease_owner=worker.id,
            lease_until=expires,
        )
        convo.active_run_id = row.id
        convo.active_run_until = datetime.now(UTC) + timedelta(seconds=30)
        await convo.save()
        model = Conversation
        execution = asyncio.create_task(asyncio.sleep(30))
    worker.settings.worker_lease_seconds = 3
    blocker = await lock_row(worker.settings, model, row.id if kind == "ingest" else convo.id)
    heartbeat = asyncio.create_task(
        worker._heartbeat(row.id) if kind == "ingest" else worker._heartbeat(row, execution)
    )
    try:
        await asyncio.sleep(2.1)  # heartbeat starts at 1s, original lease expires at 1.5s
        await blocker.commit()
        await asyncio.sleep(0.15)
        await row.refresh_from_db()
        assert row.lease_until <= datetime.now(UTC), "expired lease was resurrected after lock wait"
        assert row.lease_until == expires, "expired lease was mutated"
        if execution:
            assert execution.cancelled()
    finally:
        blocker.close()
        heartbeat.cancel()
        if execution:
            execution.cancel()
        await asyncio.gather(heartbeat, *([execution] if execution else []), return_exceptions=True)


@pytest.mark.integration
async def test_real_cancel_during_generation_never_publishes(dual_client, monkeypatch):
    client = dual_client
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    entered, release = asyncio.Event(), asyncio.Event()
    original = client.app.state.provider.stream_answer

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        async for token in original(*args, **kwargs):
            yield token

    monkeypatch.setattr(client.app.state.provider, "stream_answer", delayed)
    response = await client.post(
        "/api/chat/runs", headers=headers, json={"kb_id": kb, "query": "报销申请多久提交？"}
    )
    run_id = response.json()["data"]["run_id"]
    task = asyncio.create_task(client.app.state.chat_worker.tick())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        response = await client.post(f"/api/runs/{run_id}/cancel", headers=headers)
        assert response.status_code == 200
        release.set()
        await asyncio.wait_for(task, 10)
        assert (await AgentRun.get(id=run_id)).status == "cancelled"
        assert not await ChatMessage.filter(run_id=run_id, accepted=True).exists()
        assert not await RunEvent.filter(run_id=run_id, name__in=["token", "done"]).exists()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
