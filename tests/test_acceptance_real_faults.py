import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tortoise import Tortoise

from app.chat_jobs import LeaseLost
from app.models import AgentRun, ChatMessage, Document, RunEvent
from app.vector_models import ChunkVector
from tests.conftest import account, knowledge_base, uploaded
from tests.test_acceptance_lock_waits import lock_row


@pytest.mark.integration
async def test_mysql_rejects_publish_after_pg_commit(dual_client):
    client = dual_client
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=headers,
        files={"file": ("policy.txt", "报销申请在7天内提交。".encode())},
    )
    doc = response.json()["data"]["document"]["id"]
    conn = Tortoise.get_connection("business")
    constraint = "acceptance_reject_" + uuid4().hex[:12]
    await conn.execute_script(
        f"ALTER TABLE parentchunk ADD CONSTRAINT `{constraint}` CHECK (doc_id <> '{doc}')"
    )
    try:
        await client.app.state.worker.tick()
        assert await ChunkVector.filter(doc_id=doc).exists(), (
            "PG transaction must already be committed"
        )
        assert (await Document.get(id=doc)).status == "failed"
        result = (
            await client.post(
                "/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销"}
            )
        ).json()["data"]
        assert result["sources"] == [] and result["candidates"] == []
    finally:
        await conn.execute_script(f"ALTER TABLE parentchunk DROP CHECK `{constraint}`")
    response = await client.post(f"/api/documents/{doc}/reindex", headers=headers)
    assert response.status_code == 202
    await client.app.state.worker.tick()
    assert (await Document.get(id=doc)).status == "ready"


@pytest.mark.integration
async def test_chat_final_lock_wait_past_lease_cannot_publish(dual_client):
    client = dual_client
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    doc = await uploaded(client, headers, kb)
    response = await client.post(
        "/api/chat/runs", headers=headers, json={"kb_id": kb, "query": "报销需要几天内提交？"}
    )
    run = await AgentRun.get(id=response.json()["data"]["run_id"])
    worker = client.app.state.chat_worker
    run.status, run.lease_owner = "running", worker.id
    run.lease_until = datetime.now(UTC) + timedelta(seconds=1.5)
    await run.save()
    blocker = await lock_row(worker.settings, Document, doc)
    task = asyncio.create_task(worker._execute(run))
    try:
        await asyncio.sleep(2.1)
        assert not task.done(), "Execution must be waiting on the held evidence row"
        await blocker.commit()
        with pytest.raises(LeaseLost):
            await asyncio.wait_for(task, 5)
        assert not await ChatMessage.filter(run_id=run.id, accepted=True).exists()
        assert not await RunEvent.filter(run_id=run.id, name__in=["token", "done"]).exists()
    finally:
        blocker.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
