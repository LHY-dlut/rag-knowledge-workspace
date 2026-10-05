import asyncio
import json
import time

import pytest

from app import operations


async def test_readiness_probes_both_databases_without_model_calls(client):
    response = await client.get("/api/ready")
    assert response.status_code == 200
    assert response.json()["data"]["databases"] == {"business": True, "vectors": True}


async def test_database_outage_returns_503_without_exposing_credentials(client, monkeypatch):
    original = operations.Tortoise.get_connection

    class Broken:
        async def execute_query(self, query):
            raise RuntimeError("postgres://private:password@host/database")

    monkeypatch.setattr(
        operations.Tortoise,
        "get_connection",
        lambda alias: Broken() if alias == "vectors" else original(alias),
    )
    response = await client.get("/api/ready")
    assert response.status_code == 503
    assert response.json()["data"]["databases"] == {"business": True, "vectors": False}
    assert "password" not in response.text
    assert (await client.get("/api/live")).status_code == 200


async def test_readiness_is_bounded_for_stalled_driver(monkeypatch):
    class Stalled:
        async def execute_query(self, query):
            await asyncio.Event().wait()

    monkeypatch.setattr(operations.Tortoise, "get_connection", lambda _: Stalled())
    started = time.monotonic()
    assert await operations.database_readiness(timeout=0.02) == {
        "business": False,
        "vectors": False,
    }
    assert time.monotonic() - started < 0.5


class Worker:
    def __init__(self, finish):
        self.stopping = asyncio.Event()
        self.active = asyncio.Event()
        self.finish = finish
        self.completed = self.cancelled = False

    async def run(self):
        self.active.set()
        try:
            await self.finish.wait()
            self.completed = True
            await self.stopping.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


async def test_shutdown_allows_active_job_to_finish_and_stops_new_work():
    stop, finish = asyncio.Event(), asyncio.Event()
    worker = Worker(finish)
    task = asyncio.create_task(operations.supervise_workers([worker], stop, 1))
    await worker.active.wait()
    stop.set()
    await worker.stopping.wait()
    assert not task.done() and not worker.completed
    finish.set()
    await task
    assert worker.completed and not worker.cancelled


async def test_shutdown_cancels_hung_job_after_grace_deadline():
    stop = asyncio.Event()
    worker = Worker(asyncio.Event())
    task = asyncio.create_task(operations.supervise_workers([worker], stop, 0.02))
    await worker.active.wait()
    stop.set()
    await asyncio.wait_for(task, 0.5)
    assert worker.cancelled and not worker.completed and worker.stopping.is_set()


async def test_worker_crash_fails_process_and_stops_peer():
    class Crashed:
        stopping = asyncio.Event()

        async def run(self):
            raise RuntimeError("injected worker crash")

    peer = Worker(asyncio.Event())
    with pytest.raises(RuntimeError, match="injected worker crash"):
        await operations.supervise_workers([Crashed(), peer], asyncio.Event(), 0.02)
    assert peer.cancelled and peer.stopping.is_set()


def test_worker_health_rejects_stale_stopped_and_invalid_state(tmp_path):
    path = tmp_path / "health.json"
    assert not operations.worker_health_valid(path)
    operations.write_worker_health(path, True, {"business": True, "vectors": True})
    assert operations.worker_health_valid(path)
    value = json.loads(path.read_text())
    value["time"] -= 60
    path.write_text(json.dumps(value))
    assert not operations.worker_health_valid(path)
    operations.write_worker_health(path, False, {})
    assert not operations.worker_health_valid(path)
    path.write_text("broken-json")
    assert not operations.worker_health_valid(path)
