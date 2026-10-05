import asyncio
import os
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest

from app.artifact_cleanup import quarantine_orphans
from app.artifact_storage import (
    PendingFile,
    artifact_directory,
    artifact_transaction,
    finish_io,
    publication_lock,
)
from app.artifacts import file_path, persist_artifact
from app.models import AgentRun, AgentStep, GeneratedArtifact
from tests.test_artifacts import completed, payload


async def test_outer_rollback_removes_only_new_file_preserving_committed_artifact(client):
    _, _, _, done, _ = await completed(client)
    settings = client.app.state.settings
    original = done["artifacts"][0]
    existing = file_path(settings, original["id"], "chart")
    content = existing.read_bytes()
    run = await AgentRun.get(id=done["run_id"])
    created = None
    with pytest.raises(RuntimeError, match="publication fault"):
        async with artifact_transaction(settings) as (connection, pending):
            created, _ = await persist_artifact(
                run, "report", payload(kind="report"), settings, connection, pending
            )
            assert file_path(settings, created.id, "report").is_file()
            raise RuntimeError("publication fault after database insert")
    assert not await GeneratedArtifact.filter(id=created.id).exists()
    assert not file_path(settings, created.id, "report").exists()
    assert existing.read_bytes() == content


async def test_uncertain_commit_cannot_delete_a_committed_file(client, monkeypatch):
    import app.artifact_storage as storage

    _, _, _, done, _ = await completed(client)
    settings = client.app.state.settings
    item = await GeneratedArtifact.get(id=done["artifacts"][0]["id"])
    path = file_path(settings, item.id, item.kind)
    pending = storage.PendingFile(item.id, path, owned=True)
    await storage.rollback_files([pending])
    assert path.is_file()
    original = GeneratedArtifact.filter

    def unavailable(*args, **kwargs):
        raise ConnectionError("simulated uncertain acknowledgement")

    monkeypatch.setattr(GeneratedArtifact, "filter", unavailable)
    await storage.rollback_files([pending])
    assert path.is_file()
    monkeypatch.setattr(GeneratedArtifact, "filter", original)


async def test_partial_write_is_cleaned_after_transaction_failure(client, monkeypatch):
    import app.artifacts as artifacts

    _, _, _, done, _ = await completed(client)
    settings = client.app.state.settings
    run = await AgentRun.get(id=done["run_id"])
    before = set(artifact_directory(settings).glob("*.html"))

    def failed_flush(*args):
        raise OSError("synthetic full disk")

    monkeypatch.setattr(artifacts.os, "fsync", failed_flush)
    with pytest.raises(OSError, match="full disk"):
        async with artifact_transaction(settings) as (connection, pending):
            await persist_artifact(
                run, "report", payload(kind="report"), settings, connection, pending
            )
    assert set(artifact_directory(settings).glob("*.html")) == before
    assert await GeneratedArtifact.filter(run_id=run.id).count() == 1


async def test_cancelled_writer_settles_before_rollback_and_lock_release(client):
    settings = client.app.state.settings
    writing, release = threading.Event(), threading.Event()
    path = file_path(settings, str(uuid4()), "report")

    async def publish():
        async with artifact_transaction(settings) as (_, pending):
            entry = PendingFile(path.stem, path)
            pending.append(entry)

            def write():
                with path.open("xb") as handle:
                    entry.owned = True
                    handle.write(b"first")
                    handle.flush()
                    writing.set()
                    assert release.wait(5)
                    handle.write(b"last")

            await finish_io(write)

    task = asyncio.create_task(publish())
    assert await asyncio.to_thread(writing.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done() and path.exists()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not path.exists()
    async with publication_lock(settings, timeout_seconds=0.2):
        pass


async def test_cleanup_database_failure_preserves_orphans(client, monkeypatch):
    settings = client.app.state.settings
    base = artifact_directory(settings)
    base.mkdir(parents=True)
    path = base / (str(uuid4()) + ".html")
    path.write_text("recoverable", encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise ConnectionError("simulated database outage")

    monkeypatch.setattr(GeneratedArtifact, "filter", unavailable)
    with pytest.raises(ConnectionError):
        await quarantine_orphans(settings, apply=True, min_age_seconds=0)
    assert path.exists() and not (base / ".quarantine").exists()


async def test_cleanup_dry_run_and_reversible_quarantine_preserve_live_recent_unknown(client):
    _, _, _, done, _ = await completed(client)
    settings = client.app.state.settings
    base = artifact_directory(settings)
    live = file_path(settings, done["artifacts"][0]["id"], "chart")
    orphan = base / (str(uuid4()) + ".html")
    orphan.write_text("old orphan", encoding="utf-8")
    recent = base / (str(uuid4()) + ".json")
    recent.write_text("recent orphan", encoding="utf-8")
    unknown = base / "existing-user-file.html"
    unknown.write_text("keep", encoding="utf-8")
    for path in (live, orphan):
        os.utime(path, (time.time() - 90000, time.time() - 90000))
    report = await quarantine_orphans(settings)
    assert report["total_orphans_found"] == 1 and orphan.exists()
    assert report["valid_files_preserved"] == 1
    report = await quarantine_orphans(settings, apply=True)
    assert report["orphans"][0]["action"] == "quarantined" and not orphan.exists()
    assert (base / report["orphans"][0]["destination"]).read_text() == "old orphan"
    assert live.exists() and recent.exists() and unknown.exists()


async def test_cleanup_cannot_race_inflight_publication(client):
    settings = client.app.state.settings
    gate = asyncio.Event()
    async with publication_lock(settings):

        async def cleanup():
            gate.set()
            return await quarantine_orphans(settings, apply=True, min_age_seconds=0)

        task = asyncio.create_task(cleanup())
        await gate.wait()
        await asyncio.sleep(0.1)
        assert not task.done()
    result = await asyncio.wait_for(task, 2)
    assert result["orphans"] == []


async def test_lock_is_shared_with_an_independent_process_and_released_after_cancel(client):
    settings = client.app.state.settings
    code = (
        "import asyncio,sys; from pathlib import Path; from types import SimpleNamespace; "
        "from app.artifact_storage import publication_lock; "
        "exec('async def main():\\n    async with publication_lock(SimpleNamespace(upload_dir=Path(sys.argv[1])), timeout_seconds=.2):\\n        print(\"acquired\")'); "
        "asyncio.run(main())"
    )
    async with publication_lock(settings):
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-X", "utf8", "-c", code, str(settings.upload_dir)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode != 0 and "TimeoutError" in result.stderr
        waiter = asyncio.create_task(publication_lock(settings).__aenter__())
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-X", "utf8", "-c", code, str(settings.upload_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0 and "acquired" in result.stdout, result.stderr


async def test_conversion_trace_measures_actual_io_time(client, monkeypatch):
    import app.artifact_api as api

    headers, _, _, done, _ = await completed(client, "answer")
    original = api.persist_artifact

    async def delayed(*args, **kwargs):
        await asyncio.sleep(0.06)
        return await original(*args, **kwargs)

    monkeypatch.setattr(api, "persist_artifact", delayed)
    response = await client.post(
        f"/api/runs/{done['run_id']}/artifacts", headers=headers, json={"type": "report"}
    )
    assert response.status_code == 200, response.text
    step = await AgentStep.get(run_id=done["run_id"], node="artifact")
    assert 60 <= step.elapsed_ms < 5000
