"""Coordinate artifact publication with conservative, reversible maintenance."""

import asyncio
import errno
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from tortoise.transactions import in_transaction

from app.models import GeneratedArtifact

logger = logging.getLogger(__name__)


def artifact_directory(settings) -> Path:
    return (settings.upload_dir.parent / "generated_artifacts").resolve()


@asynccontextmanager
async def publication_lock(settings, timeout_seconds=10):
    """An OS lock shared by API, workers and cleanup; crash releases the lock.

    Nonblocking acquisition avoids an abandoned blocking thread on cancellation.
    The lock covers file writes AND the transaction's commit/rollback.
    """
    base = artifact_directory(settings)
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".publication.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        acquired = False
        started = time.monotonic()
        try:
            while not acquired:
                handle.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if time.monotonic() - started >= timeout_seconds:
                        raise TimeoutError("成果存储正在使用，请稍后重试") from None
                    await asyncio.sleep(0.05)
            yield
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass
class PendingFile:
    artifact_id: str
    path: Path
    owned: bool = False


async def finish_io(operation):
    """Wait for local I/O to settle before allowing rollback cleanup."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # A file-writing thread cannot be cancelled. Do not release the lock or
        # unlink a partially written file while it can still write afterwards.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise


async def rollback_files(files):
    for pending in files:
        if not pending.owned:
            continue
        try:
            # Transaction exit already happened. A commit acknowledgement may
            # have been lost; never delete a file with a committed database row.
            if not await GeneratedArtifact.filter(id=pending.artifact_id).exists():
                pending.path.unlink(missing_ok=True)
        except Exception as error:
            # Database uncertainty leaves a recoverable orphan for maintenance.
            logger.warning("Artifact rollback cleanup deferred (%s)", type(error).__name__)


@asynccontextmanager
async def artifact_transaction(settings, *, enabled=True):
    files = []
    if not enabled:
        async with in_transaction("business") as connection:
            yield connection, files
        return
    async with publication_lock(settings):
        try:
            async with in_transaction("business") as connection:
                yield connection, files
        except BaseException:
            task = asyncio.create_task(rollback_files(files))
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()
            raise
