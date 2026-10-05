"""Readiness and bounded worker shutdown. These checks never call a model."""

import asyncio
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

from tortoise import Tortoise


async def database_readiness(timeout: float = 2.0) -> dict[str, bool]:
    async def probe(alias):
        try:
            async with asyncio.timeout(timeout):
                await Tortoise.get_connection(alias).execute_query("SELECT 1")
            return True
        except Exception:
            # Driver exceptions can contain credentials/DSNs; expose only status.
            return False

    values = await asyncio.gather(*(probe(alias) for alias in ("business", "vectors")))
    return dict(zip(("business", "vectors"), values, strict=True))


@contextmanager
def shutdown_signals(stop: asyncio.Event):
    """Linux container signals, with a Windows main-thread fallback."""
    loop = asyncio.get_running_loop()
    previous = {}
    installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        try:
            loop.add_signal_handler(sig, stop.set)
            installed.append(sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    try:
        yield
    finally:
        for sig, handler in previous.items():
            if sig in installed:
                loop.remove_signal_handler(sig)
            signal.signal(sig, handler)


async def supervise_workers(workers, stop: asyncio.Event, grace_seconds: float):
    """Stop accepting work; finish active work, then cancel at the deadline.

    Cancellation retains existing lease fencing/recovery semantics. A worker
    that unexpectedly exits is a process failure, not a healthy idle worker.
    """
    tasks = [asyncio.create_task(worker.run()) for worker in workers]
    stopping = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait([*tasks, stopping], return_when=asyncio.FIRST_COMPLETED)
        if stopping not in done:
            for task in done:
                task.result()
            raise RuntimeError("Worker exited unexpectedly")
    finally:
        for worker in workers:
            worker.stopping.set()
        _, pending = await asyncio.wait(tasks, timeout=grace_seconds)
        for task in pending:
            task.cancel()
        stopping.cancel()
        await asyncio.gather(*tasks, stopping, return_exceptions=True)


def write_worker_health(path: Path, ready: bool, databases: dict[str, bool]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"ready": ready, "databases": databases, "pid": os.getpid(), "time": time.time()}
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


async def worker_health_loop(path: Path, stop: asyncio.Event):
    try:
        while not stop.is_set():
            databases = await database_readiness()
            write_worker_health(path, all(databases.values()), databases)
            try:
                await asyncio.wait_for(stop.wait(), 10)
            except TimeoutError:
                pass
    finally:
        write_worker_health(path, False, {})


def worker_health_valid(path: Path, max_age: float = 35) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - value["time"]
        return value["ready"] is True and 0 <= age <= max_age
    except (OSError, ValueError, KeyError, TypeError):
        return False


if __name__ == "__main__":
    from app.settings import Settings

    raise SystemExit(0 if worker_health_valid(Settings().worker_health_file) else 1)
