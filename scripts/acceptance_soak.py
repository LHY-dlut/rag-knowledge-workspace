"""Actual 24h observation on a frozen candidate. Never calls paid models."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


def now():
    return datetime.now(UTC).isoformat()


class FreezeDrift(RuntimeError):
    pass


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def container_state(project):
    names = [
        project + "-" + service
        for service in ("api-1", "worker-1", "worker-2", "mysql-1", "postgres-1", "frontend-1")
    ]
    result = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{json .Name}} {{json .Image}} {{json .State.StartedAt}} {{json .RestartCount}}",
            *names,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("Container state unavailable")
    return result.stdout.splitlines()


async def main(args):
    from scripts.freeze_release import inventory

    args.output.mkdir(parents=True, exist_ok=False)
    baseline = inventory()
    configuration_sha = hashlib.sha256((PROJECT / ".env").read_bytes()).hexdigest()
    initial_containers = container_state(args.project)
    write(
        args.output / "freeze.json",
        {
            "time": now(),
            "source": baseline,
            "containers": initial_containers,
            "configuration_sha256": configuration_sha,
        },
    )
    status = {
        "status": "running",
        "started_at": now(),
        "planned_hours": args.hours,
        "provider": "demo",
        "backend": "real MySQL + PostgreSQL/pgvector",
        "samples": 0,
        "end_to_end": 0,
        "failures": 0,
        "elapsed_seconds": 0,
        "completed": False,
    }
    started = time.monotonic()
    last_observation = started
    max_observation_gap = 0.0
    write(args.output / "status.json", status)
    async with httpx.AsyncClient(base_url=args.url, timeout=90, trust_env=False) as client:
        health = (await client.get("/api/health")).json()["data"]
        if health["provider"] != "demo" or health["mode"] != "production":
            raise RuntimeError("Soak requires real dual DB with Demo; paid execution forbidden")
        # Token TTL covers less than 24h. Renew using synthetic credentials
        # retained only in memory, never exported to observation records.
        login = {"username": "soakrun_" + uuid4().hex[:12], "password": uuid4().hex}
        auth = (await client.post("/api/auth/register", json=login)).json()["data"]
        client.headers["Authorization"] = "Bearer " + auth["access_token"]
        kb = (
            await client.post(
                "/api/knowledge-bases",
                json={
                    "name": "24小时隔离运行",
                    "config": {
                        "hybrid": True,
                        "query_rewrite": False,
                        "multi_query": False,
                        "hyde": False,
                    },
                },
            )
        ).json()["data"]["id"]
        text = "仓储报损规则：盘点周期为每周一次，盘点由运营组负责。"
        upload = (
            await client.post(
                f"/api/knowledge-bases/{kb}/documents", files={"file": ("soak.txt", text.encode())}
            )
        ).json()["data"]
        for _ in range(120):
            job = (await client.get(f"/api/jobs/{upload['job_id']}")).json()["data"]
            if job["status"] == "done":
                break
            if job["status"] == "failed":
                raise RuntimeError("Soak fixture ingestion failed")
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("Soak fixture ingestion timeout")
        next_answer, next_login = 0, 3600
        while time.monotonic() - started < args.hours * 3600:
            row = {"time": now(), "elapsed_seconds": time.monotonic() - started}
            try:
                gap = time.monotonic() - last_observation
                max_observation_gap = max(max_observation_gap, gap)
                last_observation = time.monotonic()
                if gap > 180:
                    raise RuntimeError("Observation gap exceeded 180 seconds")
                if inventory() != baseline:
                    raise FreezeDrift("Frozen candidate source changed")
                if hashlib.sha256((PROJECT / ".env").read_bytes()).hexdigest() != configuration_sha:
                    raise FreezeDrift("Frozen private configuration changed")
                current = container_state(args.project)
                if current != initial_containers:
                    raise FreezeDrift("Container/image/restart state changed during observation")
                ready = await client.get("/api/ready")
                live = await client.get("/api/live")
                row["ready"], row["live"] = ready.status_code, live.status_code
                assert ready.status_code == live.status_code == 200
                elapsed = time.monotonic() - started
                if elapsed >= next_login:
                    auth = (await client.post("/api/auth/login", json=login)).json()["data"]
                    client.headers["Authorization"] = "Bearer " + auth["access_token"]
                    next_login = elapsed + 3600
                if elapsed >= next_answer:
                    run_response = await client.post(
                        "/api/chat/runs",
                        json={"kb_id": kb, "query": "仓储报损规则的盘点周期是什么？"},
                    )
                    run_response.raise_for_status()
                    run_id = run_response.json()["data"]["run_id"]
                    for _ in range(300):
                        response = await client.get(f"/api/runs/{run_id}")
                        response.raise_for_status()
                        trace = response.json()["data"]
                        if trace["status"] in {"completed", "failed", "cancelled"}:
                            break
                        await asyncio.sleep(0.2)
                    assert trace["status"] == "completed" and "每周一次" in trace["answer"]
                    assert trace["citations"] and trace["rejected"] is False
                    row["run_id"], row["run_elapsed_ms"] = run_id, trace["elapsed_ms"]
                    stats = subprocess.run(
                        [
                            "docker",
                            "stats",
                            "--no-stream",
                            "--format",
                            "{{json .}}",
                            *[
                                args.project + "-" + service
                                for service in (
                                    "api-1",
                                    "worker-1",
                                    "worker-2",
                                    "mysql-1",
                                    "postgres-1",
                                )
                            ],
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if stats.returncode:
                        raise RuntimeError("Resource observation unavailable")
                    row["resources"] = [json.loads(line) for line in stats.stdout.splitlines()]
                    status["end_to_end"] += 1
                    next_answer = elapsed + 600
                row["status"] = "passed"
            except Exception as exc:
                row["status"], row["error_type"] = "failed", type(exc).__name__
                status["failures"] += 1
                # Keep failure data, do not silently reset the 24h clock.
            with (args.output / "samples.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            status["samples"] += 1
            status["elapsed_seconds"] = time.monotonic() - started
            status["max_observation_gap_seconds"] = max_observation_gap
            write(args.output / "status.json", status)
            if row.get("error_type") == "FreezeDrift":
                break
            await asyncio.sleep(min(60, max(0, args.hours * 3600 - (time.monotonic() - started))))
    duration_met = time.monotonic() - started >= args.hours * 3600
    max_observation_gap = max(max_observation_gap, time.monotonic() - last_observation)
    if max_observation_gap > 180:
        status["failures"] += 1
        status["observation_gap_exceeded"] = True
    status.update(
        status="passed" if duration_met and not status["failures"] else "failed",
        completed=duration_met,
        finished_at=now(),
        elapsed_seconds=time.monotonic() - started,
        max_observation_gap_seconds=max_observation_gap,
    )
    write(args.output / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    import re

    if not re.fullmatch(r"[a-z][a-z0-9_-]{1,62}", args.project):
        parser.error("Invalid explicit Compose project name")
    args.output = args.output.resolve()
    if args.output.is_relative_to(PROJECT):
        parser.error("Keep private acceptance observations outside source")
    if not 0 < args.hours <= 168:
        parser.error("Use an actual duration between 0 and 168 hours")
    try:
        asyncio.run(main(args))
    except BaseException as exc:
        path = args.output / "status.json"
        if path.exists():
            status = json.loads(path.read_text(encoding="utf-8"))
            status.update(
                status="failed", completed=False, stopped_at=now(), error_type=type(exc).__name__
            )
            write(path, status)
        raise
