"""Real independent API/worker processes; SQLite evidence is explicitly labelled."""

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import httpx

from scripts.verification_runtime import stop_process


async def frames(response):
    event = {}
    async for line in response.aiter_lines():
        if not line:
            if "data" in event:
                event["data"] = json.loads(event["data"])
                yield event
            event = {}
        elif line.startswith(("event:", "data:", "id:")):
            key, value = line.split(":", 1)
            event[key] = value.strip()


async def main():
    root = Path(__file__).resolve().parent.parent
    runtime = root / ".acceptance" / f"recovery-{uuid4().hex[:8]}"
    runtime.mkdir(parents=True)
    port = int(os.environ.get("RECOVERY_TEST_PORT", "18082"))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    env = {
        **os.environ,
        "APP_MODE": "demo",
        "MODEL_PROVIDER": "demo",
        "ENABLE_API_WORKERS": "false",
        "BUSINESS_DB_URL": f"sqlite://{runtime}/business.sqlite3",
        "VECTOR_DB_URL": f"sqlite://{runtime}/vectors.sqlite3",
        "UPLOAD_DIR": str(runtime / "uploads"),
        "WORKER_POLL_SECONDS": "0.1",
        "WORKER_LEASE_SECONDS": "30",
        "PYTHONUTF8": "1",
    }
    real_db = os.environ.get("ACCEPTANCE_ISOLATED_DATABASES") == "1"
    if real_db:
        env.update(
            APP_MODE="production",
            BUSINESS_DB_URL=os.environ["TEST_BUSINESS_DB_URL"],
            VECTOR_DB_URL=os.environ["TEST_VECTOR_DB_URL"],
        )
    processes, logs = [], []
    checks = []
    report = {"provider": "demo", "backend": "SQLite", "transport": "real TCP", "checks": checks}
    if real_db:
        report["backend"] = "MySQL + PostgreSQL/pgvector"

    def start(kind):
        log = (runtime / f"{kind}-{len(processes)}.log").open("w", encoding="utf-8")
        logs.append(log)
        command = (
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            if kind == "api"
            else [sys.executable, "-m", "scripts.acceptance_process_worker"]
        )
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
        processes.append(process)
        return process

    def stop(process):
        if process.poll() is None:
            stop_process(process)

    started = time.perf_counter()
    try:
        subprocess.run(
            [sys.executable, "-m", "app.bootstrap"],
            env=env,
            cwd=root,
            check=True,
            capture_output=True,
        )
        api = start("api")
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=30
        ) as client:

            async def ready():
                for _ in range(150):
                    try:
                        if (await client.get("/api/health")).status_code == 200:
                            return
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.1)
                raise AssertionError("API unavailable")

            await ready()
            auth = (
                await client.post(
                    "/api/auth/register",
                    json={"username": "recovery_" + uuid4().hex[:12], "password": uuid4().hex},
                )
            ).json()["data"]
            headers = {"Authorization": "Bearer " + auth["access_token"]}
            kb = (
                await client.post(
                    "/api/knowledge-bases", headers=headers, json={"name": "恢复验收"}
                )
            ).json()["data"]["id"]
            response = await client.post(
                f"/api/knowledge-bases/{kb}/documents",
                headers=headers,
                files={
                    "file": (
                        "policy.txt",
                        "公司差旅政策规定，员工出差后必须在7天内提交报销申请。".encode(),
                    )
                },
            )
            job = response.json()["data"]["job_id"]
            assert (await client.get(f"/api/jobs/{job}", headers=headers)).json()["data"][
                "status"
            ] == "queued"
            workers = [start("worker") for _ in range(2)]
            report["worker_processes"] = 2
            for _ in range(200):
                if (await client.get(f"/api/jobs/{job}", headers=headers)).json()["data"][
                    "status"
                ] == "done":
                    break
                await asyncio.sleep(0.1)
            else:
                raise AssertionError("Independent ingestion worker failed")
            checks.append("independent_api_and_ingestion_worker")
            body = {"kb_id": kb, "query": "报销申请需要几天内提交？"}
            async with client.stream(
                "POST", "/api/chat/stream", headers=headers, json=body
            ) as response:
                seen = []
                async for frame in frames(response):
                    seen.append(frame)
                    if frame["event"] == "meta":
                        run, conversation = (
                            frame["data"]["run_id"],
                            frame["data"]["conversation_id"],
                        )
                    if frame["event"] == "draft_token":
                        break
            cursor = seen[-1]["id"]
            stop(api)
            api = start("api")
            await ready()
            replay = []
            async with client.stream(
                "GET", f"/api/runs/{run}/events", headers={**headers, "Last-Event-ID": cursor}
            ) as response:
                async for frame in frames(response):
                    replay.append(frame)
            assert replay[-1]["event"] == "done"
            assert "7天" in replay[-1]["data"]["answer"]
            numbers = [int(f["id"].rsplit(":", 1)[1]) for f in seen + replay]
            assert numbers == list(range(1, max(numbers) + 1))
            checks.extend(
                [
                    "disconnect_does_not_cancel",
                    "api_restart_replay",
                    "resume_strict_sequence",
                    "verified_answer_after_restart",
                ]
            )
            full = await client.get(f"/api/runs/{run}/events", headers=headers)
            assert "event: done" in full.text
            trace = (await client.get(f"/api/runs/{run}", headers=headers)).json()["data"]
            assert trace["answer"] == replay[-1]["data"]["answer"]
            checks.append("terminal_record_compensation")
            body["conversation_id"] = conversation
            data = (await client.post("/api/chat/runs", headers=headers, json=body)).json()["data"]
            assert (
                await client.post("/api/chat/runs", headers=headers, json=body)
            ).status_code == 409
            async with client.stream(
                "GET", f"/api/runs/{data['run_id']}/events", headers=headers
            ) as response:
                async for frame in frames(response):
                    if frame["event"] == "draft_token":
                        break
            await client.post(f"/api/runs/{data['run_id']}/cancel", headers=headers)
            await asyncio.sleep(2.2)
            cancelled = (await client.get(f"/api/runs/{data['run_id']}", headers=headers)).json()[
                "data"
            ]
            assert cancelled["status"] == "cancelled" and not cancelled["answer"]
            assert (
                "event: done"
                not in (
                    await client.get(f"/api/runs/{data['run_id']}/events", headers=headers)
                ).text
            )
            checks.extend(
                ["same_conversation_concurrency_rejected", "explicit_cancel_no_late_publication"]
            )
            data = (await client.post("/api/chat/runs", headers=headers, json=body)).json()["data"]
            async with client.stream(
                "GET", f"/api/runs/{data['run_id']}/events", headers=headers
            ) as response:
                async for frame in frames(response):
                    if frame["event"] == "draft_token":
                        break
            for worker in workers:
                stop(worker)
            if real_db:
                from datetime import UTC, datetime

                from app.database import close_database, init_database
                from app.models import AgentRun
                from app.settings import Settings

                await init_database(
                    Settings(
                        _env_file=None,
                        business_db_url=env["BUSINESS_DB_URL"],
                        vector_db_url=env["VECTOR_DB_URL"],
                    )
                )
                try:
                    await AgentRun.filter(id=data["run_id"]).update(
                        lease_until=datetime(2000, 1, 1, tzinfo=UTC)
                    )
                finally:
                    await close_database()
            else:
                with closing(sqlite3.connect(runtime / "business.sqlite3")) as db:
                    db.execute(
                        "UPDATE agentrun SET lease_until=? WHERE id=?",
                        ("2000-01-01 00:00:00+00:00", data["run_id"]),
                    )
                    db.commit()
            workers = [start("worker") for _ in range(2)]
            for _ in range(150):
                failed = (await client.get(f"/api/runs/{data['run_id']}", headers=headers)).json()[
                    "data"
                ]
                if failed["status"] == "failed":
                    break
                await asyncio.sleep(0.1)
            assert failed["status"] == "failed" and not failed["answer"]
            checks.append("worker_crash_expiry_terminal_error_without_model_resubmit")
            report.update(status="passed", events=seen + replay, trace=trace)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for process in processes:
            stop(process)
        for log in logs:
            log.close()
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        (root / os.environ.get("RECOVERY_REPORT", "artifacts/acceptance/recovery.json")).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: report[k] for k in ("status", "checks", "elapsed_ms")}, ensure_ascii=False
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
