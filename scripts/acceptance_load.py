"""Small-team HTTP/queue load on isolated real databases with Demo models only."""

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx


def percentile(values, fraction):
    return sorted(values)[max(0, min(len(values) - 1, math.ceil(len(values) * fraction) - 1))]


async def main(args):
    records, profiles = [], []
    args.records = records
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.url, timeout=120, trust_env=False) as client:
        health = (await client.get("/api/health")).json()["data"]
        if health["provider"] != "demo" or health["mode"] != "production":
            raise RuntimeError("Requires production dual databases with Demo; no paid load allowed")

        async def call(kind, method, path, **kwargs):
            before = time.perf_counter()
            try:
                response = await client.request(method, path, **kwargs)
            except Exception as exc:
                records.append(
                    {
                        "kind": kind,
                        "error_type": type(exc).__name__,
                        "elapsed_ms": (time.perf_counter() - before) * 1000,
                    }
                )
                raise
            records.append(
                {
                    "kind": kind,
                    "status": response.status_code,
                    "elapsed_ms": (time.perf_counter() - before) * 1000,
                }
            )
            response.raise_for_status()
            return response.json()["data"]

        for i in range(args.users):
            auth = await call(
                "setup",
                "POST",
                "/api/auth/register",
                json={"username": "load_" + uuid4().hex[:12], "password": uuid4().hex},
            )
            headers = {"Authorization": "Bearer " + auth["access_token"]}
            kb = await call(
                "setup",
                "POST",
                "/api/knowledge-bases",
                headers=headers,
                json={"name": f"独立负载用户{i}", "config": {"hybrid": True}},
            )
            text = f"仓储报损规则：仓库{i}的盘点周期为每周一次，责任人为运营组{i}。"
            uploaded = await call(
                "setup",
                "POST",
                f"/api/knowledge-bases/{kb['id']}/documents",
                headers=headers,
                files={"file": ("load.txt", text.encode())},
            )
            profiles.append(
                {
                    "headers": headers,
                    "kb_id": kb["id"],
                    "doc_id": uploaded["document"]["id"],
                    "job": uploaded["job_id"],
                }
            )
        for profile in profiles:
            for _ in range(120):
                job = await call(
                    "setup", "GET", f"/api/jobs/{profile['job']}", headers=profile["headers"]
                )
                if job["status"] == "done":
                    break
                if job["status"] == "failed":
                    raise RuntimeError("Load setup ingestion failed")
                await asyncio.sleep(0.25)
            else:
                raise TimeoutError("Load setup ingestion did not complete")

        load_started = time.perf_counter()
        completed_rounds = {}

        async def requests(index, profile):
            headers, kb = profile["headers"], profile["kb_id"]
            j = 0
            while (
                time.perf_counter() - load_started < args.duration_seconds
                if args.duration_seconds
                else j < args.rounds
            ):
                own = await call(
                    "list", "GET", f"/api/knowledge-bases/{kb}/documents", headers=headers
                )
                assert [doc["id"] for doc in own] == [profile["doc_id"]]
                retrieval = await call(
                    "retrieval",
                    "POST",
                    "/api/retrieval/debug",
                    headers=headers,
                    json={"kb_id": kb, "query": "仓储报损规则盘点周期"},
                )
                assert retrieval["candidates"]
                assert all(c["doc_id"] == profile["doc_id"] for c in retrieval["candidates"])
                if j % 10 == 0 if args.duration_seconds else j < args.chat_rounds:
                    before = time.perf_counter()
                    run = await call(
                        "submit",
                        "POST",
                        "/api/chat/runs",
                        headers=headers,
                        json={"kb_id": kb, "query": "仓储报损规则的盘点周期是什么？"},
                    )
                    for _ in range(240):
                        trace = await call(
                            "poll", "GET", f"/api/runs/{run['run_id']}", headers=headers
                        )
                        if trace["status"] in {"completed", "failed", "cancelled"}:
                            break
                        await asyncio.sleep(0.1)
                    assert trace["status"] == "completed" and "每周" in trace["answer"]
                    assert trace["citations"] and trace["rejected"] is False
                    records.append(
                        {
                            "kind": "chat_end_to_end",
                            "terminal_status": trace["status"],
                            "elapsed_ms": (time.perf_counter() - before) * 1000,
                        }
                    )
                j += 1
                if args.pause_seconds:
                    await asyncio.sleep(args.pause_seconds)
            completed_rounds[str(index)] = j
            other = profiles[(index + 1) % len(profiles)]["kb_id"]
            before = time.perf_counter()
            denial = await client.get(f"/api/knowledge-bases/{other}/documents", headers=headers)
            records.append(
                {
                    "kind": "access_denial",
                    "status": denial.status_code,
                    "elapsed_ms": (time.perf_counter() - before) * 1000,
                }
            )
            assert denial.status_code == 404

        outcomes = await asyncio.gather(
            *(requests(i, p) for i, p in enumerate(profiles)), return_exceptions=True
        )
    grouped = defaultdict(list)
    for row in records:
        grouped[row["kind"]].append(row["elapsed_ms"])
    summary = {
        kind: {
            "count": len(values),
            "p50_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "max_ms": max(values),
        }
        for kind, values in grouped.items()
    }
    report = {
        "status": "failed" if any(isinstance(r, BaseException) for r in outcomes) else "passed",
        "time": datetime.now(UTC).isoformat(),
        "backend": "real MySQL + PostgreSQL/pgvector",
        "provider": "demo",
        "users": args.users,
        "rounds": args.rounds,
        "chat_rounds": args.chat_rounds,
        "elapsed_seconds": time.perf_counter() - started,
        "load_seconds": time.perf_counter() - load_started,
        "planned_load_seconds": args.duration_seconds,
        "completed_rounds": completed_rounds,
        "worker_failures": [type(r).__name__ for r in outcomes if isinstance(r, BaseException)],
        "latencies": summary,
        "http_request_count": sum("status" in r or "error_type" in r for r in records),
        "statuses": dict(Counter(r["status"] for r in records if "status" in r)),
        "records": records,
        "does_not_measure_real_model_capacity": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"}, ensure_ascii=False
        )
    )
    if report["status"] != "passed":
        raise RuntimeError("Load assertions failed; partial evidence retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--chat-rounds", type=int, default=2)
    parser.add_argument("--duration-seconds", type=int, default=0)
    parser.add_argument("--pause-seconds", type=float, default=0)
    args = parser.parse_args()
    if not 2 <= args.users <= 10 or not 1 <= args.chat_rounds <= args.rounds <= 100:
        parser.error("Use 2..10 users and 1 <= chat-rounds <= rounds <= 100")
    if not 0 <= args.duration_seconds <= 3600 or not 0 <= args.pause_seconds <= 10:
        parser.error("Use duration 0..3600 seconds and pause 0..10 seconds")
    if args.report.exists():
        parser.error("Output already exists; use a new report path")
    try:
        asyncio.run(main(args))
    except BaseException as exc:
        if not args.report.exists():
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as file:
                json.dump(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "records": getattr(args, "records", []),
                        "provider": "demo",
                    },
                    file,
                )
        raise
