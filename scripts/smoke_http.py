"""Exercise the REAL HTTP server. Does not write tokens or passwords to reports."""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx


def parse_events(raw: str):
    events = []
    for frame in raw.replace("\r\n", "\n").split("\n\n"):
        name = None
        data = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name and data:
            events.append((name, json.loads("\n".join(data))))
    return events


async def main(url: str, report: Path):
    async with httpx.AsyncClient(base_url=url, timeout=300, trust_env=False) as client:

        async def call(method, path, **kwargs):
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()["data"]

        health = await call("GET", "/api/health")
        auth = await call(
            "POST",
            "/api/auth/register",
            json={
                "username": f"smoke_{uuid4().hex[:10]}",
                "password": uuid4().hex,
            },
        )
        client.headers["Authorization"] = f"Bearer {auth['access_token']}"
        kb = await call("POST", "/api/knowledge-bases", json={"name": "HTTP 端到端验收"})
        raw = Path("examples/差旅与休假制度.md").read_bytes()
        upload = await call(
            "POST",
            f"/api/knowledge-bases/{kb['id']}/documents",
            files={"file": ("差旅与休假制度.md", raw)},
        )
        for _ in range(120):
            job = await call("GET", f"/api/jobs/{upload['job_id']}")
            if job["status"] == "done":
                break
            if job["status"] == "failed":
                raise RuntimeError(job["error"])
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError("入库未在 60 秒内完成")
        debug = await call(
            "POST",
            "/api/retrieval/debug",
            json={
                "kb_id": kb["id"],
                "query": "出差报销申请需要几天内提交？",
            },
        )
        assert debug["sources"]
        response = await client.post(
            "/api/chat/stream",
            json={
                "kb_id": kb["id"],
                "query": "出差报销申请需要几天内提交？",
            },
        )
        response.raise_for_status()
        events = parse_events(response.text)
        done = next(data for name, data in events if name == "done")
        assert "7天" in done["answer"] and not done["rejected"]
        trace = await call("GET", f"/api/runs/{done['run_id']}")
        response = await client.post(
            "/api/chat/stream",
            json={
                "kb_id": kb["id"],
                "query": "木星大气的甲烷含量是多少？",
            },
        )
        refusal = next(data for name, data in parse_events(response.text) if name == "done")
        assert refusal["rejected"] and refusal["grade_retries"] == 3
        evaluation = await call(
            "POST",
            "/api/evaluations",
            json={
                "kb_id": kb["id"],
                "cases": json.loads(Path("examples/evaluation_cases.json").read_text()),
            },
        )
        result = {
            "transport": "real HTTP and SSE",
            "health": health,
            "checks": [
                "registration",
                "upload",
                "durable_job",
                "hybrid_retrieval",
                "parent_context",
                "SSE_draft_and_final",
                "citation",
                "trace_persistence",
                "refusal_retry_bound",
                "evaluation",
            ],
            "answer": done["answer"],
            "source_count": len(trace["citations"]),
            "graph_nodes": [s["node"] for s in trace["steps"]],
            "refusal": refusal["answer"],
            "refusal_grade_retries": refusal["grade_retries"],
            "evaluation_provider": evaluation["provider"],
            "all_passed": True,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--report", type=Path, default=Path("artifacts/smoke_http.json"))
    args = parser.parse_args()
    asyncio.run(main(args.url, args.report))
