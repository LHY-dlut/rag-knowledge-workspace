"""One document/one question over actual Nginx -> API -> worker -> real models."""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx


def main():
    started_at = datetime.now(UTC).isoformat()
    out = Path("artifacts/acceptance_round2/production_http.json")
    result = {
        "transport": "real TCP via production Nginx",
        "backend": "MySQL + PostgreSQL/pgvector",
        "provider": "dashscope",
        "started_at": started_at,
        "status": "running",
    }
    start = time.perf_counter()
    with httpx.Client(base_url="http://127.0.0.1:18088", trust_env=False, timeout=120) as client:
        credentials = {"username": "production_" + uuid4().hex[:8], "password": uuid4().hex}
        response = client.post("/api/auth/register", json=credentials)
        assert response.status_code == 201
        token = response.json()["data"]["access_token"]
        Path(".acceptance/round2/browser_private.json").write_text(
            json.dumps({"token": token, **credentials}), encoding="utf-8"
        )
        headers = {"Authorization": "Bearer " + token}
        result["health"] = client.get("/api/health").json()
        kb = client.post(
            "/api/knowledge-bases", headers=headers, json={"name": "Docker真实模型验收"}
        ).json()["data"]["id"]
        response = client.post(
            f"/api/knowledge-bases/{kb}/documents",
            headers=headers,
            files={
                "file": (
                    "policy.txt",
                    "员工出差结束后7天内提交报销申请。需要提供发票和审批单。".encode(),
                )
            },
        )
        data = response.json()["data"]
        for _ in range(120):
            job = client.get(f"/api/jobs/{data['job_id']}", headers=headers).json()["data"]
            if job["status"] in {"done", "failed"}:
                break
            time.sleep(0.25)
        assert job["status"] == "done", job["status"]
        result["job"] = {k: job.get(k) for k in ["id", "status", "attempts"]}
        response = client.post(
            "/api/chat/runs",
            headers=headers,
            json={
                "kb_id": kb,
                "query": "出差结束后几天内提交报销申请？需要什么材料？",
                "strategy": "full",
                "mode": "agent",
            },
        )
        assert response.status_code == 202
        data = response.json()["data"]
        with client.stream("GET", f"/api/runs/{data['run_id']}/events", headers=headers) as stream:
            transcript = "\n".join(stream.iter_lines())
        result["sse"] = transcript
        trace = client.get(f"/api/runs/{data['run_id']}", headers=headers).json()["data"]
        result["trace"] = trace
        assert trace["status"] == "completed"
        assert "7" in trace["answer"] and "发票" in trace["answer"] and trace["citations"]
        assert "event: done" in transcript and "event: citations" in transcript
        result["status"] = "passed"
    logs = subprocess.run(
        ["docker", "logs", "--since", started_at, "rag-acceptance-r2-worker-1"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # Keep only allowlisted request metadata, never whole container environments/logs.
    requests = []
    for line in (logs.stdout + logs.stderr).splitlines():
        if "HTTP Request: POST" not in line:
            continue
        model = (
            "text-embedding-v4"
            if "text-embedding" in line
            else ("gte-rerank-v2" if "text-rerank" in line else "qwen-plus")
        )
        requests.append({"model": model, "status": 200 if "200 OK" in line else "non-200"})
    result["network_requests"] = requests
    result["request_count"] = len(requests)
    result["elapsed_ms"] = round((time.perf_counter() - start) * 1000)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ["status", "request_count", "elapsed_ms"]}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
