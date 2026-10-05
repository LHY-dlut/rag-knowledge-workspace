"""Small opt-in real-model acceptance. Secrets are read in memory; never reported."""

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

from app.main import create_app
from app.models import AgentRun, Document
from app.providers import DashScopeProvider
from app.settings import Settings


class MeteredTransport(httpx.AsyncBaseTransport):
    def __init__(self, records, cap):
        self.transport = httpx.AsyncHTTPTransport()
        self.records, self.cap = records, cap

    async def handle_async_request(self, request):
        if len(self.records) >= self.cap:
            raise RuntimeError("Real request cap reached")
        started = time.perf_counter()
        body = json.loads(request.content)
        record = {"model": body.get("model"), "stream": body.get("stream", False)}
        self.records.append(record)
        try:
            response = await self.transport.handle_async_request(request)
            record["http_status"] = response.status_code
            return response
        except Exception as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["response_headers_ms"] = round((time.perf_counter() - started) * 1000)

    async def aclose(self):
        await self.transport.aclose()


async def main(args):
    root = Path(__file__).resolve().parent.parent
    credentials = dict(csv.reader(args.credentials.read_text(encoding="utf-8-sig").splitlines()))
    runtime = root / ".acceptance" / f"real-{uuid4().hex[:8]}"
    runtime.mkdir(parents=True)
    report = {
        "backend": "SQLite",
        "provider": "dashscope",
        "request_cap": 40,
        "network_requests": [],
        "checks": [],
        "status": "running",
    }
    settings = Settings(
        _env_file=None,
        app_mode="demo",
        model_provider="dashscope",
        business_db_url=f"sqlite://{runtime}/business.sqlite3",
        vector_db_url=f"sqlite://{runtime}/vectors.sqlite3",
        upload_dir=runtime / "uploads",
        dashscope_api_key=credentials["apiKey"],
        dashscope_http_base_url=credentials["dashScope"],
        dashscope_chat_base_url=credentials["openAiCompatible"],
        enable_api_workers=False,
    )
    app = create_app(settings, start_worker=False)
    started = time.perf_counter()
    try:
        async with app.router.lifespan_context(app):
            await app.state.provider.close()
            provider = DashScopeProvider(settings, MeteredTransport(report["network_requests"], 40))
            app.state.provider = provider
            app.state.agent.provider = provider
            app.state.retriever.provider = provider
            app.state.worker.service.provider = provider
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=300
            ) as client:
                response = await client.post(
                    "/api/auth/register",
                    json={"username": "real_acceptance", "password": uuid4().hex},
                )
                assert response.status_code == 201
                headers = {"Authorization": "Bearer " + response.json()["data"]["access_token"]}
                response = await client.post(
                    "/api/knowledge-bases", headers=headers, json={"name": "百炼小样本验收"}
                )
                kb = response.json()["data"]["id"]
                content = "员工出差结束后必须在7天内提交报销申请。报销需要提供发票和审批单。\n\n设备维修由技术支持组负责。"
                response = await client.post(
                    f"/api/knowledge-bases/{kb}/documents",
                    headers=headers,
                    files={"file": ("acceptance.txt", content.encode())},
                )
                doc_id = response.json()["data"]["document"]["id"]
                await app.state.worker.tick()
                doc = await Document.get(id=doc_id)
                assert doc.status == "ready", doc.error
                report["checks"].append("real_embedding_ingestion_1024")
                report["strategies"] = []
                question = "员工出差结束后需要在几天内提交报销申请？"
                for strategy in ("dense", "hybrid", "full"):
                    t = time.perf_counter()
                    response = await client.post(
                        "/api/retrieval/debug",
                        headers=headers,
                        json={"kb_id": kb, "query": question, "strategy": strategy},
                    )
                    assert response.status_code == 200, response.text
                    result = response.json()["data"]
                    assert result["sources"]
                    report["strategies"].append(
                        {
                            "strategy": strategy,
                            "query": question,
                            "elapsed_ms": round((time.perf_counter() - t) * 1000),
                            **result,
                        }
                    )
                report["checks"].append("same_question_dense_hybrid_full_and_real_rerank")
                response = await client.post(
                    "/api/chat/runs",
                    headers=headers,
                    json={"kb_id": kb, "query": question, "strategy": "full", "mode": "agent"},
                )
                run_id = response.json()["data"]["run_id"]
                await app.state.chat_worker.tick()
                run = await AgentRun.get(id=run_id)
                assert run.status == "completed", run.error
                assert run.citations and "7" in run.answer
                report["answer"] = run.answer
                report["citations"] = run.citations
                report["trace"] = (await client.get(f"/api/runs/{run.id}", headers=headers)).json()[
                    "data"
                ]
                report["checks"].append("qwen_route_rewrite_retrieve_grade_generate_check_citation")
                # No independent human labels were supplied: auxiliary P/R/F1 must be missing.
                response = await client.post(
                    "/api/evaluations",
                    headers=headers,
                    json={
                        "kb_id": kb,
                        "cases": [
                            {
                                "question": question,
                                "reference_answer": "7天内提交报销申请。",
                                "reference_facts": ["员工出差结束后必须在7天内提交报销申请。"],
                            }
                        ],
                    },
                )
                assert response.status_code == 200
                report["evaluation"] = response.json()["data"]
                assert report["evaluation"]["status"] == "completed"
                assert report["evaluation"]["metrics"]["retrieval_f1"] is None
                report["checks"].append("four_real_judge_metrics_auxiliary_labels_missing")
                report["status"] = "passed"
                await provider.close()
    except Exception as exc:
        report["status"] = "failed"
        # Deliberately exclude exception text: third-party exceptions may include request data.
        report["error_type"] = type(exc).__name__
    finally:
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        report["request_count"] = len(report["network_requests"])
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {k: report[k] for k in ("status", "request_count", "checks", "elapsed_ms")},
                ensure_ascii=False,
            )
        )
    return report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True, type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/acceptance/real_models.json")
    )
    options = parser.parse_args()
    raise SystemExit(0 if asyncio.run(main(options)) else 1)
