"""Opt-in, capped real-model acceptance on explicitly isolated MySQL/PG databases."""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx
from tortoise import Tortoise

from app.evaluation import run_evaluation
from app.main import create_app
from app.models import AgentRun, Document, EvaluationRun, KnowledgeBase
from app.providers import DashScopeProvider
from app.schemas import EvaluationCase
from app.settings import Settings
from app.strategies import effective_kb
from app.vector_models import ChunkVector
from scripts.acceptance_real import MeteredTransport

SPEC = {
    "documents": {
        "travel": "员工出差结束后必须在7天内提交报销申请。报销需要提供发票和审批单。",
        "repair": "设备维修由技术支持组负责。该制度未规定设备维修完成时限。",
        "purchase": "采购金额超过5000元时，申请人必须提供两份供应商报价。",
    },
    "cases": [
        {
            "id": "answerable",
            "question": "员工出差结束后需要在几天内提交报销申请？",
            "reference_answer": "7天内。",
            "reference_facts": ["员工出差结束后必须在7天内提交报销申请。"],
            "relevant_documents": ["travel"],
        },
        {
            "id": "outside",
            "question": "木星的直径是多少？",
            "reference_answer": "资料未提供，无法依据资料回答。",
            "reference_facts": [],
            "answerable": False,
            "relevant_documents": [],
        },
        {
            "id": "coreference",
            "question": "那需要哪些材料？",
            "reference_answer": "报销需要发票和审批单。",
            "reference_facts": ["报销需要提供发票和审批单。"],
            "history": [
                {"role": "user", "content": "员工出差结束后需要在几天内提交报销申请？"},
                {"role": "assistant", "content": "出差结束后7天内提交报销申请。"},
            ],
            "relevant_documents": ["travel"],
        },
        {
            "id": "insufficient",
            "question": "设备维修是否也必须在7天内完成？",
            "reference_answer": "不能推断维修须在7天内完成，资料没有规定维修完成时限。",
            "reference_facts": [],
            "answerable": False,
            "relevant_documents": [],
        },
    ],
    "annotation_definition": "Relevant children independently support the requested positive fact; keyword overlap alone is insufficient. Empty positive-fact labels have undefined recall/F1.",
    "annotation_status": "proposed_before_retrieval_pending_human_confirmation",
}


async def main(args):
    if os.environ.get("ACCEPTANCE_ISOLATED_DATABASES") != "1":
        raise RuntimeError("Requires explicitly isolated databases")
    out = args.report.parent
    out.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(SPEC, ensure_ascii=False, indent=2)
    spec_path = out / "sample_spec_before_retrieval.json"
    if spec_path.exists() and spec_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("Refusing to overwrite a frozen sample specification")
    spec_path.write_text(serialized, encoding="utf-8")
    credentials = dict(csv.reader(args.credentials.read_text(encoding="utf-8-sig").splitlines()))
    settings = Settings(
        _env_file=None,
        app_mode="production",
        model_provider="dashscope",
        business_db_url=os.environ["TEST_BUSINESS_DB_URL"],
        vector_db_url=os.environ["TEST_VECTOR_DB_URL"],
        jwt_secret=uuid4().hex + uuid4().hex,
        upload_dir=Path(".acceptance/round2/real-" + uuid4().hex[:8]),
        dashscope_api_key=credentials["apiKey"],
        dashscope_http_base_url=credentials["dashScope"],
        dashscope_chat_base_url=credentials["openAiCompatible"],
        enable_api_workers=False,
    )
    report = {
        "backend": "MySQL + PostgreSQL/pgvector",
        "provider": "dashscope",
        "request_cap": 150,
        "network_requests": [],
        "status": "running",
        "sample_spec_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "strategies": [],
        "chat": [],
        "evaluations": [],
        "checks": [],
    }
    started = time.perf_counter()

    def save():
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        report["request_count"] = len(report["network_requests"])
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    app = create_app(settings, start_worker=False)
    try:
        async with app.router.lifespan_context(app):
            await app.state.provider.close()
            provider = DashScopeProvider(
                settings, MeteredTransport(report["network_requests"], 150)
            )
            app.state.provider = app.state.agent.provider = app.state.retriever.provider = (
                app.state.worker.service.provider
            ) = provider
            for conn_name, sql in [
                ("business", "SELECT VERSION() AS version"),
                (
                    "vectors",
                    "SELECT version(), (SELECT extversion FROM pg_extension WHERE extname='vector') AS vector_version",
                ),
            ]:
                _, rows = await Tortoise.get_connection(conn_name).execute_query(sql)
                report[conn_name + "_version"] = rows
            _, report["vector_column"] = await Tortoise.get_connection("vectors").execute_query(
                "SELECT format_type(atttypid,atttypmod) AS type FROM pg_attribute WHERE attrelid='chunk_vector'::regclass AND attname='embedding'"
            )
            _, report["vector_indexes"] = await Tortoise.get_connection("vectors").execute_query(
                "SELECT indexname,indexdef FROM pg_indexes WHERE tablename='chunk_vector'"
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=300
            ) as client:
                response = await client.post(
                    "/api/auth/register",
                    json={"username": "real_r2_" + uuid4().hex[:8], "password": uuid4().hex},
                )
                assert response.status_code == 201
                headers = {"Authorization": "Bearer " + response.json()["data"]["access_token"]}
                response = await client.post(
                    "/api/knowledge-bases", headers=headers, json={"name": "三文档独立标注验收"}
                )
                kb_id = response.json()["data"]["id"]
                kb = await KnowledgeBase.get(id=kb_id)
                report["kb_id"] = kb_id
                docs, labels = {}, {}
                for name, content in SPEC["documents"].items():
                    response = await client.post(
                        f"/api/knowledge-bases/{kb_id}/documents",
                        headers=headers,
                        files={"file": (name + ".txt", content.encode())},
                    )
                    doc_id = response.json()["data"]["document"]["id"]
                    await app.state.worker.tick()
                    doc = await Document.get(id=doc_id)
                    assert doc.status == "ready"
                    children = await ChunkVector.filter(doc_id=doc_id).values(
                        "id", "content", "parent_id", "metadata"
                    )
                    assert len(children) == 1 and children[0]["content"] == content
                    docs[name] = {"id": doc_id, "children": children}
                # Map predeclared source labels to actual child IDs before ANY query.
                for case in SPEC["cases"]:
                    labels[case["id"]] = [
                        c["id"]
                        for name in case["relevant_documents"]
                        for c in docs[name]["children"]
                    ]
                frozen = {
                    "spec_sha256": report["sample_spec_sha256"],
                    "documents": docs,
                    "labels": labels,
                    "method": "source text inspection before retrieval",
                    "human_status": "pending",
                }
                frozen_path = out / "frozen_child_labels.json"
                if frozen_path.exists():
                    raise RuntimeError("Use a new output directory per real run")
                frozen_path.write_text(
                    json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                report["frozen_labels_sha256"] = hashlib.sha256(
                    frozen_path.read_bytes()
                ).hexdigest()
                report["checks"].append("three_real_embedding_ingestions")
                save()
                for strategy in ["dense", "hybrid", "full"]:
                    for case in SPEC["cases"]:
                        t = time.perf_counter()
                        response = await client.post(
                            "/api/retrieval/debug",
                            headers=headers,
                            json={"kb_id": kb_id, "query": case["question"], "strategy": strategy},
                        )
                        assert response.status_code == 200
                        report["strategies"].append(
                            {
                                "case_id": case["id"],
                                "strategy": strategy,
                                "query": case["question"],
                                "elapsed_ms": round((time.perf_counter() - t) * 1000),
                                "result": response.json()["data"],
                            }
                        )
                    print("retrieval " + strategy + " completed", flush=True)
                    save()
                conversation = None
                for case in SPEC["cases"]:
                    body = {
                        "kb_id": kb_id,
                        "query": case["question"],
                        "strategy": "full",
                        "mode": "agent",
                    }
                    if case["id"] == "coreference":
                        body["conversation_id"] = conversation
                    response = await client.post("/api/chat/runs", headers=headers, json=body)
                    assert response.status_code == 202
                    data = response.json()["data"]
                    if case["id"] == "answerable":
                        conversation = data["conversation_id"]
                    await app.state.chat_worker.tick()
                    run = await AgentRun.get(id=data["run_id"])
                    trace = (await client.get(f"/api/runs/{run.id}", headers=headers)).json()[
                        "data"
                    ]
                    report["chat"].append(
                        {
                            "case_id": case["id"],
                            "status": run.status,
                            "answer": run.answer,
                            "citations": run.citations,
                            "trace": trace,
                        }
                    )
                    assert run.status == "completed"
                    if case["id"] == "answerable":
                        assert "7" in run.answer and run.citations
                    if case["id"] == "coreference":
                        assert "发票" in run.answer and "审批单" in run.answer and run.citations
                    if case["id"] == "outside":
                        assert not run.citations
                    if case["id"] == "insufficient":
                        assert any(w in run.answer for w in ["未", "不", "无法", "暂无"])
                    print("chat " + case["id"] + " completed", flush=True)
                    save()
                report["checks"].append("four_real_agent_cases_and_verified_citations")
                cases = [
                    EvaluationCase(
                        **{k: v for k, v in c.items() if k not in {"id", "relevant_documents"}}
                    )
                    for c in SPEC["cases"]
                ]
                for strategy in ["dense", "hybrid", "full"]:
                    evaluation = await EvaluationRun.create(
                        owner_id=kb.owner_id, kb_id=kb_id, provider="dashscope"
                    )
                    result = await run_evaluation(
                        kb.owner_id,
                        effective_kb(kb, strategy),
                        cases,
                        app.state.agent,
                        provider,
                        evaluation,
                        mode="rag",
                    )
                    report["evaluations"].append({"strategy": strategy, "result": result})
                    print("evaluation " + strategy + " " + result["status"], flush=True)
                    save()
                report["checks"].append("three_strategy_four_case_real_judges")
                report["status"] = (
                    "passed"
                    if all(e["result"]["status"] == "completed" for e in report["evaluations"])
                    else "partial_missing"
                )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
    finally:
        save()
        print(
            json.dumps(
                {k: report[k] for k in ["status", "checks", "request_count", "elapsed_ms"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    return report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(main(args)) else 1)
