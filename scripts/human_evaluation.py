"""Frozen, human-reviewed small-sample evaluation; prepare/verify never call models."""

import argparse
import asyncio
import csv
import hashlib
import json
import time
from copy import copy
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from tortoise import Tortoise

from app.agent import RAGAgent
from app.database import close_database, init_database
from app.evaluation import run_evaluation
from app.models import (
    AgentStep,
    Document,
    EvaluationDataset,
    EvaluationRun,
    IngestJob,
    KnowledgeBase,
    ParentChunk,
    PromptTemplate,
)
from app.providers import DashScopeProvider
from app.retrieval import AdvancedRetrieverPipeline
from app.schemas import EvaluationCase, RetrievalConfig
from app.settings import Settings
from app.strategies import PRESETS
from app.vector_store import VectorStore
from scripts.acceptance_real import MeteredTransport

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts/human_eval_20261002_v1"
KB_ID = "06c48af9-5605-4ac2-9683-191118fbdad0"
REQUEST_CAP = 120


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode()).hexdigest()


def write(path, value, *, frozen=False):
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if frozen and path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Refusing to overwrite frozen artifact: {path.name}")
    path.write_text(text, encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def controlled_configs():
    base = RetrievalConfig(
        query_rewrite=False,
        multi_query=False,
        hyde=False,
        top_k=8,
        candidate_k=8,
        context_k=4,
        rerank_k=4,
        context_chars=10000,
    ).model_dump()
    return {
        name: {**base, "hybrid": name != "dense", "rerank": name == "full"}
        for name in ("dense", "hybrid", "full")
    }


def settings():
    private = load(ROOT / ".acceptance/round2/database_private.json")
    return Settings(
        _env_file=None,
        app_mode="production",
        model_provider="demo",
        business_db_url=private["TEST_BUSINESS_DB_URL"],
        vector_db_url=private["TEST_VECTOR_DB_URL"],
        jwt_secret="human-evaluation-isolated-no-http-server-20261002",
        enable_api_workers=False,
    )


async def snapshot(kb_id):
    kb = await KnowledgeBase.get(id=kb_id)
    documents = await Document.filter(kb_id=kb_id, owner_id=kb.owner_id).order_by("id")
    if not documents or any(d.status != "ready" for d in documents):
        raise ValueError("All frozen documents must already be ready")
    if await IngestJob.filter(
        doc_id__in=[d.id for d in documents], status__in=["queued", "processing"]
    ).exists():
        raise ValueError("An index mutation is queued or running")
    docs = []
    for doc in documents:
        content = Path(doc.storage_path).read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        if file_hash != doc.sha256:
            raise ValueError("Original file no longer matches document hash")
        docs.append(
            {
                "id": doc.id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_sha256": file_hash,
                "source_text": content.decode("utf-8"),
                "status": doc.status,
                "index_revision": doc.index_revision,
                "embedding_fingerprint": doc.embedding_fingerprint,
                "parent_count": doc.parent_count,
                "child_count": doc.child_count,
                "tags": doc.tags,
                "metadata": doc.metadata,
            }
        )
    parents = (
        await ParentChunk.filter(kb_id=kb_id, owner_id=kb.owner_id)
        .order_by("id")
        .values("id", "doc_id", "ordinal", "content", "metadata", "edited")
    )
    _, vectors = await Tortoise.get_connection("vectors").execute_query(
        "SELECT id,doc_id,parent_id,content,metadata,embedding_fingerprint,embedding::text AS vector_text "
        "FROM chunk_vector WHERE kb_id=$1 AND owner_id=$2 ORDER BY id",
        [kb_id, kb.owner_id],
    )
    children = []
    for vector in vectors:
        row = dict(vector)
        row["embedding_sha256"] = hashlib.sha256(row.pop("vector_text").encode()).hexdigest()
        if isinstance(row["metadata"], str):
            row["metadata"] = json.loads(row["metadata"])
        children.append(row)
    prompts = (
        await PromptTemplate.filter(owner_id=kb.owner_id).order_by("name").values("name", "content")
    )
    code = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "app").rglob("*.py"))
    }
    for name in ["requirements.lock", "pyproject.toml"]:
        code[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return {
        "kb_id": kb_id,
        "owner_id": kb.owner_id,
        "kb_revision": kb.revision,
        "kb_config": kb.config,
        "documents": docs,
        "parents": parents,
        "children": children,
        "prompts": prompts,
        "application_source_sha256": digest(code),
        "application_files": code,
        "runtime_presets": PRESETS,
        "models": {
            "generation": "qwen-plus",
            "embedding": "text-embedding-v4",
            "dimension": 1024,
            "rerank": "gte-rerank-v2",
        },
    }


def make_dataset(state):
    mapping = {d["filename"].split(".")[0]: d for d in state["documents"]}

    def labels(name):
        return [c["id"] for c in state["children"] if c["doc_id"] == mapping[name]["id"]]

    history = [
        {"role": "user", "content": "员工出差结束后需要在几天内提交报销申请？"},
        {"role": "assistant", "content": "出差结束后7天内提交报销申请。"},
    ]
    return {
        "review_status": "pending_human_confirmation",
        "relevance_definition": "A child is relevant when it supports the correct answer, including explicit negative evidence. Topic/keyword overlap without support is irrelevant.",
        "cases": [
            {
                "id": "Q1",
                "category": "answerable",
                "question": "员工出差结束后需要在几天内提交报销申请？",
                "reference_answer": "出差结束后7天内提交报销申请。",
                "reference_facts": ["员工出差结束后必须在7天内提交报销申请。"],
                "answerable": True,
                "relevant_child_ids": labels("travel"),
                "quote": "员工出差结束后必须在7天内提交报销申请。",
                "expected_behavior": "grounded_answer",
            },
            {
                "id": "Q2",
                "category": "outside",
                "question": "木星的直径是多少？",
                "reference_answer": "所给资料没有木星直径的依据，无法据此回答。",
                "reference_facts": [],
                "answerable": False,
                "relevant_child_ids": [],
                "quote": None,
                "expected_behavior": "refuse_unsupported_fact",
            },
            {
                "id": "Q3",
                "category": "coreference",
                "question": "员工出差报销需要提交哪些材料？",
                "reference_answer": "发票和审批单。",
                "reference_facts": ["报销需要提供发票和审批单。"],
                "answerable": True,
                "relevant_child_ids": labels("travel"),
                "quote": "报销需要提供发票和审批单。",
                "dialogue_question": "那需要哪些材料？",
                "dialogue_history": history,
                "expected_behavior": "grounded_answer",
            },
            {
                "id": "Q4",
                "category": "similar_keywords_negative_evidence",
                "question": "设备维修是否也必须在7天内完成？",
                "reference_answer": "资料未规定设备维修完成时限，不能认定必须在7天内完成。",
                "reference_facts": ["该制度未规定设备维修完成时限。"],
                "answerable": True,
                "relevant_child_ids": labels("repair"),
                "quote": "该制度未规定设备维修完成时限。",
                "distractor_child_ids": labels("travel"),
                "expected_behavior": "qualified_negative_answer",
            },
            {
                "id": "Q5",
                "category": "insufficient_evidence",
                "question": "员工提交发票和审批单后，能否保证报销款次日到账？",
                "reference_answer": "资料只列出报销材料，没有付款时限依据，不能保证次日到账。",
                "reference_facts": [],
                "answerable": False,
                "relevant_child_ids": [],
                "quote": None,
                "distractor_child_ids": labels("travel"),
                "expected_behavior": "refuse_unsupported_guarantee",
            },
        ],
    }


def make_plan():
    configs = controlled_configs()
    return {
        "retrieval_comparison": {
            "questions": "same standalone question per case; history=[]",
            "configs": configs,
            "mode": "rag",
            "replicates": 1,
            "order": "case-major; rotate dense/hybrid/full order by case index",
            "cache": "new retriever per case/arm; no cross-arm embedding cache",
            "quality_scope": "end-to-end RAG quality plus stage retrieval metrics, not pure retrieval quality inferred from refusal",
        },
        "coreference_ablation": {
            "case_id": "Q3",
            "question": "那需要哪些材料？",
            "configs": {
                "rewrite_off": configs["full"],
                "rewrite_on": {**configs["full"], "query_rewrite": True},
            },
            "only_changed_field": "query_rewrite",
            "multi_query": False,
            "mode": "rag",
        },
        "metric_definitions": {
            "P": "TP / returned child count; null if none returned",
            "R": "TP / relevant child count; null when gold set empty",
            "F1": "2TP / (gold count + returned count) when gold nonempty; null for empty gold",
            "stages": ["raw_recall_union", "fused", "post_rank", "context"],
            "primary_stage": "post_rank",
            "aggregate": "macro over defined values, with denominator; per-case values retained",
            "four_quality_metrics": "repository Judge definitions; null where undefined/failed",
            "refusal_rate": "observed Agent rejected fraction, reported separately from retrieval P/R/F1",
            "latency": "per-case observed elapsed_ms and node/recall/ranking timings; one sample is not a significance test",
        },
        "request_cap": REQUEST_CAP,
        "before_after_guard": "snapshot hash before/after every case; abort/invalidate if changed",
        "no_index_writes": True,
    }


async def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    state = await snapshot(KB_ID)
    dataset = make_dataset(state)
    plan = make_plan()
    frozen = {"frozen_at": now(), "snapshot_sha256": digest(state), "state": state}
    if (out / "snapshot.json").exists():
        frozen = load(out / "snapshot.json")
        if digest(state) != frozen["snapshot_sha256"]:
            raise ValueError("Existing snapshot changed")
    write(out / "snapshot.json", frozen, frozen=True)
    write(out / "annotation_proposal.json", dataset, frozen=True)
    write(out / "comparison_plan.json", plan, frozen=True)
    bundle = {
        "snapshot_sha256": digest(state),
        "dataset_sha256": digest(dataset),
        "plan_sha256": digest(plan),
    }
    write(out / "review_bundle.json", bundle, frozen=True)
    rows = [
        "# 小型人工评测核对表（待确认）",
        "",
        f"固定知识库 `{KB_ID}`，revision={state['kb_revision']}。标注和评测期间不重建索引。",
        "",
        "## 固定原文与子块",
        "",
    ]
    for doc in state["documents"]:
        rows += [
            f"### {doc['filename']} / index_revision={doc['index_revision']}",
            "",
            doc["source_text"],
            "",
        ]
        for child in state["children"]:
            if child["doc_id"] == doc["id"]:
                rows += [
                    f"子块 `{child['id']}`；父块 `{child['parent_id']}`；原文偏移 {child['metadata'].get('child_start')}–{child['metadata'].get('child_end')}。",
                    "",
                ]
    rows += [
        "## 请核对以下5题",
        "",
        "相关性：支持正确答案的子块（含明确否定证据）计相关；仅关键词相似不计相关。旧预标注保持原样，以下是新提案。",
        "",
    ]
    for case in dataset["cases"]:
        rows += [
            f"### {case['id']} {case['question']}",
            "",
            f"标准答案：{case['reference_answer']}",
            "",
            f"相关子块：{', '.join(case['relevant_child_ids']) or '空集'}。",
            "",
            f"原文依据：{case['quote'] or '三段全文均没有所问事实依据；应拒绝补造事实。'}",
            "",
        ]
        if case.get("dialogue_question"):
            rows += [
                f"多轮另测：历史谈出差报销，用户问“{case['dialogue_question']}”。公平检索主实验只用上述完整问题。",
                "",
            ]
    rows += [
        "## 公平比较",
        "",
        "原预设 Dense/Hybrid 的 query_rewrite、multi_query 均为 false，Full 均为 true。旧多轮结果不能用于断言检索能力优劣。",
        "主实验三档统一完整问题、top_k=8、candidate_k=8、context_k=4、rerank_k=4、上下文10000字符；全部关闭改写、多查询和HyDE。仅比较 Vector / Vector+BM25+RRF / 再加Rerank。生产预设不修改。",
        "多轮附实验固定同一Full检索配置与历史，只切换 query_rewrite，multi_query 始终false。",
        "确认后才运行正式结果；未确认的预标注结果不混入正式结果。空召回的Precision、空金标的Recall/F1记null，并报告有效分母。",
        "",
        "## 固定摘要",
        "",
        f"快照 `{bundle['snapshot_sha256']}`",
        f"标注提案 `{bundle['dataset_sha256']}`",
        f"实验计划 `{bundle['plan_sha256']}`",
        "",
    ]
    (out / "人工核对表.md").write_text("\n".join(rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "prepared_pending_human_confirmation",
                "documents": len(state["documents"]),
                "children": len(state["children"]),
                "cases": len(dataset["cases"]),
                **bundle,
            },
            ensure_ascii=False,
        )
    )


async def verify(out):
    frozen = load(out / "snapshot.json")
    current = await snapshot(frozen["state"]["kb_id"])
    if digest(current) != frozen["snapshot_sha256"]:
        raise ValueError("Frozen document/index/config/source version changed")
    bundle = load(out / "review_bundle.json")
    if bundle != {
        "snapshot_sha256": digest(current),
        "dataset_sha256": digest(load(out / "annotation_proposal.json")),
        "plan_sha256": digest(load(out / "comparison_plan.json")),
    }:
        raise ValueError("Review bundle changed")
    return current


def require_approval(out):
    approval = load(out / "human_confirmation.json")
    bundle = load(out / "review_bundle.json")
    if (
        approval.get("reviewer") != "user"
        or approval.get("confirmed_bundle") != bundle
        or not approval.get("response_text")
    ):
        raise ValueError("Explicit human confirmation for this exact bundle is required")
    return approval


def prf(gold, returned):
    gold, returned = set(gold), set(returned)
    tp = len(gold & returned)
    return {
        "tp": tp,
        "returned": len(returned),
        "relevant": len(gold),
        "precision": tp / len(returned) if returned else None,
        "recall": tp / len(gold) if gold else None,
        "f1": 2 * tp / (len(gold) + len(returned)) if gold else None,
    }


def stage_metrics(case, retrieval):
    diag = retrieval.get("diagnostics", {})
    stages = {
        "raw_recall_union": {
            c["id"] for ranking in diag.get("rankings", []) for c in ranking["candidates"]
        },
        "fused": {c["id"] for c in diag.get("fused_candidates", [])},
        "post_rank": {c["id"] for c in retrieval.get("candidates", [])},
        "context": {cid for s in retrieval.get("sources", []) for cid in s["child_ids"]},
    }
    return {name: prf(case["relevant_child_ids"], ids) for name, ids in stages.items()}


class ObservedProvider:
    def __init__(self, provider):
        self.provider = provider
        self.records = []

    def __getattr__(self, name):
        return getattr(self.provider, name)

    async def structured(self, schema, task, payload):
        result = await self.provider.structured(schema, task, payload)
        self.records.append({"task": task, "input": payload, "result": result.model_dump()})
        return result

    async def stream_answer(self, *args, **kwargs):
        tokens = []
        async for token in self.provider.stream_answer(*args, **kwargs):
            tokens.append(token)
            yield token
        # Draft evidence stays in acceptance artifacts, never accepted conversation history.
        self.records.append({"task": "generation_draft", "text": "".join(tokens)})

    async def rerank(self, query, documents):
        result = await self.provider.rerank(query, documents)
        self.records.append(
            {"task": "rerank", "query": query, "documents": documents, "raw_result": result}
        )
        return result


def triage(case, result, audit, stages):
    """Mechanical stage localization only; semantic blame needs a reviewer."""
    gold = case["relevant_child_ids"]
    grade = next((a["result"] for a in audit if a["task"] == "grade"), None)
    check = next((a["result"] for a in reversed(audit) if a["task"] == "check"), None)
    if result["status"] == "missing":
        return "执行或Judge异常；人工检查，不能归零"
    if gold and not stages["raw_recall_union"]["tp"]:
        return "召回失败候选：金标未进入任何召回列表"
    if gold and stages["fused"]["tp"] and not stages["post_rank"]["tp"]:
        return "重排/阈值误删候选：检查原始分数与阈值"
    if gold and stages["post_rank"]["tp"] and not stages["context"]["tp"]:
        return "上下文回溯/预算截断候选"
    if gold and stages["context"]["tp"] and grade and not grade["passed"]:
        return "相关性门控误拒候选：证据已到达grade，需核对问题/改写"
    if check and not check["passed"]:
        return "生成错误或核验误拒待人工区分：保留草稿、引用及check理由"
    if result.get("metrics", {}).get("rejected") and not gold:
        return "预期拒答；不据此认定检索差"
    return "核对最终答案与引用；无自动归因"


async def run(out, config):
    destination = out / "formal"
    if destination.exists():
        raise FileExistsError("Formal results already exist; use a new evaluation version")
    approval = require_approval(out)
    state = await verify(out)
    plan = load(out / "comparison_plan.json")
    cases = load(out / "annotation_proposal.json")["cases"]
    # Human-confirmed labels are persisted by existing child IDs; no ingestion/reindex call.
    stored_cases = []
    child_map = {child["id"]: child for child in state["children"]}
    for case in cases:
        for child_id in case["relevant_child_ids"]:
            if case["quote"] not in child_map[child_id]["content"]:
                raise ValueError("Confirmed quotation does not match frozen child")
        stored_cases.append(
            EvaluationCase(
                question=case["question"],
                reference_answer=case["reference_answer"],
                reference_facts=case["reference_facts"],
                answerable=case["answerable"],
                relevant_child_ids=case["relevant_child_ids"],
                annotation_method="human",
                relevant_document_ids=list(
                    dict.fromkeys(child_map[c]["doc_id"] for c in case["relevant_child_ids"])
                ),
                provenance=[
                    {"child_id": c, "quote": case["quote"], "snapshot_sha256": digest(state)}
                    for c in case["relevant_child_ids"]
                ],
            ).model_dump()
        )
    dataset = await EvaluationDataset.create(
        owner_id=state["owner_id"],
        kb_id=state["kb_id"],
        name="人工核对闭环 20261002 v1",
        origin="manual",
        cases=stored_cases,
    )
    write(
        out / "confirmed_dataset.json",
        {
            "id": dataset.id,
            "status": "human_confirmed",
            "confirmation": approval,
            "cases": stored_cases,
        },
        frozen=True,
    )
    destination.mkdir(exist_ok=False)
    credentials = dict(
        csv.reader(
            (ROOT.parents[1] / "默认业务空间-apiKey-7545941.csv")
            .read_text(encoding="utf-8-sig")
            .splitlines()
        )
    )
    real_settings = config.model_copy(
        update={
            "model_provider": "dashscope",
            "dashscope_http_base_url": credentials["dashScope"],
            "dashscope_chat_base_url": credentials["openAiCompatible"],
        }
    )
    # SecretStr remains a SecretStr; never serialize this settings object.
    from pydantic import SecretStr

    real_settings.dashscope_api_key = SecretStr(credentials["apiKey"])
    report = {
        "started_at": now(),
        "status": "running",
        "bundle": load(out / "review_bundle.json"),
        "provider": "dashscope",
        "backend": "MySQL + PostgreSQL/pgvector",
        "request_cap": REQUEST_CAP,
        "confirmed_dataset_id": dataset.id,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requests": [],
        "retrieval_comparison": [],
        "coreference_ablation": [],
        "snapshot_checks": [],
    }
    provider = ObservedProvider(
        DashScopeProvider(real_settings, MeteredTransport(report["requests"], REQUEST_CAP))
    )
    kb = await KnowledgeBase.get(id=state["kb_id"])
    started = time.perf_counter()

    def save():
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        report["request_count"] = len(report["requests"])
        write(destination / "results.json", report)

    async def execute(case, arm, config, dialogue=False):
        await verify(out)
        report["snapshot_checks"].append(
            {"case": case["id"], "arm": arm, "when": "before", "at": now(), "same": True}
        )
        scoped = copy(kb)
        scoped.config = config
        retriever = AdvancedRetrieverPipeline(VectorStore(real_settings), provider, real_settings)
        agent = RAGAgent(provider, retriever)
        sample = EvaluationCase(
            question=case["dialogue_question"] if dialogue else case["question"],
            reference_answer=case["reference_answer"],
            reference_facts=case["reference_facts"],
            answerable=case["answerable"],
            relevant_child_ids=case["relevant_child_ids"],
            annotation_method="human",
            history=case.get("dialogue_history", []) if dialogue else [],
        )
        evaluation = await EvaluationRun.create(
            owner_id=kb.owner_id, kb_id=kb.id, provider="dashscope"
        )
        provider.records = []
        evaluation_result = await run_evaluation(
            kb.owner_id, scoped, [sample], agent, provider, evaluation, mode="rag"
        )
        result = evaluation_result["results"][0]
        stages = stage_metrics(case, result.get("retrieval", {}))
        trace = (
            await AgentStep.filter(run_id=result["run_id"])
            .order_by("ordinal")
            .values("node", "ordinal", "status", "input_summary", "output_summary", "elapsed_ms")
        )
        item = {
            "case_id": case["id"],
            "arm": arm,
            "question": sample.question,
            "history": sample.history and [m.model_dump() for m in sample.history],
            "config": config,
            "evaluation_id": evaluation.id,
            "result": result,
            "stage_prf": stages,
            "audit": provider.records,
            "trace": trace,
            "triage": triage(case, result, provider.records, stages),
        }
        # Override only the acceptance report's undefined-empty metric convention, not production code.
        item["formal_prf"] = stages["post_rank"] if result["status"] != "missing" else None
        await verify(out)
        report["snapshot_checks"].append(
            {"case": case["id"], "arm": arm, "when": "after", "at": now(), "same": True}
        )
        report["coreference_ablation" if dialogue else "retrieval_comparison"].append(item)
        save()
        print(f"{case['id']} {arm}: {result['status']}", flush=True)

    try:
        arms = ["dense", "hybrid", "full"]
        for i, case in enumerate(cases):
            for arm in arms[i % 3 :] + arms[: i % 3]:
                await execute(case, arm, plan["retrieval_comparison"]["configs"][arm])
        multi = next(c for c in cases if c["id"] == "Q3")
        for arm, cfg in plan["coreference_ablation"]["configs"].items():
            await execute(multi, arm, cfg, True)
        aggregates = {}
        for arm in arms:
            records = [r for r in report["retrieval_comparison"] if r["arm"] == arm]
            metrics = {}
            for key in [
                "context_recall",
                "context_precision",
                "faithfulness",
                "answer_relevancy",
                "elapsed_ms",
                "rejected",
            ]:
                values = [r["result"]["metrics"].get(key) for r in records]
                defined = [float(v) for v in values if v is not None]
                metrics["refusal_rate" if key == "rejected" else key] = {
                    "mean": mean(defined) if defined else None,
                    "n": len(defined),
                    "total": len(records),
                }
            for key in ["precision", "recall", "f1"]:
                values = [r["formal_prf"][key] for r in records if r["formal_prf"] is not None]
                defined = [v for v in values if v is not None]
                metrics[key] = {
                    "mean": mean(defined) if defined else None,
                    "n": len(defined),
                    "total": len(records),
                }
            aggregates[arm] = metrics
        report["aggregate"] = aggregates
        report["status"] = (
            "completed"
            if all(
                r["result"]["status"] == "passed"
                for r in report["retrieval_comparison"] + report["coreference_ablation"]
            )
            else "partial_missing"
        )
    except Exception as exc:
        report.update(status="invalidated_or_failed", error_type=type(exc).__name__)
        raise
    finally:
        await provider.close()
        save()


async def main(args):
    config = settings()
    await init_database(config)
    try:
        if args.action == "prepare":
            await prepare(args.output)
        elif args.action == "verify":
            await verify(args.output)
            print("Frozen documents, indices, configuration and code unchanged; no model calls")
        else:
            await run(args.output, config)
    finally:
        await close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "verify", "run"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    asyncio.run(main(parser.parse_args()))
