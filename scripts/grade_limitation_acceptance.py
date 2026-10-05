"""Compare a grade-only revision with the immutable human-reviewed experiment."""

import argparse
import asyncio
import csv
import hashlib
import time
from collections import Counter
from copy import copy
from pathlib import Path
from statistics import mean

from pydantic import SecretStr

from app.agent import RAGAgent, bind_grade_evidence, validate_grade_support
from app.database import close_database, init_database
from app.evaluation import run_evaluation
from app.models import AgentStep, EvaluationDataset, EvaluationRun, KnowledgeBase
from app.providers import DashScopeProvider
from app.retrieval import AdvancedRetrieverPipeline
from app.schemas import EvaluationCase, GradeDecision, Judgment
from app.vector_store import VectorStore
from scripts import human_evaluation as base
from scripts.acceptance_real import MeteredTransport

ROOT = base.ROOT
OLD = base.DEFAULT_OUT
DEFAULT_OUT = ROOT / "artifacts/grade_limitation_v2_20261002"
ALLOWED_FILES = {"app/agent.py", "app/providers.py", "app/schemas.py"}
REQUEST_CAP = 150


def data_projection(state):
    return {
        k: v
        for k, v in state.items()
        if k
        not in {
            "application_source_sha256",
            "application_files",
        }
    }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_inventory():
    roots = [OLD, ROOT / "artifacts/acceptance_round2"]
    roots += sorted((ROOT / "artifacts").glob("acceptance-history-*"))
    return {
        p.relative_to(ROOT).as_posix(): file_hash(p)
        for folder in roots
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


async def prepare(out):
    if (out / "version_manifest.json").exists():
        raise FileExistsError("Version already prepared; verify or use another output directory")
    original = base.load(OLD / "snapshot.json")
    current = await base.snapshot(original["state"]["kb_id"])
    if data_projection(current) != data_projection(original["state"]):
        raise ValueError("Document/index/prompt/config/model inputs differ from the old experiment")
    changed = {
        name
        for name, value in current["application_files"].items()
        if original["state"]["application_files"].get(name) != value
    }
    if changed != ALLOWED_FILES:
        raise ValueError(f"Unexpected application changes: {sorted(changed)}")
    for name in changed:
        before = out / "source_before" / Path(name).name
        if file_hash(before) != original["state"]["application_files"][name]:
            raise ValueError("Saved pre-change source does not match the old code")
    dataset = base.load(OLD / "confirmed_dataset.json")
    actual = await EvaluationDataset.get(id=dataset["id"])
    if actual.cases != dataset["cases"] or actual.origin != "manual":
        raise ValueError("Persisted human labels changed")
    for index in (1, 4):
        if actual.cases[index]["relevant_child_ids"] != []:
            raise ValueError("Empty human gold labels must remain empty")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "grade-limitation-v2",
        "prepared_at": base.now(),
        "authorization": "用户授权修复grade并沿用已确认标签进行新版本对照复验",
        "baseline": OLD.relative_to(ROOT).as_posix(),
        "baseline_snapshot_sha256": original["snapshot_sha256"],
        "data_projection_sha256": base.digest(data_projection(current)),
        "new_snapshot_sha256": base.digest(current),
        "changed_application_files": sorted(changed),
        "human_dataset_id": dataset["id"],
        "human_confirmation_sha256": file_hash(OLD / "human_confirmation.json"),
        "human_cases_sha256": base.digest(actual.cases),
        "plan_sha256": file_hash(OLD / "comparison_plan.json"),
        "historical_files": old_inventory(),
        "main_report_before": {},
        "runner_sha256": file_hash(Path(__file__)),
        "request_cap": REQUEST_CAP,
    }
    before_reports = out / "reports_before"
    before_reports.mkdir(exist_ok=False)
    for name in ("acceptance_report.md", "acceptance_report.json"):
        content = (ROOT / "artifacts" / name).read_bytes()
        (before_reports / name).write_bytes(content)
        manifest["main_report_before"][name] = hashlib.sha256(content).hexdigest()
    base.write(
        out / "snapshot.json",
        {"state": current, "snapshot_sha256": base.digest(current)},
        frozen=True,
    )
    base.write(out / "version_manifest.json", manifest, frozen=True)
    base.write(
        out / "human_labels_reference.json",
        {
            "dataset_id": actual.id,
            "cases": actual.cases,
            "confirmation_source": "../human_eval_20261002_v1/human_confirmation.json",
            "note": "原用户确认标签不变；新版本修改授权不冒充新的人工标签确认",
        },
        frozen=True,
    )
    print(
        "Prepared new code version; original documents, indices, models, questions and labels unchanged"
    )


async def verify(out):
    manifest = base.load(out / "version_manifest.json")
    frozen = base.load(out / "snapshot.json")
    current = await base.snapshot(frozen["state"]["kb_id"])
    if base.digest(current) != manifest["new_snapshot_sha256"]:
        raise ValueError("New experiment snapshot changed during evaluation")
    if base.digest(data_projection(current)) != manifest["data_projection_sha256"]:
        raise ValueError("Original input projection changed")
    if old_inventory() != manifest["historical_files"]:
        raise ValueError("Historical evidence changed")
    for name, expected in manifest["main_report_before"].items():
        if file_hash(out / "reports_before" / name) != expected:
            raise ValueError("Historical report backup changed")
    actual = await EvaluationDataset.get(id=manifest["human_dataset_id"])
    if base.digest(actual.cases) != manifest["human_cases_sha256"]:
        raise ValueError("Human labels changed")
    if file_hash(OLD / "comparison_plan.json") != manifest["plan_sha256"]:
        raise ValueError("Retrieval plan changed")
    return current


def boundary_cases(state):
    docs = {d["filename"]: d for d in state["documents"]}
    return [
        {
            "id": "N1",
            "kind": "具体回答",
            "question": "设备维修工作由哪个部门负责？",
            "reference_answer": "由技术支持组负责。",
            "reference_facts": ["设备维修由技术支持组负责。"],
            "answerable": True,
            "expected": "supported_answer",
            "citation_doc": docs["repair.txt"]["id"],
        },
        {
            "id": "N2",
            "kind": "条件限制",
            "question": "采购金额6000元，只提供一份供应商报价是否满足本资料的要求？",
            "reference_answer": "不满足；金额超过5000元，需提供两份供应商报价。",
            "reference_facts": [docs["purchase.txt"]["source_text"]],
            "answerable": True,
            "expected": "supported",
            "citation_doc": docs["purchase.txt"]["id"],
        },
        {
            "id": "N3",
            "kind": "主体混淆",
            "question": "设备维修申请必须提供两份供应商报价吗？",
            "reference_answer": "资料没有维修申请报价要求的依据，不能作此认定。",
            "reference_facts": [],
            "answerable": False,
            "expected": "insufficient",
        },
        {
            "id": "N4",
            "kind": "不相关的未规定干扰",
            "question": "采购申请必须在7天内完成审批吗？",
            "reference_answer": "資料未提供采购审批期限，不能使用报销期限或维修时限说明作证。",
            "reference_facts": [],
            "answerable": False,
            "expected": "insufficient",
        },
    ]


def semantic_probes():
    # New synthetic boundary fixtures; no DB/index writes and no human gold label claim.
    negative = "访客登记办法明确未规定登记完成时限。"
    condition = "仅完成年度培训的正式员工可以申领补贴。"
    grade = [
        ("G1", "访客登记必须当日完成吗？", negative, "supported_limitation"),
        ("G2", "设施检查必须当日完成吗？", "", "insufficient"),
        ("G3", "访客登记必须当日完成吗？", "承包商备案申请必须当日提交。", "insufficient"),
        ("G4", "所有人员都可以申领补贴吗？", condition, "supported_limitation"),
        ("G5", "设备保养必须三天内完成吗？", "培训办法未规定报名截止日期。", "insufficient"),
        (
            "G6",
            "档案借阅应在申请后多久办理？",
            "档案借阅应在申请后五个工作日内办理。借阅申请需要申请表和主管签字。",
            "supported_answer",
        ),
    ]
    check = [
        (
            "C1",
            "访客登记必须当日完成吗？",
            negative,
            "根据所提供访客登记办法，不能认定必须当日完成登记[S1]。",
            True,
        ),
        ("C2", "访客登记是否有时限？", negative, "现实中访客登记不存在任何完成时限[S1]。", False),
        (
            "C3",
            "访客登记必须当日完成吗？",
            "承包商备案申请必须当日提交。",
            "访客登记必须当日完成[S1]。",
            False,
        ),
        (
            "C4",
            "所有人员都可以申领补贴吗？",
            condition,
            "根据本规定，只有完成年度培训的正式员工可以申领补贴[S1]。",
            True,
        ),
        ("C5", "所有人员都可以申领补贴吗？", condition, "所有人员均可申领补贴[S1]。", False),
        (
            "C6",
            "档案借阅应在申请后多久办理？",
            "档案借阅应在申请后五个工作日内办理。借阅申请需要申请表和主管签字。",
            "档案借阅应在申请后五个工作日内办理，不因主管是否签字而改变办理期限[S1]。",
            False,
        ),
    ]
    return grade, check


def aggregate(items):
    values = {}
    for name in (
        "context_recall",
        "context_precision",
        "faithfulness",
        "answer_relevancy",
        "elapsed_ms",
        "rejected",
    ):
        present = [
            float(i["result"]["metrics"][name])
            for i in items
            if i["result"]["metrics"].get(name) is not None
        ]
        values["refusal_rate" if name == "rejected" else name] = {
            "mean": mean(present) if present else None,
            "n": len(present),
            "total": len(items),
        }
    for name in ("precision", "recall", "f1"):
        present = [
            i["formal_prf"][name]
            for i in items
            if i.get("formal_prf") is not None and i["formal_prf"][name] is not None
        ]
        values[name] = {
            "mean": mean(present) if present else None,
            "n": len(present),
            "total": len(items),
        }
    return values


async def run(out, config):
    destination = out / "formal"
    destination.mkdir(exist_ok=False)
    state = await verify(out)
    plan = base.load(OLD / "comparison_plan.json")
    cases = base.load(OLD / "annotation_proposal.json")["cases"]
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
            "dashscope_api_key": SecretStr(credentials["apiKey"]),
            "dashscope_http_base_url": credentials["dashScope"],
            "dashscope_chat_base_url": credentials["openAiCompatible"],
        }
    )
    report = {
        "version": "grade-limitation-v2",
        "started_at": base.now(),
        "status": "running",
        "backend": "real MySQL + PostgreSQL/pgvector",
        "provider": "dashscope",
        "models": state["models"],
        "runtime": "local locked Python environment; actual DB and models",
        "requests": [],
        "retrieval_comparison": [],
        "boundary_workflows": [],
        "boundary_probes": [],
        "snapshot_checks": [],
        "request_cap": REQUEST_CAP,
        "runner_sha256": file_hash(Path(__file__)),
    }
    provider = base.ObservedProvider(
        DashScopeProvider(real_settings, MeteredTransport(report["requests"], REQUEST_CAP))
    )
    kb = await KnowledgeBase.get(id=state["kb_id"])
    started = time.perf_counter()

    def save():
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
        report["request_count"] = len(report["requests"])
        report["requests_by_model"] = dict(Counter(r["model"] for r in report["requests"]))
        base.write(destination / "results.json", report)

    async def guard(case, arm, when):
        await verify(out)
        report["snapshot_checks"].append(
            {"case": case, "arm": arm, "when": when, "same": True, "at": base.now()}
        )

    async def execute(case, arm, boundary=False):
        await guard(case["id"], arm, "before")
        scoped = copy(kb)
        scoped.config = plan["retrieval_comparison"]["configs"][arm]
        agent = RAGAgent(
            provider, AdvancedRetrieverPipeline(VectorStore(real_settings), provider, real_settings)
        )
        sample = EvaluationCase(
            question=case["question"],
            reference_answer=case["reference_answer"],
            reference_facts=case["reference_facts"],
            answerable=case["answerable"],
            relevant_child_ids=None if boundary else case["relevant_child_ids"],
            annotation_method=None if boundary else "human",
        )
        evaluation = await EvaluationRun.create(
            owner_id=kb.owner_id, kb_id=kb.id, provider="dashscope"
        )
        provider.records = []
        answer = await run_evaluation(
            kb.owner_id, scoped, [sample], agent, provider, evaluation, mode="rag"
        )
        result = answer["results"][0]
        item = {
            "case_id": case["id"],
            "arm": arm,
            "question": case["question"],
            "config": scoped.config,
            "result": result,
            "audit": provider.records,
            "trace": await AgentStep.filter(run_id=result["run_id"])
            .order_by("ordinal")
            .values("node", "ordinal", "status", "output_summary", "elapsed_ms"),
            "evaluation_id": evaluation.id,
        }
        if not boundary:
            item["stage_prf"] = base.stage_metrics(case, result.get("retrieval", {}))
            item["formal_prf"] = (
                item["stage_prf"]["post_rank"] if result["status"] != "missing" else None
            )
        await guard(case["id"], arm, "after")
        report["boundary_workflows" if boundary else "retrieval_comparison"].append(item)
        save()
        print(
            f"{case['id']} {arm}: {result['status']}, rejected={result['metrics'].get('rejected')}",
            flush=True,
        )

    try:
        for index, case in enumerate(cases):
            arms = ["dense", "hybrid", "full"]
            for arm in arms[index % 3 :] + arms[: index % 3]:
                await execute(case, arm)
        extra = boundary_cases(state)
        base.write(
            out / "boundary_dataset.json",
            {
                "annotation": "Codex-authored independent boundary fixtures, not additional user-confirmed human labels",
                "cases": extra,
                "auxiliary_prf": "missing; no independent human child labels",
            },
            frozen=True,
        )
        for case in extra:
            await execute(case, "full", boundary=True)
        grade, check = semantic_probes()
        for ident, query, content, expected in grade:
            await guard(ident, "grade_probe", "before")
            source = [{"source_id": "S1", "content": content}] if content else []
            since = time.perf_counter()
            decision = (
                bind_grade_evidence(
                    await provider.structured(
                        GradeDecision,
                        "grade",
                        {
                            "query": query,
                            "sources": source,
                        },
                    ),
                    source,
                )
                if source
                else GradeDecision(
                    passed=False, support="insufficient", reason="无证据，不调用模型"
                )
            )
            decision, binding = await validate_grade_support(provider, decision, source, query)
            report["boundary_probes"].append(
                {
                    "id": ident,
                    "task": "grade",
                    "query": query,
                    "sources": source,
                    "expected": expected,
                    "actual": decision.model_dump(),
                    "binding": binding.model_dump() if binding else None,
                    "passed": decision.support == expected,
                    "elapsed_ms": round((time.perf_counter() - since) * 1000),
                    "model_calls": (1 + int(binding is not None)) if source else 0,
                }
            )
            await guard(ident, "grade_probe", "after")
            save()
        for ident, query, content, draft, expected in check:
            await guard(ident, "check_probe", "before")
            since = time.perf_counter()
            source = [{"source_id": "S1", "content": content}]
            judgment = await provider.structured(
                Judgment, "check", {"query": query, "answer": draft, "sources": source}
            )
            report["boundary_probes"].append(
                {
                    "id": ident,
                    "task": "check",
                    "query": query,
                    "sources": source,
                    "draft": draft,
                    "expected": expected,
                    "actual": judgment.model_dump(),
                    "passed": judgment.passed == expected,
                    "elapsed_ms": round((time.perf_counter() - since) * 1000),
                    "model_calls": 1,
                }
            )
            await guard(ident, "check_probe", "after")
            save()
        report["aggregate"] = {
            arm: aggregate([r for r in report["retrieval_comparison"] if r["arm"] == arm])
            for arm in ("dense", "hybrid", "full")
        }
        report["boundary_aggregate"] = aggregate(report["boundary_workflows"])
        report["status"] = (
            "completed"
            if all(
                i["result"]["status"] == "passed"
                for i in report["retrieval_comparison"] + report["boundary_workflows"]
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
    config = base.settings()
    await init_database(config)
    try:
        if args.action == "prepare":
            await prepare(args.output)
        elif args.action == "verify":
            await verify(args.output)
            print(
                "Frozen inputs, confirmed labels, original results and historical reports preserved"
            )
        else:
            await run(args.output, config)
    finally:
        await close_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "verify", "run"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    asyncio.run(main(parser.parse_args()))
