import time
from statistics import mean
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from app.agent import GraphState, RAGAgent, citation_check
from app.models import AgentRun, EvaluationRun, KnowledgeBase
from app.providers import ModelProvider, tokenize
from app.schemas import EvaluationCase, MetadataFilter, StrictModel


class JudgeCounts(StrictModel):
    supported_reference_fact_indices: list[StrictInt] = Field(default_factory=list)
    relevant_source_ids: list[str] = Field(default_factory=list)
    answer_claim_count: int = Field(ge=0, le=200)
    supported_answer_claim_count: int = Field(ge=0, le=200)
    answer_relevancy: float = Field(ge=0, le=1)
    reason: str = Field(max_length=2000)

    @model_validator(mode="after")
    def check_counts(self):
        if self.supported_answer_claim_count > self.answer_claim_count:
            raise ValueError("Supported claims cannot exceed total claims")
        return self


def context_average_precision(source_ids: list[str], relevant_ids: set[str]) -> float:
    """AP over returned evidence; relevant blocks near the top receive credit."""
    precisions, hits = [], 0
    for rank, source_id in enumerate(source_ids, 1):
        if source_id in relevant_ids:
            hits += 1
            precisions.append(hits / rank)
    return mean(precisions) if precisions else 0.0


async def run_evaluation(
    owner_id: str,
    kb: KnowledgeBase,
    cases: list[EvaluationCase],
    agent: RAGAgent,
    provider: ModelProvider,
    evaluation: EvaluationRun,
    mode: Literal["rag", "agent"] = "agent",
):
    results = []
    from app.vector_models import ChunkVector

    for case in cases:
        if case.relevant_child_ids is not None:
            labels = set(case.relevant_child_ids)
            owned = set(
                await ChunkVector.filter(
                    id__in=list(labels), owner_id=owner_id, kb_id=kb.id
                ).values_list("id", flat=True)
            )
            if labels != owned:
                raise ValueError("人工相关子块标注包含不存在或未授权的 ID")
    for case in cases:
        started = time.perf_counter()
        run = await AgentRun.create(
            owner_id=owner_id,
            kb_id=kb.id,
            conversation_id="",
            query=case.question,
            config_snapshot=kb.config,
        )
        initial: GraphState = {
            "run_id": run.id,
            "owner_id": owner_id,
            "kb": kb,
            "query": case.question,
            "history": [message.model_dump() for message in case.history],
            "filters": MetadataFilter(),
            "mode": mode,
            "grade_retries": 0,
            "check_retries": 0,
            "step_ordinal": 0,
        }
        try:
            state = await agent.graph.ainvoke(initial, config={"recursion_limit": 64})
            sources = (
                [s.model_dump() for s in state.get("retrieval", {}).sources]
                if state.get("retrieval")
                else []
            )
            answer = state["answer"]
            context = "\n".join(s["content"] for s in sources)
            tokens = set(tokenize(case.question))
            demo_counts = {
                "supported_reference_fact_indices": [
                    i for i, f in enumerate(case.reference_facts) if f in context
                ],
                "relevant_source_ids": [
                    s["source_id"]
                    for s in sources
                    if any(f in s["content"] for f in case.reference_facts)
                ],
                "answer_claim_count": 0 if state.get("rejected") else 1,
                "supported_answer_claim_count": 0 if state.get("rejected") else 1,
                "answer_relevancy": len(tokens & set(tokenize(answer))) / max(len(tokens), 1),
                "reason": "演示：词法匹配与抽取答案；不能用作真实质量结论",
            }
            # Demo lexical counts are never sent to a real Judge as suggested scores.
            judge_payload = {"demo_metrics": demo_counts} if evaluation.provider == "demo" else {}
            judge = await provider.structured(
                JudgeCounts,
                "evaluation",
                {
                    "question": case.question,
                    "history": [message.model_dump() for message in case.history],
                    "reference_answer": case.reference_answer,
                    "reference_facts": case.reference_facts,
                    "answer": answer,
                    "sources": sources,
                    "definitions": {
                        "supported_reference_fact_indices": "被至少一条检索证据支持的参考事实的零基索引",
                        "relevant_source_ids": "对回答问题或支持参考事实有用的证据ID",
                        "answer_claim_count": "答案中可核验的事实声明总数；纯拒答为0",
                        "supported_answer_claim_count": "由所引用证据支持的答案事实声明数",
                        "answer_relevancy": "答案对原问题的回应程度，范围0到1",
                    },
                    **judge_payload,
                },
            )
            if any(
                i < 0 or i >= len(case.reference_facts)
                for i in judge.supported_reference_fact_indices
            ):
                raise ValueError("Judge returned an invalid fact index")
            source_ids = {s["source_id"] for s in sources}
            if not set(judge.relevant_source_ids) <= source_ids:
                raise ValueError("Judge returned an invalid source ID")
            retrieved_doc_ids = {s["document_id"] for s in sources}
            retrieved_children = (
                {c.id for c in state["retrieval"].candidates} if state.get("retrieval") else set()
            )
            labels = set(case.relevant_child_ids) if case.relevant_child_ids is not None else None
            tp = len(retrieved_children & labels) if labels is not None else None
            precision = (
                tp / len(retrieved_children)
                if labels is not None and retrieved_children
                else (0.0 if labels is not None else None)
            )
            recall = tp / len(labels) if labels else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else (0.0 if recall is not None else None)
            )
            metrics = {
                "retrieval_precision": precision,
                "retrieval_recall": recall,
                "retrieval_f1": f1,
                "context_recall": len(set(judge.supported_reference_fact_indices))
                / len(case.reference_facts)
                if case.reference_facts
                else None,
                "context_precision": context_average_precision(
                    [s["source_id"] for s in sources], set(judge.relevant_source_ids)
                ),
                "context_relevance_fraction": len(set(judge.relevant_source_ids)) / len(sources)
                if sources
                else 0,
                "faithfulness": judge.supported_answer_claim_count / judge.answer_claim_count
                if judge.answer_claim_count
                else None,
                "answer_relevancy": judge.answer_relevancy,
                "document_recall": len(retrieved_doc_ids & set(case.relevant_document_ids))
                / len(set(case.relevant_document_ids))
                if case.relevant_document_ids
                else None,
                "citation_validity": citation_check(answer, sources).passed
                if not state.get("rejected")
                else None,
                "rejected": state.get("rejected", False),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            result = {
                "question": case.question,
                "history": [message.model_dump() for message in case.history],
                "answer": answer,
                "sources": sources,
                "metrics": metrics,
                "judge": judge.model_dump(),
                "run_id": run.id,
                "config": kb.config,
                "mode": mode,
                "status": "passed",
                "auxiliary_label_status": "human" if labels is not None else "missing",
                "retrieval": state["retrieval"].model_dump(
                    exclude={"candidates": {"__all__": {"embedding"}}}
                )
                if state.get("retrieval")
                else {},
            }
            results.append(result)
            await AgentRun.filter(id=run.id).update(
                status="completed",
                answer=answer,
                citations=sources,
                grade_retries=state.get("grade_retries", 0),
                check_retries=state.get("check_retries", 0),
                elapsed_ms=metrics["elapsed_ms"],
            )
            await EvaluationRun.filter(id=evaluation.id).update(results=results)
        except Exception as exc:
            # Missing or failed judgments are missing values, never invented scores.
            missing = dict.fromkeys(
                (
                    "context_recall",
                    "context_precision",
                    "faithfulness",
                    "answer_relevancy",
                    "document_recall",
                    "context_relevance_fraction",
                    "retrieval_precision",
                    "retrieval_recall",
                    "retrieval_f1",
                )
            )
            missing.update(elapsed_ms=round((time.perf_counter() - started) * 1000), rejected=None)
            results.append(
                {
                    "question": case.question,
                    "status": "missing",
                    "error_type": type(exc).__name__,
                    "metrics": missing,
                    "run_id": run.id,
                    "config": kb.config,
                    "mode": mode,
                }
            )
            await AgentRun.filter(id=run.id).update(
                status="failed", error="评测执行或 Judge 失败，指标缺失"
            )
            await EvaluationRun.filter(id=evaluation.id).update(results=results)
        except BaseException:
            await AgentRun.filter(id=run.id).update(status="failed", error="评测执行中断")
            raise
    aggregate = {}
    for name in (
        "retrieval_precision",
        "retrieval_recall",
        "retrieval_f1",
        "context_recall",
        "context_precision",
        "faithfulness",
        "answer_relevancy",
        "document_recall",
        "context_relevance_fraction",
        "elapsed_ms",
    ):
        values = [r["metrics"][name] for r in results if r["metrics"][name] is not None]
        aggregate[name] = mean(values) if values else None
    refusals = [
        float(r["metrics"]["rejected"]) for r in results if r["metrics"]["rejected"] is not None
    ]
    aggregate["refusal_rate"] = mean(refusals) if refusals else None
    status = "partial_missing" if any(r["status"] == "missing" for r in results) else "completed"
    await EvaluationRun.filter(id=evaluation.id).update(
        status=status, metrics=aggregate, results=results
    )
    return {
        "id": evaluation.id,
        "status": status,
        "metrics": aggregate,
        "results": results,
        "provider": evaluation.provider,
        "demo": evaluation.provider == "demo",
    }
