"""Applications, feedback and reusable evidence-grounded evaluation datasets."""

import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from tortoise.transactions import in_transaction

from app.evaluation import run_evaluation
from app.models import (
    AgentRun,
    Application,
    Document,
    EvaluationDataset,
    EvaluationRun,
    Feedback,
    IngestJob,
    ParentChunk,
    User,
)
from app.parsers import SUPPORTED_TYPES
from app.schemas import (
    ApplicationInput,
    ComparisonRequest,
    DatasetGenerate,
    DatasetInput,
    EvaluationCase,
    FeedbackDatasetInput,
    FeedbackInput,
    GeneratedCases,
    Judgment,
)
from app.security import current_user, owned_kb
from app.strategies import PRESETS, effective_kb

router = APIRouter(prefix="/api")


@router.put("/documents/{doc_id}/file", status_code=202)
async def replace_file(
    doc_id: str, request: Request, file: UploadFile = File(...), user: User = Depends(current_user)
):
    settings = request.app.state.settings
    doc = await Document.get_or_none(id=doc_id, owner_id=user.id)
    if doc is None:
        raise HTTPException(404, "文档不存在")
    filename = Path(file.filename or doc.filename).name[:255]
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_TYPES:
        raise HTTPException(400, "仅支持 .pdf / .docx / .xlsx / .md / .txt")
    raw = await file.read(settings.upload_max_bytes + 1)
    await file.close()
    if not raw or len(raw) > settings.upload_max_bytes:
        raise HTTPException(413, "上传文件为空或超过大小上限")
    digest = hashlib.sha256(raw).hexdigest()
    if await Document.filter(kb_id=doc.kb_id, sha256=digest).exclude(id=doc_id).exists():
        raise HTTPException(409, "同内容文件已在知识库中，请使用现有文档")
    path = settings.upload_dir / f"{uuid4()}{suffix}"
    await asyncio.to_thread(path.write_bytes, raw)
    old_path = doc.storage_path
    try:
        async with in_transaction("business") as connection:
            current = (
                await Document.filter(id=doc_id, owner_id=user.id)
                .using_db(connection)
                .select_for_update()
                .first()
            )
            if current is None or current.status not in {"ready", "failed"}:
                raise HTTPException(409, "文档正在处理或已删除")
            old_path = current.storage_path
            kb = await owned_kb(doc.kb_id, user.id)
            await (
                Document.filter(id=doc_id)
                .using_db(connection)
                .update(
                    filename=filename,
                    file_type=suffix[1:],
                    storage_path=str(path.resolve()),
                    sha256=digest,
                    status="queued",
                    error="",
                )
            )
            job = await IngestJob.create(
                owner_id=user.id, doc_id=doc_id, payload={"config": kb.config}, using_db=connection
            )
    except BaseException:
        await asyncio.to_thread(path.unlink, missing_ok=True)
        raise
    await asyncio.to_thread(Path(old_path).unlink, missing_ok=True)
    return {"data": {"document": {"id": doc_id}, "job_id": job.id, "replacement": True}}


def dto(item, fields):
    return {key: getattr(item, key) for key in fields}


APPLICATION_FIELDS = [
    "id",
    "kb_id",
    "name",
    "description",
    "strategy",
    "mode",
    "welcome",
    "fallback",
    "prompt",
    "enabled",
]


async def owned_application(application_id: str, owner_id: str, kb_id: str):
    item = await Application.get_or_none(
        id=application_id, owner_id=owner_id, kb_id=kb_id, enabled=True
    )
    if item is None:
        raise HTTPException(404, "应用不存在、未启用或不属于当前知识库")
    return item


@router.get("/strategies")
async def strategies(user: User = Depends(current_user)):
    return {"data": PRESETS}


@router.get("/applications")
async def applications(kb_id: str, user: User = Depends(current_user)):
    await owned_kb(kb_id, user.id)
    return {
        "data": [
            dto(x, APPLICATION_FIELDS)
            for x in await Application.filter(kb_id=kb_id, owner_id=user.id).order_by("created_at")
        ]
    }


@router.post("/applications", status_code=201)
async def create_application(body: ApplicationInput, user: User = Depends(current_user)):
    await owned_kb(body.kb_id, user.id)
    item = await Application.create(owner_id=user.id, **body.model_dump())
    return {"data": dto(item, APPLICATION_FIELDS)}


@router.put("/applications/{application_id}")
async def update_application(
    application_id: str, body: ApplicationInput, user: User = Depends(current_user)
):
    item = await Application.get_or_none(id=application_id, owner_id=user.id)
    if item is None:
        raise HTTPException(404, "应用不存在")
    # Existing conversations keep their original KB/application binding.
    if item.kb_id != body.kb_id:
        raise HTTPException(409, "更换知识库请创建新应用")
    await Application.filter(id=item.id, owner_id=user.id).update(**body.model_dump())
    return {"data": {"id": item.id, **body.model_dump()}}


@router.put("/runs/{run_id}/feedback")
async def submit_feedback(run_id: str, body: FeedbackInput, user: User = Depends(current_user)):
    run = await AgentRun.get_or_none(id=run_id, owner_id=user.id, status="completed")
    if run is None or not run.conversation_id:
        raise HTTPException(404, "完整对话回答不存在")
    item, _ = await Feedback.update_or_create(
        owner_id=user.id, run_id=run_id, defaults=body.model_dump()
    )
    return {"data": dto(item, ["id", "run_id", "helpful", "comment"])}


@router.get("/feedback")
async def feedback_list(kb_id: str, user: User = Depends(current_user)):
    await owned_kb(kb_id, user.id)
    runs = await AgentRun.filter(owner_id=user.id, kb_id=kb_id, status="completed")
    run_map = {run.id: run for run in runs}
    items = await Feedback.filter(owner_id=user.id, run_id__in=list(run_map)).order_by(
        "-created_at"
    )
    return {
        "data": [
            {
                **dto(item, ["id", "run_id", "helpful", "comment"]),
                "question": run_map[item.run_id].query,
                "answer": run_map[item.run_id].answer,
            }
            for item in items
        ]
    }


@router.get("/datasets")
async def datasets(kb_id: str, user: User = Depends(current_user)):
    await owned_kb(kb_id, user.id)
    return {
        "data": await EvaluationDataset.filter(kb_id=kb_id, owner_id=user.id)
        .order_by("-created_at")
        .values("id", "name", "origin", "cases")
    }


@router.post("/datasets", status_code=201)
async def create_dataset(body: DatasetInput, user: User = Depends(current_user)):
    await owned_kb(body.kb_id, user.id)
    item = await EvaluationDataset.create(owner_id=user.id, **body.model_dump())
    return {"data": dto(item, ["id", "name", "origin", "cases"])}


@router.post("/datasets/generate", status_code=201)
async def generate_dataset(
    body: DatasetGenerate, request: Request, user: User = Depends(current_user)
):
    await owned_kb(body.kb_id, user.id)
    docs = Document.filter(kb_id=body.kb_id, owner_id=user.id, status="ready")
    if body.document_ids:
        docs = docs.filter(id__in=body.document_ids)
    document_map = {d.id: d.filename for d in await docs}
    parents = (
        await ParentChunk.filter(kb_id=body.kb_id, owner_id=user.id, doc_id__in=list(document_map))
        .order_by("doc_id", "ordinal")
        .limit(20)
    )
    if not parents:
        raise HTTPException(409, "请先完成至少一份文档入库")
    sources = []
    used = 0
    for parent in parents:
        content = parent.content[: min(1200, 12000 - used)]
        if not content:
            break
        used += len(content)
        sources.append(
            {
                "source_id": f"S{len(sources) + 1}",
                "document_id": parent.doc_id,
                "filename": document_map[parent.doc_id],
                "content": content,
                "location": parent.metadata.get("location", "全文"),
            }
        )
    async with asyncio.timeout(120):
        generated = await request.app.state.provider.structured(
            GeneratedCases, "dataset", {"count": body.count, "sources": sources}
        )
    if len(generated.cases) != body.count:
        raise ValueError("模型未返回指定数量的测评样本，请重试")
    source_map = {s["source_id"]: s for s in sources}
    cases = []
    for case in generated.cases:
        if not set(case.source_ids) <= source_map.keys():
            raise ValueError("生成样本含无效来源")
        evidence = [source_map[s] for s in case.source_ids]
        if any(
            not fact.strip() or not any(fact in s["content"] for s in evidence)
            for fact in case.reference_facts
        ):
            raise ValueError("生成参考事实并非所引用原文，样本未保存")
        validated = EvaluationCase(
            question=case.question,
            reference_answer=case.reference_answer,
            reference_facts=case.reference_facts,
            relevant_document_ids=list(dict.fromkeys(s["document_id"] for s in evidence)),
        )
        cases.append({**validated.model_dump(), "provenance": evidence})
    item = await EvaluationDataset.create(
        owner_id=user.id, kb_id=body.kb_id, name=body.name, origin="document_generated", cases=cases
    )
    return {"data": dto(item, ["id", "name", "origin", "cases"])}


@router.post("/datasets/from-feedback", status_code=201)
async def feedback_dataset(body: FeedbackDatasetInput, user: User = Depends(current_user)):
    run = await AgentRun.get_or_none(id=body.run_id, owner_id=user.id, status="completed")
    if run is None or not await Feedback.exists(run_id=body.run_id, owner_id=user.id):
        raise HTTPException(404, "反馈或回答不存在")
    case = EvaluationCase(
        question=run.query,
        reference_answer=body.reference_answer,
        reference_facts=body.reference_facts,
    )
    item = await EvaluationDataset.create(
        owner_id=user.id,
        kb_id=run.kb_id,
        name=body.name,
        origin="feedback_corrected",
        cases=[case.model_dump()],
    )
    return {"data": dto(item, ["id", "name", "origin", "cases"])}


@router.post("/evaluations/compare")
async def compare(body: ComparisonRequest, request: Request, user: User = Depends(current_user)):
    kb = await owned_kb(body.kb_id, user.id)
    results = []
    async with asyncio.timeout(600):
        for strategy in ("dense", "hybrid", "full"):
            configured = effective_kb(kb, strategy)
            evaluation = await EvaluationRun.create(
                owner_id=user.id, kb_id=kb.id, provider=request.app.state.settings.model_provider
            )
            try:
                result = await run_evaluation(
                    user.id,
                    configured,
                    body.cases,
                    request.app.state.agent,
                    request.app.state.provider,
                    evaluation,
                    mode="rag",
                )
            except BaseException:
                await EvaluationRun.filter(id=evaluation.id).update(
                    status="failed", error="策略对比中断"
                )
                raise
            results.append({"strategy": strategy, "config": configured.config, **result})
    return {
        "data": {
            "mode": "rag",
            "comparisons": results,
            "demo": request.app.state.settings.model_provider == "demo",
        }
    }


@router.get("/models")
async def model_status(request: Request, user: User = Depends(current_user)):
    settings = request.app.state.settings
    return {
        "data": {
            "provider": settings.model_provider,
            "generation": settings.generation_model,
            "embedding": settings.embedding_model,
            "dimension": settings.embedding_dimension,
            "rerank": settings.rerank_model,
            "credential_configured": bool(settings.dashscope_api_key.get_secret_value()),
            "fingerprint": settings.embedding_fingerprint,
            "configuration_source": ".env / deployment environment",
        }
    }


@router.post("/models/test")
async def model_test(request: Request, user: User = Depends(current_user)):
    provider = request.app.state.provider
    async with asyncio.timeout(120):
        vectors = await provider.embed(["公司设备保修期为24个月。"], "document")
        ranking = await provider.rerank(
            "保修多久？", ["公司设备保修期为24个月。", "今天举行会议。"]
        )
        judgment = await provider.structured(
            Judgment,
            "grade",
            {
                "query": "保修多久？",
                "sources": [{"source_id": "S1", "content": "设备保修期为24个月。"}],
            },
        )
    return {
        "data": {
            "embedding_dimensions": len(vectors[0]),
            "rerank": ranking,
            "generation_structured": judgment.model_dump(),
            "demo": request.app.state.settings.model_provider == "demo",
        }
    }
