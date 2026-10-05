import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from app.artifact_storage import artifact_transaction
from app.artifacts import (
    ArtifactUnavailable,
    artifact_dto,
    build_payload,
    digest,
    file_path,
    list_artifacts,
    persist_artifact,
)
from app.chat_jobs import journal
from app.models import AgentRun, AgentStep, ChatMessage, Document, GeneratedArtifact, RunEvent, User
from app.schemas import StrictModel
from app.security import current_user, owned_kb

router = APIRouter(prefix="/api")


class ArtifactRequest(StrictModel):
    type: Literal["chart", "report", "webpage"]


@router.get("/runs/{run_id}/artifacts")
async def artifacts_for_run(run_id: str, user: User = Depends(current_user)):
    run = await AgentRun.get_or_none(id=run_id, owner_id=user.id)
    if not run:
        raise HTTPException(404, "运行记录不存在")
    await owned_kb(run.kb_id, user.id)
    return {"data": await list_artifacts(user.id, run_id)}


@router.post("/runs/{run_id}/artifacts")
async def create_artifact(
    run_id: str, body: ArtifactRequest, request: Request, user: User = Depends(current_user)
):
    started = time.perf_counter()
    run = await AgentRun.get_or_none(id=run_id, owner_id=user.id)
    if not run:
        raise HTTPException(404, "运行记录不存在")
    await owned_kb(run.kb_id, user.id)
    async with artifact_transaction(request.app.state.settings) as (connection, pending_files):
        documents = (
            await Document.filter(
                owner_id=user.id, id__in=sorted({s["document_id"] for s in run.citations})
            )
            .using_db(connection)
            .select_for_update()
            .order_by("id")
        )
        current = (
            await AgentRun.filter(id=run_id, owner_id=user.id)
            .using_db(connection)
            .select_for_update()
            .first()
        )
        done = await RunEvent.filter(run_id=run_id, name="done").using_db(connection).first()
        accepted = (
            await ChatMessage.filter(
                run_id=run_id, owner_id=user.id, role="assistant", accepted=True
            )
            .using_db(connection)
            .exists()
        )
        if (
            not current
            or current.status != "completed"
            or not accepted
            or not done
            or done.data.get("rejected") is not False
        ):
            raise HTTPException(409, "仅可转换已接受且未拒答的核验答案")
        live = {
            d.id: d.index_revision
            for d in documents
            if d.status == "ready"
            and d.embedding_fingerprint == request.app.state.settings.embedding_fingerprint
        }
        if not current.citations or any(
            live.get(s["document_id"]) != s["document_revision"] for s in current.citations
        ):
            raise HTTPException(409, "来源版本已改变，请重新提问后生成成果")
        step = (
            await AgentStep.filter(run_id=run_id, node="check", status="completed")
            .using_db(connection)
            .order_by("-ordinal")
            .first()
        )
        check = step.output_summary.get("check", {}) if step else {}
        try:
            payload = build_payload(
                body.type,
                current.query,
                current.answer,
                current.citations,
                check,
                request.app.state.settings.model_provider,
            )
            item, created = await persist_artifact(
                current, body.type, payload, request.app.state.settings, connection, pending_files
            )
        except ArtifactUnavailable as error:
            raise HTTPException(409, str(error)) from None
        if created:
            await journal(current, "artifact_ready", {"artifact": artifact_dto(item)}, connection)
            last = (
                await AgentStep.filter(run_id=run_id)
                .using_db(connection)
                .order_by("-ordinal")
                .first()
            )
            await AgentStep.create(
                run_id=run_id,
                node="artifact",
                ordinal=(last.ordinal if last else 0) + 1,
                status="completed",
                input_summary={"output_type": body.type},
                output_summary={"id": item.id, "type": body.type, "conversion": True},
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                using_db=connection,
            )
        return {"data": {**artifact_dto(item), "created": created}}


@router.get("/artifacts/{artifact_id}/download")
async def download_artifact(artifact_id: str, request: Request, user: User = Depends(current_user)):
    item = await GeneratedArtifact.get_or_none(id=artifact_id, owner_id=user.id)
    if not item:
        raise HTTPException(404, "成果不存在")
    await owned_kb(item.kb_id, user.id)
    path = file_path(request.app.state.settings, item.id, item.kind)
    if (
        path.name != item.storage_path
        or not path.is_file()
        or digest(path.read_bytes()) != item.file_sha256
    ):
        raise HTTPException(409, "成果文件缺失或校验失败")
    return FileResponse(
        path,
        filename=f"rag-{item.kind}-{item.id[:8]}{path.suffix}",
        media_type="application/json" if item.kind == "chart" else "text/html; charset=utf-8",
        headers={
            "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )
