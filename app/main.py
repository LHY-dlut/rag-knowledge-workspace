import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from tortoise.exceptions import IntegrityError
from tortoise.expressions import F
from tortoise.transactions import in_transaction

from app.agent import RAGAgent
from app.artifact_api import router as artifact_router
from app.artifacts import list_artifacts
from app.chat_jobs import ChatWorker, finish_error, journal, replay_events
from app.database import close_database, init_database
from app.evaluation import run_evaluation
from app.ingestion import IngestionService, IngestionWorker
from app.management import owned_application
from app.management import router as management_router
from app.models import (
    AgentRun,
    AgentStep,
    ChatMessage,
    Conversation,
    Document,
    EvaluationRun,
    IngestJob,
    KnowledgeBase,
    ParentChunk,
    PromptTemplate,
    RunEvent,
    ToolDefinition,
    User,
)
from app.operations import database_readiness
from app.parsers import SUPPORTED_TYPES
from app.providers import ProviderError, make_provider
from app.retrieval import AdvancedRetrieverPipeline
from app.schemas import (
    ChatRequest,
    ChunkEdit,
    Credentials,
    EvaluationRequest,
    KBCreate,
    PromptEdit,
    QueryRequest,
    RetrievalConfig,
)
from app.security import current_user, hash_password, issue_token, owned_kb, verify_password
from app.settings import Settings
from app.strategies import effective_kb, expanded_queries
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ok(data):
    return {"data": data}


def document_dto(doc: Document):
    return {
        k: getattr(doc, k)
        for k in (
            "id",
            "filename",
            "file_type",
            "status",
            "error",
            "parent_count",
            "child_count",
            "tags",
            "embedding_fingerprint",
            "index_revision",
        )
    }


def create_app(settings: Settings | None = None, *, start_worker: bool | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_database(settings, create_schema=settings.app_mode == "demo")
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        provider = make_provider(settings)
        store = VectorStore(settings)
        retriever = AdvancedRetrieverPipeline(store, provider, settings)
        ingestion = IngestionService(store, provider, settings)
        worker = IngestionWorker(ingestion, settings)
        app.state.settings, app.state.provider, app.state.store = settings, provider, store
        app.state.retriever, app.state.agent, app.state.worker = (
            retriever,
            RAGAgent(provider, retriever),
            worker,
        )
        local_workers = (
            settings.app_mode == "demo"
            if settings.enable_api_workers is None
            else settings.enable_api_workers
        )
        chat_worker = ChatWorker(app.state.agent, settings)
        app.state.chat_worker = chat_worker
        chat_task = asyncio.create_task(chat_worker.run()) if local_workers else None
        run_worker = local_workers if start_worker is None else start_worker
        task = asyncio.create_task(worker.run()) if run_worker else None
        try:
            yield
        finally:
            if task:
                worker.stopping.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if chat_task:
                chat_worker.stopping.set()
                chat_task.cancel()
                await asyncio.gather(chat_task, return_exceptions=True)
            await provider.close()
            await close_database()

    app = FastAPI(title="知序 Agentic RAG 知识工作台", version="1.81.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID"],
    )

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(
            {"error": {"code": exc.status_code, "message": exc.detail}}, status_code=exc.status_code
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        errors = [{"loc": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            {"error": {"code": 422, "message": "输入格式不符合要求", "details": errors}},
            status_code=422,
        )

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"error": {"code": 400, "message": str(exc)}}, status_code=400)

    @app.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse({"error": {"code": 502, "message": str(exc)}}, status_code=502)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logger.exception("Unhandled API error", exc_info=exc)
        return JSONResponse(
            {"error": {"code": 500, "message": "服务内部错误，请查看服务日志"}}, status_code=500
        )

    @app.get("/api/health")
    async def health():
        return ok({"status": "ok", "mode": settings.app_mode, "provider": settings.model_provider})

    @app.get("/api/live")
    async def live():
        return ok({"status": "ok"})

    @app.get("/api/ready")
    async def ready():
        databases = await database_readiness()
        available = all(databases.values())
        return JSONResponse(
            ok({"status": "ready" if available else "unavailable", "databases": databases}),
            status_code=200 if available else 503,
        )

    @app.post("/api/auth/register", status_code=201)
    async def register(body: Credentials):
        if not settings.enable_registration:
            raise HTTPException(403, "注册已关闭")
        try:
            user = await User.create(
                username=body.username, password_hash=await hash_password(body.password)
            )
        except IntegrityError:
            raise HTTPException(409, "用户名已存在") from None
        return ok(
            {
                "access_token": issue_token(user.id, settings),
                "token_type": "bearer",
                "username": user.username,
            }
        )

    @app.post("/api/auth/login")
    async def login(body: Credentials):
        user = await User.get_or_none(username=body.username, is_active=True)
        if user is None or not await verify_password(user.password_hash, body.password):
            raise HTTPException(401, "用户名或密码错误")
        return ok(
            {
                "access_token": issue_token(user.id, settings),
                "token_type": "bearer",
                "username": user.username,
            }
        )

    @app.get("/api/knowledge-bases")
    async def list_kbs(user: User = Depends(current_user)):
        return ok(
            await KnowledgeBase.filter(owner_id=user.id)
            .order_by("created_at")
            .values("id", "name", "description", "config", "revision")
        )

    @app.post("/api/knowledge-bases", status_code=201)
    async def create_kb(body: KBCreate, user: User = Depends(current_user)):
        kb = await KnowledgeBase.create(owner_id=user.id, **body.model_dump())
        return ok(
            {"id": kb.id, "name": kb.name, "description": kb.description, "config": kb.config}
        )

    @app.put("/api/knowledge-bases/{kb_id}/config")
    async def update_config(kb_id: str, body: RetrievalConfig, user: User = Depends(current_user)):
        await owned_kb(kb_id, user.id)
        await KnowledgeBase.filter(id=kb_id, owner_id=user.id).update(
            config=body.model_dump(), revision=F("revision") + 1
        )
        return ok(body.model_dump())

    @app.get("/api/knowledge-bases/{kb_id}/documents")
    async def list_documents(kb_id: str, user: User = Depends(current_user)):
        await owned_kb(kb_id, user.id)
        return ok(
            [
                document_dto(doc)
                for doc in await Document.filter(kb_id=kb_id, owner_id=user.id).all()
            ]
        )

    @app.post("/api/knowledge-bases/{kb_id}/documents", status_code=202)
    async def upload_document(
        kb_id: str,
        request: Request,
        file: UploadFile = File(...),
        tags: str = Form(""),
        user: User = Depends(current_user),
    ):
        kb = await owned_kb(kb_id, user.id)
        filename = Path(file.filename or "document.txt").name[:255]
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_TYPES:
            raise HTTPException(400, "仅支持 .pdf / .docx / .xlsx / .md / .txt")
        raw = await file.read(settings.upload_max_bytes + 1)
        await file.close()
        if not raw or len(raw) > settings.upload_max_bytes:
            raise HTTPException(413, "上传文件为空或超过大小上限")
        digest = hashlib.sha256(raw).hexdigest()
        existing = await Document.get_or_none(kb_id=kb_id, owner_id=user.id, sha256=digest)
        if existing:
            return ok({"document": document_dto(existing), "duplicate": True})
        parsed_tags = [t.strip()[:100] for t in tags.split(",") if t.strip()][:20]
        path = settings.upload_dir / f"{uuid4()}{suffix}"
        await asyncio.to_thread(path.write_bytes, raw)
        try:
            async with in_transaction("business") as connection:
                doc = await Document.create(
                    owner_id=user.id,
                    kb_id=kb_id,
                    filename=filename,
                    file_type=suffix[1:],
                    storage_path=str(path.resolve()),
                    sha256=digest,
                    tags=parsed_tags,
                    using_db=connection,
                )
                job = await IngestJob.create(
                    owner_id=user.id,
                    doc_id=doc.id,
                    payload={"config": kb.config},
                    using_db=connection,
                )
        except BaseException:
            await asyncio.to_thread(path.unlink, missing_ok=True)
            raise
        return ok({"document": document_dto(doc), "job_id": job.id, "duplicate": False})

    @app.post("/api/documents/{doc_id}/reindex", status_code=202)
    async def reindex(doc_id: str, user: User = Depends(current_user)):
        async with in_transaction("business") as connection:
            doc = (
                await Document.filter(id=doc_id, owner_id=user.id)
                .using_db(connection)
                .select_for_update()
                .first()
            )
            if doc is None:
                raise HTTPException(404, "文档不存在")
            if doc.status not in {"ready", "failed"}:
                raise HTTPException(409, "文档正在处理")
            kb = await owned_kb(doc.kb_id, user.id)
            await Document.filter(id=doc.id).using_db(connection).update(status="queued", error="")
            job = await IngestJob.create(
                owner_id=user.id, doc_id=doc.id, payload={"config": kb.config}, using_db=connection
            )
        return ok({"job_id": job.id})

    @app.delete("/api/documents/{doc_id}")
    async def delete_doc(doc_id: str, request: Request, user: User = Depends(current_user)):
        doc = await Document.get_or_none(id=doc_id, owner_id=user.id)
        if doc is None:
            raise HTTPException(404, "文档不存在")
        async with in_transaction("business") as connection:
            await Document.filter(id=doc_id).using_db(connection).select_for_update().first()
            await Document.filter(id=doc_id).using_db(connection).update(status="deleting")
        await KnowledgeBase.filter(id=doc.kb_id, owner_id=user.id).update(
            revision=F("revision") + 1
        )
        await request.app.state.store.delete_document(doc_id)
        async with in_transaction("business") as connection:
            await ParentChunk.filter(doc_id=doc_id).using_db(connection).delete()
            await IngestJob.filter(doc_id=doc_id).using_db(connection).delete()
            await Document.filter(id=doc_id, owner_id=user.id).using_db(connection).delete()
        await asyncio.to_thread(Path(doc.storage_path).unlink, missing_ok=True)
        return ok({"deleted": True})

    @app.get("/api/documents/{doc_id}/chunks")
    async def list_chunks(doc_id: str, user: User = Depends(current_user)):
        if not await Document.exists(id=doc_id, owner_id=user.id):
            raise HTTPException(404, "文档不存在")
        return ok(
            await ParentChunk.filter(doc_id=doc_id, owner_id=user.id)
            .order_by("ordinal")
            .values("id", "ordinal", "content", "metadata", "edited")
        )

    @app.put("/api/chunks/{parent_id}", status_code=202)
    async def edit_chunk(parent_id: str, body: ChunkEdit, user: User = Depends(current_user)):
        parent = await ParentChunk.get_or_none(id=parent_id, owner_id=user.id)
        if parent is None:
            raise HTTPException(404, "片段不存在")
        async with in_transaction("business") as connection:
            doc = (
                await Document.filter(id=parent.doc_id, owner_id=user.id)
                .using_db(connection)
                .select_for_update()
                .first()
            )
            if doc is None or doc.status != "ready":
                raise HTTPException(409, "请等待文档入库完成")
            kb = await owned_kb(doc.kb_id, user.id)
            if doc.embedding_fingerprint != settings.embedding_fingerprint:
                raise HTTPException(409, "模型已切换，请先完整重建索引")
            await Document.filter(id=doc.id).using_db(connection).update(status="queued")
            job = await IngestJob.create(
                owner_id=user.id,
                doc_id=doc.id,
                kind="edit_parent",
                payload={"parent_id": parent_id, "content": body.content, "config": kb.config},
                using_db=connection,
            )
        return ok({"job_id": job.id})

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, user: User = Depends(current_user)):
        job = await IngestJob.get_or_none(id=job_id, owner_id=user.id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return ok({k: getattr(job, k) for k in ("id", "status", "error", "attempts", "doc_id")})

    @app.post("/api/retrieval/debug")
    async def retrieval_debug(
        body: QueryRequest, request: Request, user: User = Depends(current_user)
    ):
        kb = effective_kb(await owned_kb(body.kb_id, user.id), body.strategy)
        from app.schemas import RewriteDecision

        config = RetrievalConfig.model_validate(kb.config)
        rewrite = RewriteDecision(standalone_query=body.query, queries=[body.query])
        if config.query_rewrite or config.multi_query or config.hyde:
            rewrite = await request.app.state.provider.structured(
                RewriteDecision,
                "rewrite",
                {
                    "query": body.query,
                    "history": [],
                    "multi_query": config.multi_query,
                    "generate_hyde": config.hyde,
                    "resolve_coreference": config.query_rewrite,
                },
            )
        query = rewrite.standalone_query if config.query_rewrite else body.query
        result = await request.app.state.retriever.retrieve(
            owner_id=user.id,
            kb=kb,
            original_query=query,
            queries=expanded_queries(body.query, query, rewrite.queries, config.multi_query),
            hypothetical_document=rewrite.hypothetical_document if config.hyde else "",
            filters=body.filters,
        )
        return ok(result.model_dump(exclude={"candidates": {"__all__": {"embedding"}}}))

    @app.get("/api/conversations")
    async def conversations(
        kb_id: str, application_id: str | None = None, user: User = Depends(current_user)
    ):
        await owned_kb(kb_id, user.id)
        items = Conversation.filter(kb_id=kb_id, owner_id=user.id)
        if application_id is not None:
            items = items.filter(application_id=application_id)
        return ok(
            await items.order_by("-created_at").values(
                "id", "title", "application_id", "active_run_id"
            )
        )

    @app.get("/api/conversations/{conversation_id}/messages")
    async def messages(conversation_id: str, user: User = Depends(current_user)):
        if not await Conversation.exists(id=conversation_id, owner_id=user.id):
            raise HTTPException(404, "会话不存在")
        items = (
            await ChatMessage.filter(
                conversation_id=conversation_id, owner_id=user.id, accepted=True
            )
            .order_by("created_at")
            .values("role", "content", "citations", "run_id")
        )
        for item in items:
            if item["role"] == "assistant":
                item["artifacts"] = await list_artifacts(user.id, item["run_id"])
        return ok(items)

    async def submit_chat(body: ChatRequest, user: User):
        kb = effective_kb(await owned_kb(body.kb_id, user.id), body.strategy)
        application = None
        mode = body.mode
        if body.application_id:
            application = await owned_application(body.application_id, user.id, kb.id)
            kb = effective_kb(kb, application.strategy)
            mode = application.mode
        if body.conversation_id:
            conversation = await Conversation.get_or_none(
                id=body.conversation_id,
                owner_id=user.id,
                kb_id=kb.id,
                application_id=body.application_id or "",
            )
            if conversation is None:
                raise HTTPException(404, "会话不存在")
        else:
            conversation = await Conversation.create(
                owner_id=user.id,
                kb_id=kb.id,
                title=body.query[:120],
                application_id=body.application_id or "",
            )
        run_id = str(uuid4())
        now = datetime.now(UTC)
        async with in_transaction("business") as connection:
            current = (
                await Conversation.filter(id=conversation.id)
                .using_db(connection)
                .select_for_update()
                .first()
            )
            if current.active_run_id:
                active = (
                    await AgentRun.filter(id=current.active_run_id).using_db(connection).first()
                )
                if active and active.status in {"queued", "running"}:
                    raise HTTPException(409, "同一会话已有回答正在生成")
            history = list(
                reversed(
                    await ChatMessage.filter(
                        conversation_id=conversation.id, owner_id=user.id, accepted=True
                    )
                    .using_db(connection)
                    .order_by("-created_at")
                    .limit(12)
                    .values("role", "content")
                )
            )
            snapshot = {
                "history": history,
                "filters": body.filters.model_dump(),
                "mode": mode,
                "output_type": body.output_type,
            }
            if application:
                snapshot.update(
                    application_prompt=application.prompt, fallback_answer=application.fallback
                )
            run = await AgentRun.create(
                id=run_id,
                owner_id=user.id,
                kb_id=kb.id,
                conversation_id=conversation.id,
                query=body.query,
                config_snapshot=kb.config,
                status="queued",
                request_snapshot=snapshot,
                using_db=connection,
            )
            await ChatMessage.create(
                owner_id=user.id,
                conversation_id=conversation.id,
                run_id=run.id,
                role="user",
                content=body.query,
                accepted=False,
                using_db=connection,
            )
            current.active_run_id = run.id
            current.active_run_until = now + timedelta(
                seconds=settings.chat_run_timeout_seconds + 30
            )
            await current.save(
                using_db=connection, update_fields=["active_run_id", "active_run_until"]
            )
            await journal(
                run,
                "meta",
                {
                    "run_id": run.id,
                    "conversation_id": conversation.id,
                    "provider": settings.model_provider,
                    "demo": settings.model_provider == "demo",
                    "application_id": body.application_id,
                    "mode": mode,
                },
                connection,
            )
        return {"run_id": run.id, "conversation_id": conversation.id, "status": "queued"}

    def event_response(run_id, after=0):
        return EventSourceResponse(
            replay_events(run_id, after),
            ping=15,
            send_timeout=30,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/chat/runs", status_code=202)
    async def submit(body: ChatRequest, user: User = Depends(current_user)):
        return ok(await submit_chat(body, user))

    @app.post("/api/chat/stream")
    async def chat(body: ChatRequest, user: User = Depends(current_user)):
        result = await submit_chat(body, user)
        return event_response(result["run_id"])

    @app.get("/api/runs/{run_id}/events")
    async def events(
        run_id: str, request: Request, after: int = 0, user: User = Depends(current_user)
    ):
        run = await AgentRun.get_or_none(id=run_id, owner_id=user.id)
        if run is None:
            raise HTTPException(404, "运行记录不存在")
        cursor = request.headers.get("last-event-id")
        if cursor:
            try:
                cursor_run, number = cursor.rsplit(":", 1)
                if cursor_run != run_id:
                    raise ValueError()
                after = int(number)
            except ValueError:
                raise HTTPException(400, "无效事件序号") from None
        if after < 0 or after > run.event_sequence:
            raise HTTPException(400, "事件序号超出范围")
        return event_response(run.id, after)

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel(run_id: str, user: User = Depends(current_user)):
        if not await AgentRun.exists(id=run_id, owner_id=user.id):
            raise HTTPException(404, "运行记录不存在")
        await finish_error(run_id, "cancelled", "用户已取消生成")
        return ok({"status": (await AgentRun.get(id=run_id)).status})

    @app.get("/api/runs/{run_id}")
    async def trace(run_id: str, user: User = Depends(current_user)):
        run = await AgentRun.get_or_none(id=run_id, owner_id=user.id)
        if run is None:
            raise HTTPException(404, "运行记录不存在")
        terminal = await RunEvent.filter(run_id=run.id, name="done").first()
        rejected = terminal.data.get("rejected") if terminal else None
        return ok(
            {
                "id": run.id,
                "conversation_id": run.conversation_id,
                "query": run.query,
                "event_sequence": run.event_sequence,
                "elapsed_ms": run.elapsed_ms,
                "status": run.status,
                "answer": run.answer,
                "rejected": rejected if isinstance(rejected, bool) else None,
                "citations": run.citations,
                "artifacts": await list_artifacts(user.id, run.id),
                "grade_retries": run.grade_retries,
                "check_retries": run.check_retries,
                "error": run.error,
                "steps": await AgentStep.filter(run_id=run.id)
                .order_by("ordinal")
                .values(
                    "node", "ordinal", "status", "input_summary", "output_summary", "elapsed_ms"
                ),
            }
        )

    @app.get("/api/tools")
    async def list_tools(user: User = Depends(current_user)):
        disabled = set(
            await ToolDefinition.filter(owner_id=user.id, enabled=False).values_list(
                "name", flat=True
            )
        )
        return ok(
            [
                {
                    "name": "search_knowledge",
                    "enabled": True,
                    "description": "复用当前授权知识库检索管道",
                },
                {
                    "name": "calculator",
                    "enabled": "calculator" not in disabled,
                    "description": "安全算术工具",
                },
                {
                    "name": "current_time",
                    "enabled": "current_time" not in disabled,
                    "description": "读取上海或 UTC 当前系统时间",
                },
            ]
        )

    @app.put("/api/tools/calculator")
    async def toggle_calculator(enabled: bool, user: User = Depends(current_user)):
        tool, _ = await ToolDefinition.get_or_create(
            owner_id=user.id, name="calculator", defaults={"enabled": enabled}
        )
        tool.enabled = enabled
        await tool.save(update_fields=["enabled"])
        return ok({"name": tool.name, "enabled": enabled})

    @app.get("/api/prompts/generation")
    async def get_prompt(user: User = Depends(current_user)):
        prompt = await PromptTemplate.get_or_none(owner_id=user.id, name="generation")
        return ok({"content": prompt.content if prompt else "用中文回答，先给结论，再列依据。"})

    @app.put("/api/tools/current_time")
    async def toggle_clock(enabled: bool, user: User = Depends(current_user)):
        tool, _ = await ToolDefinition.get_or_create(
            owner_id=user.id, name="current_time", defaults={"enabled": enabled}
        )
        tool.enabled = enabled
        await tool.save(update_fields=["enabled"])
        return ok({"name": tool.name, "enabled": enabled})

    @app.put("/api/prompts/generation")
    async def edit_prompt(body: PromptEdit, user: User = Depends(current_user)):
        await PromptTemplate.update_or_create(
            owner_id=user.id, name="generation", defaults={"content": body.content}
        )
        return ok(body.model_dump())

    @app.post("/api/evaluations")
    async def evaluate(
        body: EvaluationRequest, request: Request, user: User = Depends(current_user)
    ):
        kb = await owned_kb(body.kb_id, user.id)
        evaluation = await EvaluationRun.create(
            owner_id=user.id, kb_id=kb.id, provider=settings.model_provider
        )
        try:
            async with asyncio.timeout(300):
                result = await run_evaluation(
                    user.id,
                    kb,
                    body.cases,
                    request.app.state.agent,
                    request.app.state.provider,
                    evaluation,
                    mode=body.mode,
                )
        except BaseException:
            await EvaluationRun.filter(id=evaluation.id).update(
                status="failed", error="评测失败或超时；已保留完成样本"
            )
            raise
        return ok(result)

    @app.get("/api/evaluations")
    async def list_evaluations(kb_id: str, user: User = Depends(current_user)):
        await owned_kb(kb_id, user.id)
        return ok(
            await EvaluationRun.filter(owner_id=user.id, kb_id=kb_id)
            .order_by("-created_at")
            .values("id", "status", "metrics", "provider", "error")
        )

    @app.get("/api/evaluations/{evaluation_id}")
    async def get_evaluation(evaluation_id: str, user: User = Depends(current_user)):
        item = await EvaluationRun.get_or_none(id=evaluation_id, owner_id=user.id)
        if item is None:
            raise HTTPException(404, "评测不存在")
        return ok(
            {
                "id": item.id,
                "status": item.status,
                "results": item.results,
                "metrics": item.metrics,
                "provider": item.provider,
                "error": item.error,
            }
        )

    app.include_router(management_router)
    app.include_router(artifact_router)
    return app


app = create_app()
