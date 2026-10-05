import asyncio
import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from tortoise.expressions import F
from tortoise.transactions import in_transaction

from app.chunking import children_for_parent, parent_child_chunks
from app.models import Document, IngestJob, KnowledgeBase, ParentChunk
from app.parsers import parse_document
from app.providers import ModelProvider
from app.schemas import ChildRecord, RetrievalConfig
from app.settings import Settings
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestLeaseLost(RuntimeError):
    pass


async def lock_execution(job, connection):
    doc = await Document.filter(id=job.doc_id).using_db(connection).select_for_update().first()
    current = await IngestJob.filter(id=job.id).using_db(connection).select_for_update().first()
    if (
        not doc
        or doc.status == "deleting"
        or not current
        or current.status != "processing"
        or current.lease_owner != job.lease_owner
        or current.attempts != job.attempts
        or not current.lease_until
        or current.lease_until <= datetime.now(UTC)
    ):
        raise IngestLeaseLost()
    return doc


class IngestionService:
    def __init__(self, store: VectorStore, provider: ModelProvider, settings: Settings):
        self.store, self.provider, self.settings = store, provider, settings

    async def _records(self, doc: Document, parents, config: RetrievalConfig):
        children = []
        for parent in parents:
            slices = (
                parent.children
                if hasattr(parent, "children")
                else children_for_parent(parent.id, parent.content, config)
            )
            for child_id, text, start, end in slices:
                children.append((parent, child_id, text, start, end))
        vectors = await self.provider.embed([c[2] for c in children], "document")
        return [
            ChildRecord(
                id=child_id,
                owner_id=doc.owner_id,
                kb_id=doc.kb_id,
                doc_id=doc.id,
                parent_id=parent.id,
                content=text,
                embedding=vector,
                embedding_fingerprint=self.provider.fingerprint,
                metadata={
                    **parent.metadata,
                    "child_start": start,
                    "child_end": end,
                    "file_type": doc.file_type,
                    "tags": doc.tags,
                },
            )
            for (parent, child_id, text, start, end), vector in zip(children, vectors, strict=True)
        ]

    async def execute(self, job: IngestJob):
        doc = await Document.get_or_none(id=job.doc_id, owner_id=job.owner_id)
        if doc is None or doc.status == "deleting":
            return
        kb = await KnowledgeBase.get(id=doc.kb_id, owner_id=doc.owner_id)
        config = RetrievalConfig.model_validate(job.payload.get("config", kb.config))
        async with in_transaction("business") as connection:
            await lock_execution(job, connection)
            await (
                Document.filter(id=doc.id)
                .using_db(connection)
                .update(status="processing", error="")
            )
        if job.kind == "edit_parent":
            parent = await ParentChunk.get(
                id=job.payload["parent_id"], doc_id=doc.id, owner_id=doc.owner_id
            )
            parent.content = job.payload["content"]
            records = await self._records(doc, [parent], config)
            async with in_transaction("business") as connection:
                await lock_execution(job, connection)
                await self.store.replace_parent(parent.id, records)
                await lock_execution(job, connection)
                parent.edited = True
                await parent.save(using_db=connection, update_fields=["content", "edited"])
                # Count vector table after committing the vector transaction.
                from app.vector_models import ChunkVector

                count = await ChunkVector.filter(doc_id=doc.id).count()
                await (
                    Document.filter(id=doc.id)
                    .using_db(connection)
                    .update(
                        status="ready",
                        child_count=count,
                        embedding_fingerprint=self.provider.fingerprint,
                        index_revision=F("index_revision") + 1,
                    )
                )
                await (
                    KnowledgeBase.filter(id=kb.id)
                    .using_db(connection)
                    .update(revision=F("revision") + 1)
                )
            return
        units = await asyncio.to_thread(
            parse_document,
            __import__("pathlib").Path(doc.storage_path),
            max_chars=self.settings.parsed_max_chars,
            ocr=self.settings.enable_ocr,
        )
        parents = parent_child_chunks(doc.id, units, config)
        records = await self._records(doc, parents, config)
        # Vector write first. Ready-only retrieval blocks partially published data.
        async with in_transaction("business") as connection:
            await lock_execution(job, connection)
            await self.store.replace_document(doc.id, records)
            await lock_execution(job, connection)
            await ParentChunk.filter(doc_id=doc.id).using_db(connection).delete()
            await ParentChunk.bulk_create(
                [
                    ParentChunk(
                        id=p.id,
                        owner_id=doc.owner_id,
                        kb_id=doc.kb_id,
                        doc_id=doc.id,
                        ordinal=p.ordinal,
                        content=p.content,
                        metadata=p.metadata,
                    )
                    for p in parents
                ],
                using_db=connection,
                batch_size=100,
            )
            await (
                Document.filter(id=doc.id)
                .using_db(connection)
                .update(
                    status="ready",
                    error="",
                    parent_count=len(parents),
                    child_count=len(records),
                    embedding_fingerprint=self.provider.fingerprint,
                    index_revision=F("index_revision") + 1,
                )
            )
            await (
                KnowledgeBase.filter(id=kb.id)
                .using_db(connection)
                .update(revision=F("revision") + 1)
            )


class IngestionWorker:
    def __init__(self, service: IngestionService, settings: Settings):
        self.service, self.settings = service, settings
        self.id = str(uuid4())
        self.stopping = asyncio.Event()

    async def _heartbeat(self, job_id: str):
        while True:
            await asyncio.sleep(self.settings.worker_lease_seconds / 3)
            async with in_transaction("business") as connection:
                job = (
                    await IngestJob.filter(id=job_id)
                    .using_db(connection)
                    .select_for_update()
                    .first()
                )
                now = datetime.now(UTC)
                if (
                    not job
                    or job.status != "processing"
                    or job.lease_owner != self.id
                    or not job.lease_until
                    or job.lease_until <= now
                ):
                    return
                job.lease_until = now + timedelta(seconds=self.settings.worker_lease_seconds)
                await job.save(using_db=connection, update_fields=["lease_until"])

    async def tick(self) -> bool:
        now = datetime.now(UTC)
        expired = await IngestJob.filter(status="processing", lease_until__lt=now).all()
        for old in expired:
            async with in_transaction("business") as connection:
                await (
                    Document.filter(id=old.doc_id).using_db(connection).select_for_update().first()
                )
                current = (
                    await IngestJob.filter(id=old.id)
                    .using_db(connection)
                    .select_for_update()
                    .first()
                )
                if (
                    not current
                    or current.status != "processing"
                    or not current.lease_until
                    or current.lease_until >= datetime.now(UTC)
                ):
                    continue
                if current.attempts >= 3:
                    await (
                        IngestJob.filter(id=old.id)
                        .using_db(connection)
                        .update(status="failed", error="任务租约连续过期")
                    )
                    await (
                        Document.filter(id=old.doc_id, status="processing")
                        .using_db(connection)
                        .update(status="failed", error="任务租约连续过期，请重试")
                    )
                else:
                    await (
                        IngestJob.filter(id=old.id)
                        .using_db(connection)
                        .update(status="queued", lease_owner="")
                    )
        job = await IngestJob.filter(status="queued").order_by("created_at").first()
        if job is None:
            return False
        async with in_transaction("business") as connection:
            await Document.filter(id=job.doc_id).using_db(connection).select_for_update().first()
            job = await IngestJob.filter(id=job.id).using_db(connection).select_for_update().first()
            if job is None or job.status != "queued":
                return False
            job.status, job.lease_owner = "processing", self.id
            job.lease_until = datetime.now(UTC) + timedelta(
                seconds=self.settings.worker_lease_seconds
            )
            job.attempts += 1
            await job.save(
                using_db=connection,
                update_fields=["status", "lease_owner", "lease_until", "attempts"],
            )
        heartbeat = asyncio.create_task(self._heartbeat(job.id))
        try:
            await self.service.execute(job)
            async with in_transaction("business") as connection:
                await lock_execution(job, connection)
                await (
                    IngestJob.filter(id=job.id)
                    .using_db(connection)
                    .update(status="done", lease_until=None)
                )
        except asyncio.CancelledError:
            # Keep lease: the next worker safely recovers it after expiry.
            raise
        except IngestLeaseLost:
            # A recovered worker owns this job now. Never mutate its publication.
            pass
        except Exception as exc:
            logger.exception("Ingest job failed: %s", job.id)
            error = str(exc)[:1000]
            async with in_transaction("business") as connection:
                try:
                    await lock_execution(job, connection)
                except IngestLeaseLost:
                    pass
                else:
                    await (
                        IngestJob.filter(id=job.id)
                        .using_db(connection)
                        .update(status="failed", error=error, lease_until=None)
                    )
                    await (
                        Document.filter(id=job.doc_id)
                        .using_db(connection)
                        .update(status="failed", error=error)
                    )
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        return True

    async def run(self):
        while not self.stopping.is_set():
            if not await self.tick():
                try:
                    await asyncio.wait_for(self.stopping.wait(), self.settings.worker_poll_seconds)
                except TimeoutError:
                    pass
