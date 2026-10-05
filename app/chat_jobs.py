"""Durable chat execution and SSE journal. HTTP subscribers never own execution."""

import asyncio
import json
import re
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from tortoise.transactions import in_transaction

from app.agent import FALLBACK_ANSWER
from app.artifact_storage import artifact_transaction
from app.artifacts import ArtifactUnavailable, artifact_dto, build_payload, persist_artifact
from app.models import (
    AgentRun,
    AgentStep,
    ChatMessage,
    Conversation,
    Document,
    KnowledgeBase,
    RunEvent,
)
from app.providers import ProviderError
from app.schemas import MetadataFilter

TERMINAL = {"completed", "failed", "cancelled"}


class LeaseLost(RuntimeError):
    pass


async def journal(run, name, data, connection):
    run.event_sequence += 1
    await RunEvent.create(
        run_id=run.id, sequence=run.event_sequence, name=name, data=data, using_db=connection
    )
    await run.save(using_db=connection, update_fields=["event_sequence"])


async def finish_error(run_id, status, message, *, lease_owner=None, expired_only=False):
    snapshot = await AgentRun.get_or_none(id=run_id)
    if snapshot is None:
        return
    async with in_transaction("business") as conn:
        conversation = (
            await Conversation.filter(id=snapshot.conversation_id)
            .using_db(conn)
            .select_for_update()
            .first()
        )
        run = await AgentRun.filter(id=run_id).using_db(conn).select_for_update().first()
        if run is None or run.status in TERMINAL:
            return
        if lease_owner is not None and run.lease_owner != lease_owner:
            return
        if expired_only and (not run.lease_until or run.lease_until >= datetime.now(UTC)):
            return
        run.status, run.error, run.lease_until = status, message, None
        await run.save(using_db=conn, update_fields=["status", "error", "lease_until"])
        await journal(run, "draft_reset", {"reason": message}, conn)
        await journal(run, "error", {"message": message, "status": status}, conn)
        if conversation and conversation.active_run_id == run.id:
            conversation.active_run_id, conversation.active_run_until = "", None
            await conversation.save(
                using_db=conn, update_fields=["active_run_id", "active_run_until"]
            )


async def replay_events(run_id, after=0):
    sequence = after
    while True:
        events = (
            await RunEvent.filter(run_id=run_id, sequence__gt=sequence)
            .order_by("sequence")
            .limit(200)
        )
        for event in events:
            sequence = event.sequence
            yield {
                "event": event.name,
                "data": json.dumps(event.data, ensure_ascii=False),
                "id": f"{run_id}:{sequence}",
            }
        run = await AgentRun.get(id=run_id)
        if run.status in TERMINAL and sequence >= run.event_sequence:
            return
        if not events:
            await asyncio.sleep(0.05)


class ChatWorker:
    def __init__(self, agent, settings):
        self.agent, self.settings = agent, settings
        self.id = str(uuid4())
        self.stopping = asyncio.Event()

    async def _append(self, run_id, name, data):
        async with in_transaction("business") as conn:
            run = await AgentRun.filter(id=run_id).using_db(conn).select_for_update().first()
            if (
                run is None
                or run.status != "running"
                or run.lease_owner != self.id
                or not run.lease_until
                or run.lease_until <= datetime.now(UTC)
            ):
                raise LeaseLost()
            await journal(run, name, data, conn)

    async def _heartbeat(self, run, execution):
        while True:
            await asyncio.sleep(min(self.settings.worker_lease_seconds / 3, 2))
            async with in_transaction("business") as conn:
                conversation = (
                    await Conversation.filter(id=run.conversation_id)
                    .using_db(conn)
                    .select_for_update()
                    .first()
                )
                current = (
                    await AgentRun.filter(id=run.id).using_db(conn).select_for_update().first()
                )
                # Lock acquisition may outlast the lease. Read time only afterwards.
                now = datetime.now(UTC)
                if (
                    current is None
                    or current.status != "running"
                    or current.lease_owner != self.id
                    or not current.lease_until
                    or current.lease_until <= now
                    or not conversation
                    or conversation.active_run_id != run.id
                    or not conversation.active_run_until
                    or conversation.active_run_until <= now
                ):
                    execution.cancel()
                    return
                current.lease_until = now + timedelta(seconds=self.settings.worker_lease_seconds)
                conversation.active_run_until = now + timedelta(
                    seconds=self.settings.chat_run_timeout_seconds + 30
                )
                await current.save(using_db=conn, update_fields=["lease_until"])
                await conversation.save(using_db=conn, update_fields=["active_run_until"])

    async def _execute(self, run):
        started = time.perf_counter()
        snapshot = run.request_snapshot
        kb = await KnowledgeBase.get(id=run.kb_id, owner_id=run.owner_id)
        kb.config = run.config_snapshot
        state = {
            "run_id": run.id,
            "owner_id": run.owner_id,
            "kb": kb,
            "query": run.query,
            "history": snapshot["history"],
            "filters": MetadataFilter.model_validate(snapshot["filters"]),
            "mode": snapshot["mode"],
            "grade_retries": 0,
            "check_retries": 0,
            "step_ordinal": 0,
        }
        for key in ("application_prompt", "fallback_answer"):
            if key in snapshot:
                state[key] = snapshot[key]
        async with asyncio.timeout(self.settings.chat_run_timeout_seconds):
            async for mode, update in self.agent.graph.astream(
                state,
                config={"recursion_limit": 64},
                stream_mode=["custom", "updates"],
                version="v1",
            ):
                if mode == "custom":
                    await self._append(run.id, update["event"], update["data"])
                else:
                    for delta in update.values():
                        state.update(delta)
        answer, rejected = state.get("answer", FALLBACK_ANSWER), state.get("rejected", True)
        retrieval = state.get("retrieval")
        cited = set(re.findall(r"\[(S\d+)\]", answer))
        citations = (
            [s.model_dump() for s in retrieval.sources if s.source_id in cited]
            if retrieval and not rejected
            else []
        )
        # Lock evidence and conversation through the business commit, preventing
        # reindex/cancel/takeover between validation and accepted-history publication.
        async with artifact_transaction(
            self.settings,
            enabled=snapshot.get("output_type", "answer") != "answer" and not rejected,
        ) as (conn, pending_files):
            documents = (
                await Document.filter(
                    id__in=sorted({s["document_id"] for s in citations}), owner_id=run.owner_id
                )
                .using_db(conn)
                .select_for_update()
                .order_by("id")
            )
            live = {
                d.id: d.index_revision
                for d in documents
                if d.status == "ready"
                and d.embedding_fingerprint == self.settings.embedding_fingerprint
            }
            conversation = (
                await Conversation.filter(id=run.conversation_id)
                .using_db(conn)
                .select_for_update()
                .first()
            )
            current = await AgentRun.filter(id=run.id).using_db(conn).select_for_update().first()
            now = datetime.now(UTC)
            if (
                current.status != "running"
                or current.lease_owner != self.id
                or not current.lease_until
                or current.lease_until <= now
                or not conversation
                or conversation.active_run_id != run.id
                or not conversation.active_run_until
                or conversation.active_run_until <= now
            ):
                raise LeaseLost()
            if any(live.get(s["document_id"]) != s["document_revision"] for s in citations):
                answer, citations, rejected = "回答期间资料发生变化，请重新提问。", [], True
            artifact = None
            output_type = snapshot.get("output_type", "answer")
            if output_type != "answer" and not rejected:
                artifact_started = time.perf_counter()
                try:
                    check = state.get("check")
                    payload = build_payload(
                        output_type,
                        run.query,
                        answer,
                        citations,
                        check.model_dump() if check else {},
                        self.settings.model_provider,
                    )
                    item, _ = await persist_artifact(
                        current, output_type, payload, self.settings, conn, pending_files
                    )
                    artifact = artifact_dto(item)
                    await journal(current, "artifact_ready", {"artifact": artifact}, conn)
                    artifact_summary = {
                        "type": output_type,
                        "id": item.id,
                        "verification": payload["verification"],
                        "chart_kind": payload["chart_kind"],
                    }
                    artifact_status = "completed"
                except ArtifactUnavailable as error:
                    artifact_summary = {"type": output_type, "reason": str(error)}
                    artifact_status = "unavailable"
                    await journal(current, "artifact_unavailable", artifact_summary, conn)
                artifact_elapsed_ms = round((time.perf_counter() - artifact_started) * 1000)
                await AgentStep.create(
                    run_id=run.id,
                    node="artifact",
                    ordinal=state.get("step_ordinal", 0) + 1,
                    status=artifact_status,
                    input_summary={"output_type": output_type},
                    output_summary=artifact_summary,
                    elapsed_ms=artifact_elapsed_ms,
                    using_db=conn,
                )
                await journal(
                    current,
                    "agent_step",
                    {
                        "node": "artifact",
                        "ordinal": state.get("step_ordinal", 0) + 1,
                        "status": artifact_status,
                        "elapsed_ms": artifact_elapsed_ms,
                        "output": artifact_summary,
                    },
                    conn,
                )
                # Local file I/O also may outlast a lease. Read time after it.
                if current.lease_until <= datetime.now(
                    UTC
                ) or conversation.active_run_until <= datetime.now(UTC):
                    raise LeaseLost()
            current.status, current.answer, current.citations = "completed", answer, citations
            current.grade_retries, current.check_retries = (
                state.get("grade_retries", 0),
                state.get("check_retries", 0),
            )
            current.elapsed_ms = round((time.perf_counter() - started) * 1000)
            current.lease_until = None
            await current.save(
                using_db=conn,
                update_fields=[
                    "status",
                    "answer",
                    "citations",
                    "grade_retries",
                    "check_retries",
                    "elapsed_ms",
                    "lease_until",
                ],
            )
            await (
                ChatMessage.filter(run_id=run.id, role="user").using_db(conn).update(accepted=True)
            )
            await ChatMessage.create(
                owner_id=run.owner_id,
                conversation_id=run.conversation_id,
                run_id=run.id,
                role="assistant",
                content=answer,
                citations=citations,
                using_db=conn,
            )
            await journal(current, "draft_reset", {"reason": "最终答案已提交"}, conn)
            for i in range(0, len(answer), 24):
                await journal(
                    current, "token", {"text": answer[i : i + 24], "verified": not rejected}, conn
                )
            await journal(current, "citations", {"sources": citations}, conn)
            await journal(
                current,
                "done",
                {
                    "answer": answer,
                    "rejected": rejected,
                    "run_id": run.id,
                    "grade_retries": current.grade_retries,
                    "check_retries": current.check_retries,
                    "artifacts": [artifact] if artifact else [],
                },
                conn,
            )
            conversation.active_run_id, conversation.active_run_until = "", None
            await conversation.save(
                using_db=conn, update_fields=["active_run_id", "active_run_until"]
            )

    async def tick(self):
        now = datetime.now(UTC)
        for old in await AgentRun.filter(status="running", lease_until__lt=now):
            # Chat APIs have no remote resumable task ID. Never silently reissue
            # a possibly paid generation after a worker crash.
            await finish_error(old.id, "failed", "执行器租约过期，请重新提问", expired_only=True)
        run = await AgentRun.filter(status="queued").order_by("created_at").first()
        if run is None:
            return False
        async with in_transaction("business") as conn:
            await (
                Conversation.filter(id=run.conversation_id)
                .using_db(conn)
                .select_for_update()
                .first()
            )
            run = await AgentRun.filter(id=run.id).using_db(conn).select_for_update().first()
            if run is None or run.status != "queued":
                return False
            run.status, run.lease_owner = "running", self.id
            run.lease_until = datetime.now(UTC) + timedelta(
                seconds=self.settings.worker_lease_seconds
            )
            await run.save(using_db=conn, update_fields=["status", "lease_owner", "lease_until"])
        execution = asyncio.create_task(self._execute(run))
        heartbeat = asyncio.create_task(self._heartbeat(run, execution))
        try:
            await execution
        except (LeaseLost, asyncio.CancelledError):
            if self.stopping.is_set():
                raise
            await finish_error(run.id, "failed", "执行租约失效，答案未发布", lease_owner=self.id)
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, (ValueError, ProviderError))
                else "回答生成失败，请检查服务日志"
            )
            if isinstance(exc, TimeoutError):
                message = "回答超时"
            await finish_error(run.id, "failed", message[:1000], lease_owner=self.id)
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
