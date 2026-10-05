import pytest

from app.models import Document, KnowledgeBase
from app.providers import ProviderError
from app.schemas import MetadataFilter
from app.vector_models import ChunkVector
from tests.conftest import account, knowledge_base, parse_events, uploaded


async def test_full_ingestion_retrieval_chat(client):
    headers = await account(client)
    kb_id = await knowledge_base(client, headers)
    doc_id = await uploaded(client, headers, kb_id)
    document = await Document.get(id=doc_id)
    assert document.status == "ready" and document.child_count > 0
    vector = await ChunkVector.filter(doc_id=doc_id).first()
    assert len(vector.embedding) == 1024
    debug = await client.post(
        "/api/retrieval/debug",
        headers=headers,
        json={"kb_id": kb_id, "query": "出差报销申请需要几天内提交？"},
    )
    assert debug.status_code == 200, debug.text
    assert debug.json()["data"]["sources"][0]["document_id"] == doc_id
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb_id, "query": "出差报销申请需要几天内提交？"},
    )
    events = parse_events(response)
    names = [n for n, _ in events]
    assert "draft_token" in names and "token" in names and names[-1] == "done", response.text
    assert "7天" in events[-1][1]["answer"]
    assert not events[-1][1]["rejected"]
    assert names.index("draft_token") < names.index("token")
    run_id = events[-1][1]["run_id"]
    trace = (await client.get(f"/api/runs/{run_id}", headers=headers)).json()["data"]
    assert [s["node"] for s in trace["steps"]] == [
        "route",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "check",
    ]
    conversation_id = next(d["conversation_id"] for n, d in events if n == "meta")
    messages = (
        await client.get(f"/api/conversations/{conversation_id}/messages", headers=headers)
    ).json()["data"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


async def test_irrelevant_evidence_exact_retry_limit(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "木星大气的甲烷含量是多少？"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert done["rejected"] and done["grade_retries"] == 3 and done["check_retries"] == 0
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    assert sum(s["node"] == "retrieve" for s in trace["steps"]) == 4
    assert not any(s["node"] == "generate" for s in trace["steps"])
    assert next(d for n, d in events if n == "citations")["sources"] == []


@pytest.mark.parametrize(
    "bad_answer", ["凭空编造的答案 [S999]", "报销规定必须100天后才能办理 [S1]"]
)
async def test_hallucination_regenerates_without_retrieval(client, monkeypatch, bad_answer):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)

    async def hallucination(*args, **kwargs):
        yield bad_answer

    monkeypatch.setattr(client.app.state.provider, "stream_answer", hallucination)
    response = await client.post(
        "/api/chat/stream", headers=headers, json={"kb_id": kb, "query": "出差报销申请怎么提交？"}
    )
    events = parse_events(response)
    done = events[-1][1]
    assert done["rejected"] and done["check_retries"] == 2
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    assert sum(s["node"] == "generate" for s in trace["steps"]) == 3
    assert sum(s["node"] == "retrieve" for s in trace["steps"]) == 1
    assert all(
        "S999" not in d["text"] and "100天" not in d["text"] for n, d in events if n == "token"
    )


async def test_simple_rag_has_no_retry(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        "/api/chat/stream", headers=headers, json={"kb_id": kb, "query": "未知问题", "mode": "rag"}
    )
    done = parse_events(response)[-1][1]
    assert done["rejected"] and done["grade_retries"] == 0


async def test_acl_every_resource_and_metadata_filter(client):
    first, second = await account(client, "first"), await account(client, "second")
    kb1, kb2 = await knowledge_base(client, first), await knowledge_base(client, second)
    doc1 = await uploaded(client, first, kb1, "保密政策：秘密项目代号是雾山。")
    await uploaded(client, second, kb2, "公开指南：报销需要发票。")
    assert (await client.get(f"/api/documents/{doc1}/chunks", headers=second)).status_code == 404
    assert (
        await client.post("/api/chat/stream", headers=second, json={"kb_id": kb1, "query": "秘密"})
    ).status_code == 404
    own = await client.post(
        "/api/retrieval/debug",
        headers=second,
        json={"kb_id": kb2, "query": "秘密项目代号", "filters": {"document_ids": [doc1]}},
    )
    assert own.status_code == 200 and own.json()["data"]["sources"] == []


async def test_jwt_and_duplicate_upload(client):
    assert (await client.get("/api/knowledge-bases")).status_code == 401
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    first = await uploaded(client, headers, kb, "重复文本")
    response = await client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=headers,
        files={"file": ("again.txt", "重复文本".encode())},
    )
    assert response.json()["data"]["duplicate"]
    assert response.json()["data"]["document"]["id"] == first
    assert await Document.filter(kb_id=kb).count() == 1
    assert (
        await client.get("/api/knowledge-bases", headers={"Authorization": "Bearer bad-token"})
    ).status_code == 401


async def test_chunk_edit_reindexes_and_delete_invalidates_cache(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    doc = await uploaded(client, headers, kb)
    await client.post("/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销"})
    chunks = (await client.get(f"/api/documents/{doc}/chunks", headers=headers)).json()["data"]
    response = await client.put(
        f"/api/chunks/{chunks[0]['id']}",
        headers=headers,
        json={"content": "报销政策已修订，员工应在14天内提交申请。"},
    )
    assert response.status_code == 202, response.text
    await client.app.state.worker.tick()
    debug = await client.post(
        "/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销申请"}
    )
    assert "14天" in debug.json()["data"]["sources"][0]["content"]
    assert (await client.delete(f"/api/documents/{doc}", headers=headers)).status_code == 200
    assert not await ChunkVector.filter(doc_id=doc).exists()
    debug = await client.post(
        "/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销申请"}
    )
    assert debug.json()["data"]["sources"] == []


async def test_failure_not_published(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        f"/api/knowledge-bases/{kb}/documents",
        headers=headers,
        files={"file": ("broken.pdf", b"not a pdf")},
    )
    assert response.status_code == 202
    doc_id = response.json()["data"]["document"]["id"]
    await client.app.state.worker.tick()
    assert (await Document.get(id=doc_id)).status == "failed"
    assert not await ChunkVector.filter(doc_id=doc_id).exists()


async def test_provider_error_releases_conversation(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)

    async def fail(*args, **kwargs):
        raise ProviderError("模拟断线")

    monkeypatch.setattr(client.app.state.provider, "route", fail)
    response = await client.post(
        "/api/chat/stream", headers=headers, json={"kb_id": kb, "query": "报销"}
    )
    events = parse_events(response)
    assert events[-1][0] == "error"
    from app.models import ChatMessage, Conversation

    conversation = await Conversation.get(id=events[0][1]["conversation_id"])
    assert conversation.active_run_id == ""
    assert not await ChatMessage.filter(conversation_id=conversation.id, accepted=True).exists()


async def test_calculator_function_call(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        "/api/chat/stream", headers=headers, json={"kb_id": kb, "query": "计算(2+3)*4"}
    )
    done = parse_events(response)[-1][1]
    assert done["answer"] == "计算结果：20"


async def test_evaluation_four_metrics(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    response = await client.post(
        "/api/evaluations",
        headers=headers,
        json={
            "kb_id": kb,
            "cases": [
                {
                    "question": "出差报销申请需要几天内提交？",
                    "reference_answer": "7天内提交",
                    "reference_facts": ["7天内提交报销申请"],
                    "relevant_document_ids": [],
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["demo"]
    assert set(data["metrics"]) >= {
        "context_recall",
        "context_precision",
        "faithfulness",
        "answer_relevancy",
    }
    assert data["metrics"]["context_recall"] == 1


async def test_embedding_fingerprint_change_rejected(client):
    headers = await account(client)
    kb_id = await knowledge_base(client, headers)
    doc = await uploaded(client, headers, kb_id)
    await Document.filter(id=doc).update(embedding_fingerprint="other-model:1024")
    response = await client.post(
        "/api/retrieval/debug", headers=headers, json={"kb_id": kb_id, "query": "报销"}
    )
    assert response.status_code == 400 and "重新索引" in response.text


async def test_hyde_never_enters_evidence(client):
    headers = await account(client)
    kb_id = await knowledge_base(client, headers, {"hyde": True})
    await uploaded(client, headers, kb_id)
    kb = await KnowledgeBase.get(id=kb_id)
    result = await client.app.state.retriever.retrieve(
        owner_id=kb.owner_id,
        kb=kb,
        original_query="报销",
        queries=["报销"],
        hypothetical_document="HyDE虚构：报销需要100天并奖励9999元",
        filters=MetadataFilter(),
    )
    assert result.diagnostics["hyde_used"]
    assert all("9999元" not in s.content for s in result.sources)
