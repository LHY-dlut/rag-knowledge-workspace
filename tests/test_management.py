import pytest

from app.chunking import ParsedUnit, parent_child_chunks
from app.evaluation import context_average_precision
from app.models import KnowledgeBase
from app.schemas import GeneratedCases, RetrievalConfig
from app.strategies import effective_kb, expanded_queries
from tests.conftest import account, knowledge_base, parse_events, uploaded


def test_context_precision_rewards_early_evidence():
    assert context_average_precision(["S1", "S2"], {"S1"}) == 1
    assert context_average_precision(["S1", "S2"], {"S2"}) == 0.5
    assert context_average_precision([], set()) == 0


def test_presets_preserve_chunking_and_do_not_mutate_kb():
    original = KnowledgeBase(config=RetrievalConfig(parent_size=1600).model_dump())
    dense = effective_kb(original, "dense")
    full = effective_kb(original, "full")
    assert dense.config["top_k"] == 5 and dense.config["cosine_threshold"] == 0.3
    assert full.config["candidate_k"] == 10 and full.config["rerank_k"] == 4
    assert full.config["rerank_threshold"] == 0.05
    assert full.config["parent_size"] == 1600
    assert not original.config["rerank"]
    assert expanded_queries("原问题", "完整问题", ["改写1", "改写2", "改写3"], True)[0] == "原问题"
    assert len(expanded_queries("问题", "问题", ["1", "2", "3"], True)) == 4


@pytest.mark.parametrize("strategy,budget", [("recursive", 500), ("recursive_short", 250)])
def test_recursive_modes_embed_each_complete_chunk(strategy, budget):
    text = "文档中的一句话。" * 250
    config = RetrievalConfig(chunk_strategy=strategy)
    chunks = parent_child_chunks("document", [ParsedUnit(text)], config)
    assert len(chunks) > 1
    assert all(len(c.content) <= budget for c in chunks)
    assert all(len(c.children) == 1 and c.children[0][1] == c.content for c in chunks)
    assert chunks[0].content[:20] in text and chunks[-1].content.endswith("话。")


async def test_application_binding_and_custom_refusal(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        "/api/applications",
        headers=headers,
        json={
            "kb_id": kb,
            "name": "制度客服",
            "mode": "rag",
            "strategy": "full",
            "fallback": "当前制度未规定，请咨询行政部门。",
        },
    )
    assert response.status_code == 201, response.text
    app_id = response.json()["data"]["id"]
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "application_id": app_id, "query": "火星气温是多少？", "mode": "agent"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert done["answer"] == "当前制度未规定，请咨询行政部门。"
    assert done["grade_retries"] == 0
    meta = next(d for n, d in events if n == "meta")
    assert meta["mode"] == "rag"
    assert (await KnowledgeBase.get(id=kb)).config["rerank"] is False
    # A direct-KB request cannot reuse an application-bound conversation.
    mismatch = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "另一个问题", "conversation_id": meta["conversation_id"]},
    )
    assert mismatch.status_code == 404
    other = await account(client, "stranger")
    assert (
        await client.put(
            f"/api/applications/{app_id}", headers=other, json={"kb_id": kb, "name": "篡改"}
        )
    ).status_code == 404


async def test_feedback_is_owner_scoped_and_requires_manual_reference(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    response = await client.post(
        "/api/chat/stream", headers=headers, json={"kb_id": kb, "query": "报销申请多久提交？"}
    )
    run_id = parse_events(response)[-1][1]["run_id"]
    result = await client.put(
        f"/api/runs/{run_id}/feedback",
        headers=headers,
        json={"helpful": False, "comment": "需进一步核对"},
    )
    assert result.status_code == 200
    assert (
        await client.post("/api/datasets/from-feedback", headers=headers, json={"run_id": run_id})
    ).status_code == 422
    result = await client.post(
        "/api/datasets/from-feedback",
        headers=headers,
        json={
            "run_id": run_id,
            "reference_answer": "7天内提交",
            "reference_facts": ["7天内提交报销申请"],
        },
    )
    assert result.status_code == 201 and result.json()["data"]["origin"] == "feedback_corrected"
    other = await account(client, "otheruser")
    assert (
        await client.put(f"/api/runs/{run_id}/feedback", headers=other, json={"helpful": True})
    ).status_code == 404


async def test_generated_dataset_has_real_quotes_and_invalid_quote_is_rejected(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    response = await client.post(
        "/api/datasets/generate", headers=headers, json={"kb_id": kb, "count": 5}
    )
    assert response.status_code == 201, response.text
    cases = response.json()["data"]["cases"]
    assert len(cases) == 5 and cases[0]["provenance"]
    assert all(
        any(fact in s["content"] for s in c["provenance"])
        for c in cases
        for fact in c["reference_facts"]
    )

    async def invalid(schema, task, payload):
        return GeneratedCases(
            cases=[
                {
                    "question": "凭空问题",
                    "reference_answer": "凭空答案",
                    "reference_facts": ["不存在的事实"],
                    "source_ids": ["S1"],
                }
            ]
        )

    monkeypatch.setattr(client.app.state.provider, "structured", invalid)
    invalid_response = await client.post(
        "/api/datasets/generate", headers=headers, json={"kb_id": kb, "count": 1}
    )
    assert invalid_response.status_code == 400
    datasets = (await client.get(f"/api/datasets?kb_id={kb}", headers=headers)).json()["data"]
    assert len(datasets) == 1


async def test_comparison_runs_three_real_presets_with_same_cases(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    body = {
        "kb_id": kb,
        "cases": [
            {
                "question": "报销申请多久提交？",
                "reference_answer": "7天内提交",
                "reference_facts": ["7天内提交报销申请"],
            }
        ],
    }
    response = await client.post("/api/evaluations/compare", headers=headers, json=body)
    assert response.status_code == 200, response.text
    comparisons = response.json()["data"]["comparisons"]
    assert [c["strategy"] for c in comparisons] == ["dense", "hybrid", "full"]
    assert len({c["id"] for c in comparisons}) == 3
    assert all(
        c["results"][0]["mode"] == "rag" and c["metrics"]["elapsed_ms"] > 0 for c in comparisons
    )
    assert comparisons[-1]["config"]["rerank"]
    assert not (await KnowledgeBase.get(id=kb)).config["rerank"]


async def test_clock_tool_and_model_connectivity_without_credentials_leak(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        "/api/chat/stream", headers=headers, json={"kb_id": kb, "query": "当前时间"}
    )
    assert "当前系统时间" in parse_events(response)[-1][1]["answer"]
    tools = (await client.get("/api/tools", headers=headers)).json()["data"]
    assert len(tools) == 3
    test = await client.post("/api/models/test", headers=headers)
    assert test.json()["data"]["embedding_dimensions"] == 1024
    status = (await client.get("/api/models", headers=headers)).json()["data"]
    assert not any("secret" in key or "api_key" in key for key in status)


async def test_file_replacement_publishes_new_revision_and_removes_old_vectors(client):
    from pathlib import Path

    from app.models import Document
    from app.vector_models import ChunkVector

    headers = await account(client)
    kb = await knowledge_base(client, headers)
    doc_id = await uploaded(client, headers, kb)
    original = await Document.get(id=doc_id)
    old_ids = set(await ChunkVector.filter(doc_id=doc_id).values_list("id", flat=True))
    assert original.index_revision == 1
    response = await client.put(
        f"/api/documents/{doc_id}/file",
        headers=headers,
        files={"file": ("新版制度.md", "报销申请应在14天内提交。".encode())},
    )
    assert response.status_code == 202, response.text
    assert not Path(original.storage_path).exists()
    pending = await client.post(
        "/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销"}
    )
    assert pending.json()["data"]["sources"] == []
    await client.app.state.worker.tick()
    updated = await Document.get(id=doc_id)
    assert updated.index_revision == 2 and updated.status == "ready"
    assert await Document.filter(kb_id=kb).count() == 1
    new_ids = set(await ChunkVector.filter(doc_id=doc_id).values_list("id", flat=True))
    assert new_ids and not (old_ids & new_ids)
    debug = await client.post(
        "/api/retrieval/debug", headers=headers, json={"kb_id": kb, "query": "报销"}
    )
    source = debug.json()["data"]["sources"][0]
    assert "14天" in source["content"] and source["document_revision"] == 2


async def test_changed_evidence_revision_cannot_be_submitted_as_final_answer(client, monkeypatch):
    from tortoise.expressions import F

    from app.models import Document

    headers = await account(client)
    kb = await knowledge_base(client, headers)
    doc_id = await uploaded(client, headers, kb)
    original = client.app.state.provider.stream_answer

    async def update_during_generation(*args, **kwargs):
        await Document.filter(id=doc_id).update(index_revision=F("index_revision") + 1)
        async for token in original(*args, **kwargs):
            yield token

    monkeypatch.setattr(client.app.state.provider, "stream_answer", update_during_generation)
    response = await client.post(
        "/api/chat/stream", headers=headers, json={"kb_id": kb, "query": "报销申请多久提交？"}
    )
    events = parse_events(response)
    assert events[-1][1]["rejected"]
    assert "资料发生变化" in events[-1][1]["answer"]
    assert next(d for n, d in events if n == "citations")["sources"] == []
    assert not any("7天" in d["text"] for n, d in events if n == "token")
