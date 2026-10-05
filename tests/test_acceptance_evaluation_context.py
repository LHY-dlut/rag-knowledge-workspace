import pytest
from pydantic import ValidationError

from app.models import KnowledgeBase
from app.schemas import EvaluationCase, RetrievalConfig
from tests.conftest import account, knowledge_base, uploaded


def test_answerable_cases_still_require_reference_facts():
    with pytest.raises(ValidationError, match="require reference facts"):
        EvaluationCase(question="报销?", reference_answer="7天")
    case = EvaluationCase(question="资料外?", reference_answer="无依据", answerable=False)
    assert case.reference_facts == []


async def test_multiturn_evaluation_preserves_history_and_undefined_recall(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers, RetrievalConfig(query_rewrite=True).model_dump())
    await uploaded(client, headers, kb)
    history = [{"role": "user", "content": "我在问出差报销"}]
    seen = []
    original = client.app.state.provider.structured

    async def capture(schema, task, payload):
        if task in {"rewrite", "evaluation"}:
            seen.append((task, payload.get("history")))
        return await original(schema, task, payload)

    monkeypatch.setattr(client.app.state.provider, "structured", capture)
    response = await client.post(
        "/api/evaluations",
        headers=headers,
        json={
            "kb_id": kb,
            "cases": [
                {
                    "question": "那需要哪些材料？",
                    "reference_answer": "资料不足",
                    "answerable": False,
                    "history": history,
                    "reference_facts": [],
                }
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "completed"
    assert data["metrics"]["context_recall"] is None
    assert ("rewrite", history) in seen and ("evaluation", history) in seen
    assert data["results"][0]["history"] == history
    assert await KnowledgeBase.filter(id=kb).exists()
