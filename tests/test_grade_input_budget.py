import copy
import json

import httpx
import pytest

from app.evidence_protocol import evidence_spans
from app.providers import DashScopeProvider, ProviderError
from app.schemas import GradeDecision
from tests.test_providers_parsers import remote_settings


def response():
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "passed": False,
                                "support": "insufficient",
                                "reason": "Mock transport budget check",
                            }
                        )
                    }
                }
            ]
        },
    )


async def test_registry_overhead_falls_back_without_losing_source_or_raising_limit():
    source = {
        "source_id": "S1",
        "content": "响应2小时。\n" * 1000,
        "document_id": "doc",
        "parent_id": "parent",
    }
    payload = {
        "query": "响应要求？",
        "sources": [{**source, "evidence_spans": evidence_spans(source)}],
    }
    before = copy.deepcopy(payload)
    seen = []
    settings = remote_settings()

    def handler(request):
        body = json.loads(request.content)
        actual = json.loads(body["messages"][1]["content"])
        assert actual["sources"] == [source]
        assert actual["evidence_protocol_mode"] == "continuous_quote_budget_fallback"
        assert (
            len(json.dumps(body, ensure_ascii=False).encode()) <= settings.model_max_request_bytes
        )
        seen.append(actual)
        return response()

    provider = DashScopeProvider(settings, httpx.MockTransport(handler))
    try:
        await provider.structured(GradeDecision, "grade", payload)
        assert len(seen) == 1 and payload == before and settings.model_max_request_bytes == 100000
    finally:
        await provider.close()


async def test_normal_span_request_preserves_v15_protocol_without_budget_fallback():
    source = {"source_id": "S1", "content": "响应2小时。维修完成期限未规定。"}
    payload = {
        "query": "响应要求？",
        "sources": [{**source, "evidence_spans": evidence_spans(source)}],
    }
    seen = []

    def handler(request):
        body = json.loads(request.content)
        actual = json.loads(body["messages"][1]["content"])
        assert "evidence_protocol_mode" not in actual
        assert actual["sources"][0]["evidence_spans"] == payload["sources"][0]["evidence_spans"]
        assert (
            "content" not in actual["sources"][0]
            and "本次因输入字节预算" not in body["messages"][0]["content"]
        )
        seen.append(actual)
        return response()

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        await provider.structured(GradeDecision, "grade", payload)
        assert len(seen) == 1
    finally:
        await provider.close()


async def test_full_original_context_still_over_limit_sends_no_request():
    source = {"source_id": "S1", "content": "甲" * 40000}

    def forbidden(request):
        pytest.fail("Oversized raw context must never be dispatched")

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(forbidden))
    try:
        with pytest.raises(ProviderError, match="未发送请求"):
            await provider.structured(
                GradeDecision,
                "grade",
                {
                    "query": "问题？",
                    "sources": [{**source, "evidence_spans": evidence_spans(source)}],
                },
            )
    finally:
        await provider.close()
