import json

import httpx
import pytest

from app.check_protocol import answer_citation_manifest, answer_spans, bind_check
from app.providers import DashScopeProvider, ProviderError
from app.schemas import CheckDecision, Judgment
from tests.test_providers_parsers import remote_settings


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("申请须提交编号[S1]。办理由受理组负责[S2]。", [["S1"], ["S2"]]),
        ("申请须提交编号。办理由受理组负责[S1][S2]。", [["S1", "S2"], ["S1", "S2"]]),
        ("申请须提交编号[S1][S1]。", [["S1"]]),
        ("无引用的独立声明。\n\n申请须提交编号[S2]。", [[], ["S2"]]),
        ("申请须提交编号[S9]。", [["S9"]]),
    ],
)
def test_manifest_preserves_raw_spans_and_actual_paragraph_citation_scope(answer, expected):
    manifest = answer_citation_manifest(answer)
    assert [s["cited_source_ids"] for s in manifest] == expected
    original = answer_spans(answer)
    for row, before in zip(manifest, original, strict=True):
        assert row["answer_span_id"] == before["span_id"]
        assert answer[before["source_start"] : before["source_end"]] == before["quote"]
        assert all(
            before["source_start"] <= s["text_start"] < s["text_end"] <= before["source_end"]
            for s in row["citation_segments"]
        )


async def test_remote_check_receives_exact_citation_map_and_keeps_correct_bindings():
    answer = "申请须提交编号[S1]。办理由受理组负责[S2]。"
    sources = [
        {"source_id": "S1", "content": "申请须提交编号。"},
        {"source_id": "S2", "content": "办理由受理组负责。"},
    ]
    captured = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        captured.append(payload)
        assert payload["answer_spans"] == answer_spans(answer)
        assert [s["cited_source_ids"] for s in payload["answer_citation_manifest"]] == [
            ["S1"],
            ["S2"],
        ]
        assert payload["answer"] == answer
        checks = [
            {
                "answer_span_id": span["span_id"],
                "verdict": "supported",
                "evidence": [{"source_id": sid, "span_id": sid + ":E1"}],
                "reason": "当前片段由实际引用的原文支持",
            }
            for span, sid in zip(payload["answer_spans"], ("S1", "S2"), strict=True)
        ]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"checks": checks})}}]}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        result = await provider.structured(
            Judgment,
            "check",
            {"query": "申请要求与责任组是什么？", "answer": answer, "sources": sources},
        )
        assert result.passed and len(captured) == 1
        assert result.evidence_protocol["passed"]
    finally:
        await provider.close()


def test_equivalent_text_in_uncited_source_still_cannot_replace_actual_citation():
    answer = "申请须提交编号[S1]。"
    sources = [{"source_id": sid, "content": "申请须提交编号。"} for sid in ("S1", "S2")]
    manifest = answer_citation_manifest(answer)
    assert manifest[0]["cited_source_ids"] == ["S1"]
    wrong = CheckDecision.model_validate(
        {
            "checks": [
                {
                    "answer_span_id": manifest[0]["answer_span_id"],
                    "verdict": "supported",
                    "evidence": [{"source_id": "S2", "span_id": "S2:E1"}],
                    "reason": "另一来源也有相同事实",
                }
            ]
        }
    )
    result = bind_check(wrong, answer, sources)
    assert not result.passed
    assert "check_evidence_not_cited_by_claim" in result.evidence_protocol["failure_types"]
    assert "check_cited_source_not_verified" in result.evidence_protocol["failure_types"]


async def test_correct_citation_map_never_upgrades_semantically_unsupported_assertion():
    answer = "申请保证次日完成[S1]。"
    sources = [{"source_id": "S1", "content": "申请须提交编号。"}]

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][1]["content"])
        span = payload["answer_spans"][0]
        assert payload["answer_citation_manifest"][0]["cited_source_ids"] == ["S1"]
        result = {
            "checks": [
                {
                    "answer_span_id": span["span_id"],
                    "verdict": "unsupported",
                    "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                    "reason": "原文只规定材料，没有完成时间保证",
                }
            ]
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        result = await provider.structured(
            Judgment, "check", {"query": "次日完成吗？", "answer": answer, "sources": sources}
        )
        assert not result.passed and result.evidence_protocol["passed"]
    finally:
        await provider.close()


async def test_citation_manifest_respects_request_budget_without_truncation_or_dispatch():
    answer = "申请须提交编号[S1]。" * 12
    payload = {
        "query": "申请要求是什么？",
        "answer": answer,
        "sources": [{"source_id": "S1", "content": "申请须提交编号。"}],
    }
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        wire = json.loads(body["messages"][1]["content"])
        checks = [
            {
                "answer_span_id": span["span_id"],
                "verdict": "supported",
                "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                "reason": "编号要求由实际来源支持",
            }
            for span in wire["answer_spans"]
        ]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({"checks": checks})}}]}
        )

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handler))
    try:
        assert (await provider.structured(Judgment, "check", payload)).passed
        body = requests[0]
        full_size = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        wire = json.loads(body["messages"][1]["content"])
        del wire["answer_citation_manifest"]
        smaller_body = {
            **body,
            "messages": [
                body["messages"][0],
                {"role": "user", "content": json.dumps(wire, ensure_ascii=False)},
            ],
        }
        limit = len(json.dumps(smaller_body, ensure_ascii=False).encode("utf-8"))
        assert 1000 <= limit < full_size
        provider.settings = provider.settings.model_copy(update={"model_max_request_bytes": limit})
        with pytest.raises(ProviderError, match="未发送请求"):
            await provider.structured(Judgment, "check", payload)
        assert len(requests) == 1
        assert payload["answer"] == answer
        assert payload["sources"][0]["content"] == "申请须提交编号。"
    finally:
        await provider.close()
