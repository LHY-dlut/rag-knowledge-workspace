import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.artifacts import ArtifactUnavailable, build_payload, file_path, render_html
from app.check_protocol import answer_spans, bind_check
from app.models import AgentRun, ChatMessage, Document, GeneratedArtifact, RunEvent
from app.schemas import CheckDecision
from tests.conftest import account, knowledge_base, parse_events, uploaded


def source(text="河岸监测共有17支巡检队参与。", sid="S1"):
    return {
        "source_id": sid,
        "content": text,
        "document_id": "doc-1",
        "parent_id": "p-1",
        "document_revision": 3,
        "filename": "监测.md",
        "location": "第1段",
        "child_ids": ["c-1"],
    }


def payload(answer=None, sources=None, verdict="supported", evidence="S1:E1", kind="chart"):
    sources = sources or [source()]
    answer = answer or sources[0]["content"] + "[S1]"
    check = bind_check(
        CheckDecision(
            checks=[
                {
                    "answer_span_id": s["span_id"],
                    "verdict": verdict,
                    "evidence": [{"source_id": "S1", "span_id": evidence}],
                    "reason": "核对冻结证据",
                }
                for s in answer_spans(answer)
            ]
        ),
        answer,
        sources,
    )
    return build_payload(kind, "监测情况？", answer, sources, check.model_dump(), "dashscope")


def test_numeric_value_is_bound_to_original_source_position():
    data = payload()
    point = data["charts"][0]["points"][0]
    assert point["value"] == 17 and point["unit"] == "支"
    ref = point["numeric_positions"][0]
    assert source()["content"][ref["source_start"] : ref["source_end"]] == "17支"
    assert ref["document_id"] == "doc-1" and data["calculation"]["performed"] is False


@pytest.mark.parametrize(
    "text,value", [("库存0件。", 0), ("净流量-2吨。", -2), ("采购预算1,234万元。", 1234)]
)
def test_literal_zero_negative_and_grouped_numbers(text, value):
    data = payload(sources=[source(text)])
    assert data["charts"][0]["points"][0]["value"] == value


@pytest.mark.parametrize("text", ["库存1e9件。", "净流量+17吨。", "采购预算17,5万元。"])
def test_unsupported_number_syntax_cannot_be_partially_parsed(text):
    assert not payload(sources=[source(text)])["charts"]


@pytest.mark.parametrize(
    "text",
    [
        "维修资料未规定维修完成时限。",
        "河岸有17支巡检队和246名志愿者。",
        "本年度预算约17万元。",
        "并不能保证7天内完成维修。",
    ],
)
def test_ambiguous_or_negative_quantities_remain_evidence_graph(text):
    data = payload(sources=[source(text)])
    assert not data["charts"] and data["chart_kind"] == "evidence_graph"
    assert data["graph"]["edges"] == [
        {"source": "source:S1", "target": "fact:A:E1", "relation": "cited_evidence"}
    ]


def test_bounded_denial_does_not_invent_quantity_missing_from_evidence():
    data = payload(
        answer="根据所提供维修资料，不能认定必须7天完成。[S1]",
        sources=[source("维修资料未规定维修完成时限。")],
    )
    assert not data["charts"] and "7天" in data["facts"][0]["text"]
    assert "7天" not in data["facts"][0]["evidence"][0]["original_quote"]


def test_different_units_are_separate_and_paraphrases_not_numeric_observations():
    text = "采购预算17万元。志愿者246名。"
    answer = "采购预算17万元。[S1]志愿者246名。[S1]"
    checks = CheckDecision(
        checks=[
            {
                "answer_span_id": s["span_id"],
                "verdict": "supported",
                "evidence": [{"source_id": "S1", "span_id": f"S1:E{i + 1}"}],
                "reason": "原文直接数值",
            }
            for i, s in enumerate(answer_spans(answer))
        ]
    )
    data = build_payload(
        "chart",
        "预算与人数",
        answer,
        [source(text)],
        bind_check(checks, answer, [source(text)]).model_dump(),
        "dashscope",
    )
    assert [g["unit"] for g in data["charts"]] == ["万元", "名"]
    assert not payload(answer="巡检队数量为17支。[S1]")["charts"]


@pytest.mark.parametrize("verdict,evidence", [("unsupported", "S1:E1"), ("supported", "S1:E999")])
def test_unsupported_or_nonexistent_evidence_not_promoted(verdict, evidence):
    with pytest.raises(ArtifactUnavailable):
        payload(verdict=verdict, evidence=evidence)


@pytest.mark.parametrize("check", [{}, {"passed": True}, {"passed": False}, None])
def test_missing_semantic_check_rejected(check):
    with pytest.raises(ArtifactUnavailable):
        build_payload("report", "问题", "内容[S1]", [source()], check, "dashscope")


def test_report_html_is_escaped_and_has_no_executable_model_content():
    data = payload(kind="report")
    data["title"] = "<script>alert(1)</script>"
    data["facts"][0]["text"] = "<img src=x onerror=alert(1)>"
    result = render_html(data)
    assert "<script>" not in result and "<img" not in result
    assert "&lt;script&gt;" in result and "default-src 'none'" in result


async def completed(client, output_type="chart"):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    doc = await uploaded(
        client, headers, kb, text="河岸监测共有17支巡检队参与。", filename="河岸监测.md"
    )
    submitted = await client.post(
        "/api/chat/runs",
        headers=headers,
        json={"kb_id": kb, "query": "河岸监测巡检队参与情况？", "output_type": output_type},
    )
    assert submitted.status_code == 202, submitted.text
    run = submitted.json()["data"]
    await client.app.state.chat_worker.tick()
    response = await client.get(f"/api/runs/{run['run_id']}/events", headers=headers)
    events = parse_events(response)
    done = next(d for n, d in events if n == "done")
    assert not done["rejected"], response.text
    return headers, kb, doc, done, events


async def test_artifact_is_durable_before_terminal_and_owned_after_refresh(client):
    headers, kb, doc, done, events = await completed(client)
    names = [n for n, _ in events]
    assert names.index("artifact_ready") < names.index("done")
    item = done["artifacts"][0]
    assert item["payload"]["charts"][0]["points"][0]["value"] == 17
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    assert trace["artifacts"][0]["id"] == item["id"]
    assert trace["steps"][-1]["node"] == "artifact" and trace["steps"][-1]["status"] == "completed"
    artifact_event = next(d for n, d in events if n == "agent_step" and d["node"] == "artifact")
    assert artifact_event["elapsed_ms"] == trace["steps"][-1]["elapsed_ms"]
    assert isinstance(artifact_event["elapsed_ms"], int) and artifact_event["elapsed_ms"] >= 0
    conversation = next(d["conversation_id"] for n, d in events if n == "meta")
    history = (
        await client.get(f"/api/conversations/{conversation}/messages", headers=headers)
    ).json()["data"]
    assert history[-1]["artifacts"][0]["id"] == item["id"]
    assert (await client.get(item["download_url"], headers=headers)).status_code == 200
    stranger = await account(client, "stranger")
    for method, path, body in [
        ("GET", item["download_url"], None),
        ("GET", f"/api/runs/{done['run_id']}/artifacts", None),
        ("POST", f"/api/runs/{done['run_id']}/artifacts", {"type": "report"}),
    ]:
        assert (await client.request(method, path, headers=stranger, json=body)).status_code == 404


async def test_conversion_is_idempotent_and_stale_version_cannot_create_new_artifact(client):
    headers, kb, doc, done, _ = await completed(client, "answer")
    url = f"/api/runs/{done['run_id']}/artifacts"
    first = await client.post(url, headers=headers, json={"type": "report"})
    assert first.status_code == 200, first.text
    item = first.json()["data"]
    second = (await client.post(url, headers=headers, json={"type": "report"})).json()["data"]
    assert second["id"] == item["id"] and second["created"] is False
    assert await GeneratedArtifact.filter(run_id=done["run_id"]).count() == 1
    assert await RunEvent.filter(run_id=done["run_id"], name="artifact_ready").count() == 1
    assert item["preview_html"].startswith("<!doctype html>")
    await Document.filter(id=doc).update(index_revision=99)
    assert (await client.post(url, headers=headers, json={"type": "webpage"})).status_code == 409
    old = (await client.get(url, headers=headers)).json()["data"][0]
    assert (
        old["current_evidence"] is False
        and old["payload"]["citations"][0]["document_revision"] != 99
    )


async def test_tampered_file_fails_download_hash_check(client):
    headers, _, _, done, _ = await completed(client, "webpage")
    item = done["artifacts"][0]
    file_path(client.app.state.settings, item["id"], "webpage").write_text(
        "tampered", encoding="utf-8"
    )
    assert (await client.get(item["download_url"], headers=headers)).status_code == 409


async def test_refusal_does_not_create_artifact_or_convert(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "木星直径？", "output_type": "report"},
    )
    done = next(d for n, d in parse_events(response) if n == "done")
    assert done["rejected"] and done["artifacts"] == []
    assert await GeneratedArtifact.all().count() == 0
    assert (
        await client.post(
            f"/api/runs/{done['run_id']}/artifacts", headers=headers, json={"type": "report"}
        )
    ).status_code == 409


async def test_lease_expiring_during_artifact_io_cannot_publish(client, monkeypatch):
    import app.chat_jobs as jobs

    real = jobs.persist_artifact
    calls = 0

    async def expires(run, *args):
        nonlocal calls
        calls += 1
        item = await real(run, *args)
        assert file_path(client.app.state.settings, item[0].id, item[0].kind).is_file()
        run.lease_until = datetime.now(UTC) - timedelta(seconds=1)
        return item

    monkeypatch.setattr(jobs, "persist_artifact", expires)
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb, text="河岸监测共有17支巡检队参与。", filename="监测.md")
    submitted = (
        await client.post(
            "/api/chat/runs",
            headers=headers,
            json={"kb_id": kb, "query": "河岸监测巡检队参与情况？", "output_type": "report"},
        )
    ).json()["data"]
    await client.app.state.chat_worker.tick()
    # Demo's background chat worker may have claimed the run before manual
    # tick. Wait for its actual terminal event instead of asserting too early.
    response = await asyncio.wait_for(
        client.get(f"/api/runs/{submitted['run_id']}/events", headers=headers), 10
    )
    assert response.status_code == 200
    assert any(name == "error" for name, _ in parse_events(response))
    assert calls == 1, "The fence must be exercised after actual artifact file I/O"
    assert await GeneratedArtifact.filter(run_id=submitted["run_id"]).count() == 0
    assert not await ChatMessage.filter(
        run_id=submitted["run_id"], role="assistant", accepted=True
    ).exists()
    assert not await RunEvent.filter(
        run_id=submitted["run_id"], name__in=["done", "artifact_ready"]
    ).exists()
    assert not list(
        (client.app.state.settings.upload_dir.parent / "generated_artifacts").glob("*.html")
    )


async def test_cancel_during_generation_cannot_publish_artifact(client, monkeypatch):
    gate, release = asyncio.Event(), asyncio.Event()
    original = client.app.state.provider.stream_answer

    async def delayed(*args, **kwargs):
        gate.set()
        await release.wait()
        async for token in original(*args, **kwargs):
            yield token

    monkeypatch.setattr(client.app.state.provider, "stream_answer", delayed)
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    run = (
        await client.post(
            "/api/chat/runs",
            headers=headers,
            json={"kb_id": kb, "query": "出差报销申请期限？", "output_type": "report"},
        )
    ).json()["data"]
    task = asyncio.create_task(client.app.state.chat_worker.tick())
    await asyncio.wait_for(gate.wait(), 10)
    await client.post(f"/api/runs/{run['run_id']}/cancel", headers=headers)
    release.set()
    await task
    assert (await AgentRun.get(id=run["run_id"])).status == "cancelled"
    assert not await GeneratedArtifact.filter(run_id=run["run_id"]).exists()
    assert not await ChatMessage.filter(
        run_id=run["run_id"], role="assistant", accepted=True
    ).exists()
