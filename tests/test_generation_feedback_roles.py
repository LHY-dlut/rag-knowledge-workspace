import json

from tests.conftest import account, knowledge_base, parse_events, uploaded


async def test_grade_reason_is_debug_data_but_only_check_feedback_requests_revision(
    client, monkeypatch
):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    provider = client.app.state.provider
    structured = provider.structured
    stream = provider.stream_answer
    sentinel = "内部审阅备注：另一流程的期限未讨论，不是所问材料的新事实"
    received = []

    async def grade_note(schema, task, payload):
        result = await structured(schema, task, payload)
        if task == "grade" and result.passed:
            return result.model_copy(update={"reason": sentinel})
        return result

    async def wrong_then_correct(query, sources, feedback, prompt):
        received.append(json.loads(feedback))
        if len(received) == 1:
            yield "报销材料必须在100天后提交[S1]。"
        else:
            async for token in stream(query, sources, feedback, prompt):
                yield token

    monkeypatch.setattr(provider, "structured", grade_note)
    monkeypatch.setattr(provider, "stream_answer", wrong_then_correct)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "出差报销需要哪些材料？", "mode": "agent"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and not done["rejected"] and done["check_retries"] == 1
    assert len(received) == 2 and received[0]["revision_feedback"] == ""
    assert sentinel not in json.dumps(received[0], ensure_ascii=False)
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    grade = next(s for s in trace["steps"] if s["node"] == "grade")
    checks = [s for s in trace["steps"] if s["node"] == "check"]
    assert grade["output_summary"]["grade"]["reason"] == sentinel
    assert received[1]["revision_feedback"] == checks[0]["output_summary"]["check"]["reason"]
    assert [s["output_summary"]["check"]["passed"] for s in checks] == [False, True]
    assert "100天" not in done["answer"]


async def test_insufficient_grade_keeps_retry_feedback_without_entering_generation(
    client, monkeypatch
):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    generated = []

    async def unexpected_generate(*args, **kwargs):
        generated.append(True)
        yield "不应执行"

    monkeypatch.setattr(client.app.state.provider, "stream_answer", unexpected_generate)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "木星大气的甲烷含量是多少？", "mode": "agent"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and done["rejected"] and done["grade_retries"] == 3
    assert not generated
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    grades = [s["output_summary"] for s in trace["steps"] if s["node"] == "grade"]
    assert len(grades) == 4
    assert all(not s["grade"]["passed"] and s["feedback"] == s["grade"]["reason"] for s in grades)
