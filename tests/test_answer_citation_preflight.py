import pytest

from app.check_protocol import answer_citation_coverage
from tests.conftest import account, knowledge_base, parse_events, uploaded


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("甲事项需要申请单[S1]。\n\n乙事项需要审批单[S2]。", True),
        ("甲事项需要申请单。乙事项需要审批单[S1][S2]。", True),
        ("甲事项需要申请单\n以及审批单[S1]。", True),
        ("甲事项需要申请单[S1][S1]。", True),
        ("所有事项保证当日完成：\n\n甲事项需要申请单[S1]。", False),
        ("甲事项需要申请单[S1]。保证当日完成。", False),
        ("甲事项保证当日完成。\n\n乙事项需要审批单[S2]。", False),
        ("[S1]", False),
        ("甲站有6个班组[S1]和43名工作人员[S1]参与值守。", False),
        ("甲站有6个班组和43名工作人员参与值守[S1]。", True),
        ("甲站有6个班组参与值守[S1]。乙站有43名工作人员参与值守[S2]。", True),
        ("甲站有6个班组参与值守。乙站有43名工作人员参与值守[S1][S2]。", True),
        ("甲站有6个班组参与值守[S1]，乙站保证次日完成。", False),
        ("甲站有6个班组参与值守[S1]。\n\n乙站有43名工作人员参与值守。", False),
    ],
)
def test_position_coverage_never_inserts_references_or_crosses_blank_lines(answer, expected):
    result = answer_citation_coverage(answer)
    assert result["passed"] is expected
    assert result["references_added"] is False and result["semantic_support_checked"] is False
    if not expected and answer != "[S1]":
        assert result["uncovered_answer_positions"] and result["uncited_answer_spans"]
        assert all(
            answer[s["source_start"] : s["source_end"]] == s["quote"]
            for s in result["uncited_answer_spans"]
        )


async def test_uncited_intro_is_revised_before_semantic_call_then_verified(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    provider = client.app.state.provider
    original_structured, original_stream = provider.structured, provider.stream_answer
    generated, checks, feedback = [], [], []

    async def capture_check(schema, task, payload):
        if task == "check":
            checks.append(payload["answer"])
        return await original_structured(schema, task, payload)

    async def once_uncited(query, sources, revision, prompt):
        feedback.append(revision)
        generated.append(True)
        if len(generated) == 1:
            yield "这些规定保证所有申请当日完成：\n\n出差报销需要发票和审批单[S1]。"
        else:
            async for token in original_stream(query, sources, revision, prompt):
                yield token

    monkeypatch.setattr(provider, "structured", capture_check)
    monkeypatch.setattr(provider, "stream_answer", once_uncited)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "出差报销需要哪些材料？", "mode": "agent"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and not done["rejected"] and done["check_retries"] == 1
    assert len(generated) == 2 and len(checks) == 1
    assert "保证所有申请当日完成" in feedback[1] and "保证" not in done["answer"]
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    steps = [s["output_summary"] for s in trace["steps"] if s["node"] == "check"]
    assert [s["check"]["passed"] for s in steps] == [False, True]
    assert steps[0]["citation_protocol"]["answer_citation_coverage"]["uncited_answer_spans"]


async def test_complete_citations_do_not_accept_unsupported_fact(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    provider = client.app.state.provider
    original = provider.structured
    checked = []

    async def semantic(schema, task, payload):
        if task == "check":
            checked.append(payload["answer"])
        return await original(schema, task, payload)

    async def false_claim(*args, **kwargs):
        yield "出差报销保证款项次日到账[S1]。"

    monkeypatch.setattr(provider, "structured", semantic)
    monkeypatch.setattr(provider, "stream_answer", false_claim)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "出差报销需要哪些材料？", "mode": "agent"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and done["rejected"] and done["check_retries"] == 2
    assert len(checked) == 3 and "次日到账" not in done["answer"]
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    checks = [s["output_summary"] for s in trace["steps"] if s["node"] == "check"]
    assert all(s["citation_protocol"]["answer_citation_coverage"]["passed"] for s in checks)
    assert all(not s["check"]["passed"] for s in checks)


async def test_inline_citations_leave_predicate_unpublished_then_regenerate(client, monkeypatch):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    provider = client.app.state.provider
    original_stream, original_structured = provider.stream_answer, provider.structured
    attempts, checked = [], []

    async def inline_then_correct(query, sources, feedback, prompt):
        attempts.append(feedback)
        if len(attempts) == 1:
            yield "出差报销需要发票[S1]和审批单[S1]作为材料。"
        else:
            async for token in original_stream(query, sources, feedback, prompt):
                yield token

    async def capture_check(schema, task, payload):
        if task == "check":
            checked.append(payload["answer"])
        return await original_structured(schema, task, payload)

    monkeypatch.setattr(provider, "stream_answer", inline_then_correct)
    monkeypatch.setattr(provider, "structured", capture_check)
    response = await client.post(
        "/api/chat/stream",
        headers=headers,
        json={"kb_id": kb, "query": "出差报销需要哪些材料？", "mode": "agent"},
    )
    events = parse_events(response)
    done = events[-1][1]
    assert events[-1][0] == "done" and not done["rejected"]
    assert done["check_retries"] == 1 and len(attempts) == 2 and len(checked) == 1
    assert "作为材料" in attempts[1] and "陈述末尾" in attempts[1]
    assert "发票[S1]和审批单[S1]作为材料" not in done["answer"]
    trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
    checks = [s["output_summary"] for s in trace["steps"] if s["node"] == "check"]
    assert [s["check"]["passed"] for s in checks] == [False, True]
    coverage = checks[0]["citation_protocol"]["answer_citation_coverage"]
    assert coverage["references_added"] is False and coverage["semantic_support_checked"] is False
    assert any("作为材料" in s["quote"] for s in coverage["uncited_answer_spans"])


@pytest.mark.parametrize(
    "answer,sources,expected",
    [
        # 年份在被引用原文中出现：通过
        (
            "2023年7月制造业PMI为49.3%[S1]。",
            [{"source_id": "S1", "content": "2023-07-31 发布：7月份制造业PMI为49.3%。"}],
            True,
        ),
        # 原文未标年份，答案回写问题里的年份：拦截（真实缺陷形态）
        (
            "2021年7月份，制造业PMI为49.3%[S1]。",
            [
                {
                    "source_id": "S1",
                    "content": "7月份，制造业采购经理指数为49.3%，比上月上升0.3个百分点。",
                }
            ],
            False,
        ),
        # 年份只出现在另一条未被引用的来源里：仍然拦截（不取全来源并集）
        (
            "2021年7月份，制造业PMI为49.3%[S1]。",
            [
                {"source_id": "S1", "content": "7月份，制造业采购经理指数为49.3%。"},
                {"source_id": "S2", "content": "纽约联储：2021年美国民众通胀预期升温。"},
            ],
            False,
        ),
        # 同一引用组内分别引用到含年份的来源：通过
        (
            "2021年数据见甲文[S1]，2023年数据见乙文[S2]。",
            [
                {"source_id": "S1", "content": "2021年该项为8%。"},
                {"source_id": "S2", "content": "2023年该项为10%。"},
            ],
            True,
        ),
        # 没有年份：不触发
        (
            "7月份制造业PMI为49.3%[S1]。",
            [{"source_id": "S1", "content": "7月份PMI为49.3%。"}],
            True,
        ),
        # 句末标记按既有规则覆盖整段前置文字，段内年份同样受该引用约束
        (
            "2021年发布过一份文件。另有陈述[S1]。",
            [{"source_id": "S1", "content": "另有陈述。"}],
            False,
        ),
        # 标记之后、未再出现引用的年份不由本规则报出（属位置预检职责）
        (
            "另有陈述[S1]。2021年发布过一份文件。",
            [{"source_id": "S1", "content": "另有陈述。"}],
            True,
        ),
    ],
)
def test_year_binding_requires_the_year_in_the_cited_source(answer, sources, expected):
    from app.check_protocol import answer_year_binding

    result = answer_year_binding(answer, sources)
    assert result["passed"] is expected
    assert result["semantic_support_checked"] is False
    if not expected:
        assert result["failure_types"] == ["check_year_not_in_cited_source"]
        assert result["unsupported_years"][0]["year"] == "2021"
