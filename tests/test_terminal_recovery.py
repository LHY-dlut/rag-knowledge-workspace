from tests.conftest import account, knowledge_base, parse_events, uploaded


async def test_persisted_terminal_record_retains_acceptance_and_refusal(client):
    headers = await account(client)
    kb = await knowledge_base(client, headers)
    await uploaded(client, headers, kb)
    for question, rejected in (
        ("出差报销申请需要几天内提交？", False),
        ("木星大气甲烷含量是多少？", True),
    ):
        response = await client.post(
            "/api/chat/stream", headers=headers, json={"kb_id": kb, "query": question}
        )
        done = next(data for name, data in parse_events(response) if name == "done")
        trace = (await client.get(f"/api/runs/{done['run_id']}", headers=headers)).json()["data"]
        assert trace["status"] == "completed" and trace["rejected"] is rejected
        assert trace["answer"] == done["answer"]
        assert bool(trace["citations"]) is not rejected
