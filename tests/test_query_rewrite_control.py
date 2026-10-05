import json
from types import SimpleNamespace

import pytest

from app.agent import RAGAgent
from app.schemas import RetrievalConfig

ORIGINAL = "派件周期是多久？"
ROUTER_QUERY = "验收示例的派件周期及负责组"


class Router:
    def __init__(self):
        self.rewrite_inputs = []

    async def route(self, query, history, tools):
        return {
            "tool_calls": [
                {
                    "function": {
                        "name": "search_knowledge",
                        "arguments": json.dumps({"query": ROUTER_QUERY}),
                    }
                }
            ]
        }

    async def structured(self, schema, task, payload):
        assert task == "rewrite"
        self.rewrite_inputs.append(payload)
        return schema(
            standalone_query="模型建议的另一种问题",
            queries=["派件频率"],
            hypothetical_document="模拟假设资料",
        )


def state(config):
    return {
        "mode": "agent",
        "owner_id": "u",
        "query": ORIGINAL,
        "history": [],
        "kb": SimpleNamespace(config=config.model_dump()),
        "standalone_query": ROUTER_QUERY,
    }


@pytest.mark.parametrize("enabled", [False, True])
async def test_router_search_arguments_respect_query_rewrite_switch(client, enabled):
    config = RetrievalConfig(query_rewrite=enabled, multi_query=False, hyde=False)
    provider = Router()
    agent = RAGAgent(provider, None)
    result = await agent.route(state(config))
    assert result["standalone_query"] == (ROUTER_QUERY if enabled else ORIGINAL)
    assert result["route"] == "knowledge"


@pytest.mark.parametrize(
    "multi_query,hyde", [(False, False), (True, False), (False, True), (True, True)]
)
async def test_disabled_coreference_never_inherits_stale_router_query(multi_query, hyde):
    provider = Router()
    agent = RAGAgent(provider, None)
    result = await agent.rewrite(
        state(RetrievalConfig(query_rewrite=False, multi_query=multi_query, hyde=hyde))
    )
    assert result["standalone_query"] == ORIGINAL
    assert ORIGINAL in result["queries"]
    assert all(payload["query"] == ORIGINAL for payload in provider.rewrite_inputs)
    if not multi_query and not hyde:
        assert result["queries"] == [ORIGINAL]
        assert not provider.rewrite_inputs


async def test_enabled_rewrite_retains_explicit_router_resolution():
    provider = Router()
    result = await RAGAgent(provider, None).rewrite(state(RetrievalConfig(query_rewrite=True)))
    assert provider.rewrite_inputs[0]["query"] == ROUTER_QUERY
    assert result["standalone_query"] == "模型建议的另一种问题"
