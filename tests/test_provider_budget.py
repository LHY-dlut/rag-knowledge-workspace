import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from app import providers
from app.models import ProviderBudget
from app.provider_budget import DispatchLimitExceeded, reserve_provider_call
from app.providers import DashScopeProvider, ProviderError
from app.settings import Settings


def profile(**overrides):
    return Settings(
        _env_file=None,
        app_mode="production",
        model_provider="dashscope",
        business_db_url="mysql://unused@localhost/test",
        vector_db_url="postgres://unused@localhost/test",
        jwt_secret=uuid4().hex + uuid4().hex,
        dashscope_api_key="mock-only",
        dashscope_http_base_url="https://mock.invalid/api/v1",
        dashscope_chat_base_url="https://mock.invalid/compatible-mode/v1",
        **overrides,
    )


async def test_persistent_limit_is_not_reset_when_provider_restarts(client):
    assert await reserve_provider_call(2) == 1
    assert await reserve_provider_call(2) == 2
    with pytest.raises(DispatchLimitExceeded):
        await reserve_provider_call(2)
    day = datetime.now(UTC).date().isoformat()
    assert (await ProviderBudget.get(id=day)).dispatches == 2


async def test_budget_database_failure_blocks_network_and_redacts_error(monkeypatch):
    calls = []

    async def broken(_):
        raise RuntimeError("mysql://secret:password@host/database")

    monkeypatch.setattr(providers, "reserve_provider_call", broken)
    provider = DashScopeProvider(
        profile(), httpx.MockTransport(lambda request: calls.append(request))
    )
    try:
        with pytest.raises(ProviderError, match="预算记录不可用") as exc:
            await provider.embed(["test"])
        assert not calls and "password" not in str(exc.value)
    finally:
        await provider.close()


async def test_stalled_budget_database_times_out_before_network_dispatch(monkeypatch):
    network = []

    async def stalled(_):
        await asyncio.Event().wait()

    monkeypatch.setattr(providers, "reserve_provider_call", stalled)
    provider = DashScopeProvider(
        profile(model_timeout_seconds=1), httpx.MockTransport(lambda req: network.append(req))
    )
    try:
        with pytest.raises(ProviderError, match="预算记录不可用"):
            await asyncio.wait_for(provider.embed(["test"]), 1.5)
        assert not network
    finally:
        await provider.close()


async def test_all_http_retry_attempts_reserve_before_dispatch(monkeypatch):
    charged, network = [], []

    async def reserve(_):
        charged.append(len(charged) + 1)
        if len(charged) == 3:
            raise DispatchLimitExceeded("limit")

    def handler(request):
        network.append(request)
        return httpx.Response(503)

    monkeypatch.setattr(providers, "reserve_provider_call", reserve)
    provider = DashScopeProvider(profile(), httpx.MockTransport(handler))
    try:
        with pytest.raises(ProviderError, match="limit"):
            await provider.embed(["test"])
        assert len(charged) == 3 and len(network) == 2
    finally:
        await provider.close()


async def test_streaming_cannot_bypass_dispatch_limit(monkeypatch):
    network = []

    async def exhausted(_):
        raise DispatchLimitExceeded("limit")

    monkeypatch.setattr(providers, "reserve_provider_call", exhausted)
    provider = DashScopeProvider(profile(), httpx.MockTransport(lambda req: network.append(req)))
    try:
        with pytest.raises(ProviderError, match="limit"):
            async for _ in provider.stream_answer("question", [], "", ""):
                pass
        assert not network
    finally:
        await provider.close()


async def test_oversize_request_is_rejected_before_budget_or_network(monkeypatch):
    charged, network = [], []

    async def reserve(_):
        charged.append(True)

    monkeypatch.setattr(providers, "reserve_provider_call", reserve)
    provider = DashScopeProvider(
        profile(model_max_request_bytes=1000),
        httpx.MockTransport(lambda req: network.append(req)),
    )
    try:
        with pytest.raises(ProviderError, match="输入大小上限"):
            await provider.embed(["资料" * 1000])
        assert not charged and not network
    finally:
        await provider.close()


@pytest.mark.integration
async def test_real_mysql_serializes_budget_across_competing_reservations(dual_client):
    day = datetime.now(UTC).date().isoformat()
    assert await ProviderBudget.get_or_none(id=day) is None
    try:
        results = await asyncio.gather(
            *(reserve_provider_call(5) for _ in range(20)), return_exceptions=True
        )
        assert sorted(r for r in results if isinstance(r, int)) == [1, 2, 3, 4, 5]
        assert sum(isinstance(r, DispatchLimitExceeded) for r in results) == 15
        assert (await ProviderBudget.get(id=day)).dispatches == 5
    finally:
        await ProviderBudget.filter(id=day).delete()
