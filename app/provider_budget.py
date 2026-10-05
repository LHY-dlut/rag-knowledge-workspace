"""A shared, fail-closed daily dispatch ceiling for production providers."""

from datetime import UTC, datetime

from tortoise.transactions import in_transaction

from app.models import ProviderBudget


class DispatchLimitExceeded(RuntimeError):
    pass


async def reserve_provider_call(limit: int):
    # The business database serializes reservations across API and workers.
    # Never refund an attempted request: timeout/disconnect may still be billed.
    while True:
        day = datetime.now(UTC).date().isoformat()
        await ProviderBudget.get_or_create(id=day)
        async with in_transaction("business") as connection:
            row = await ProviderBudget.filter(id=day).using_db(connection).select_for_update().get()
            if day != datetime.now(UTC).date().isoformat():
                continue
            if row.dispatches >= limit:
                raise DispatchLimitExceeded("今日模型请求上限已达到，未发送请求")
            row.dispatches += 1
            await row.save(using_db=connection, update_fields=["dispatches"])
            return row.dispatches
