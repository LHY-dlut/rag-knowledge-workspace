"""Test-only separate worker, with delayed demo generation for real TCP interruption tests."""

import asyncio
import os

from app.main import create_app
from app.settings import Settings


async def main():
    settings = Settings(enable_api_workers=False)
    if settings.model_provider != "demo":
        raise ValueError("This fault-injection worker permits demo only")
    app = create_app(settings, start_worker=False)
    async with app.router.lifespan_context(app):
        original = app.state.provider.stream_answer

        async def slow(*args, **kwargs):
            async for token in original(*args, **kwargs):
                await asyncio.sleep(float(os.environ.get("ACCEPTANCE_TOKEN_DELAY", "0.3")))
                yield token

        app.state.provider.stream_answer = slow
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(app.state.worker.run())
            tasks.create_task(app.state.chat_worker.run())


if __name__ == "__main__":
    asyncio.run(main())
