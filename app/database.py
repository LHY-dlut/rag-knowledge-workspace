from pathlib import Path
from threading import Lock

from tortoise import Tortoise
from tortoise.backends.base.executor import EXECUTOR_CACHE
from tortoise.context import TortoiseContext

from app.settings import Settings


def build_db_config(settings: Settings) -> dict:
    return {
        "connections": {"business": settings.business_db_url, "vectors": settings.vector_db_url},
        "apps": {
            "business": {
                "models": ["app.models"],
                "default_connection": "business",
                "migrations": "app.migrations.business",
            },
            "vectors": {
                "models": ["app.vector_models"],
                "default_connection": "vectors",
                "migrations": "app.migrations.vectors",
            },
        },
        "use_tz": True,
        "timezone": "UTC",
    }


TORTOISE_ORM = build_db_config(Settings())
_lifecycle_lock = Lock()
_database_context: TortoiseContext | None = None


def _clear_executor_cache():
    # Tortoise 1.1.8 keys SQL by alias/schema/table, without the dialect.
    # Only clear this application's aliases between closed lifetimes.
    for key in list(EXECUTOR_CACHE):
        if key[0] in {"business", "vectors"}:
            del EXECUTOR_CACHE[key]


async def init_database(settings: Settings, *, create_schema: bool = False):
    global _database_context
    # Models and the ASGI fallback are process-wide. Never let a second app
    # replace a live context; distinct environments run in separate processes.
    if not _lifecycle_lock.acquire(blocking=False):
        raise RuntimeError(
            "Database lifetime already active or initializing; "
            "run separate application environments in independent processes"
        )
    _clear_executor_cache()
    try:
        for url in (settings.business_db_url, settings.vector_db_url):
            if url.startswith("sqlite://") and url != "sqlite://:memory:":
                Path(url.removeprefix("sqlite://")).parent.mkdir(parents=True, exist_ok=True)
        # ORM 1.x needs a fallback for request tasks outside the lifespan task.
        _database_context = await Tortoise.init(
            config=build_db_config(settings), _enable_global_fallback=True
        )
        if create_schema:
            if settings.vector_db_url.startswith(("postgres://", "postgresql://")):
                await Tortoise.get_connection("vectors").execute_script(
                    "CREATE EXTENSION IF NOT EXISTS vector;"
                )
            await Tortoise.generate_schemas(safe=True)
    except BaseException:
        await close_database()
        raise


async def close_database():
    global _database_context
    if _database_context is not None:
        # Close the context we opened, rather than whichever context a caller
        # happens to inherit. Leave the lock held if closing fails.
        await _database_context.close_connections()
        _database_context = None
    else:
        # Offline schema export also uses this cleanup after direct ORM init.
        await Tortoise.close_connections()
    _clear_executor_cache()
    if _lifecycle_lock.locked():
        _lifecycle_lock.release()
