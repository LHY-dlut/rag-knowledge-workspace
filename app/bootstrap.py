"""Apply checked-in, versioned migrations; create PG extension before vector DDL."""

import asyncio
import subprocess
import sys

from tortoise import Tortoise

from app.database import close_database, init_database
from app.settings import Settings


async def pg_prepare(settings: Settings, after_migration: bool = False):
    await init_database(settings)
    try:
        if settings.vector_db_url.startswith(("postgres://", "postgresql://")):
            connection = Tortoise.get_connection("vectors")
            await connection.execute_script("CREATE EXTENSION IF NOT EXISTS vector;")
            if after_migration:
                await connection.execute_script(
                    "CREATE INDEX IF NOT EXISTS chunk_vector_scope_idx "
                    "ON chunk_vector (owner_id, kb_id, doc_id);"
                    "CREATE INDEX IF NOT EXISTS chunk_vector_hnsw_idx "
                    "ON chunk_vector USING hnsw (embedding vector_cosine_ops);"
                )
    finally:
        await close_database()


def main():
    settings = Settings()
    asyncio.run(pg_prepare(settings))
    subprocess.run(
        [sys.executable, "-m", "tortoise", "-c", "app.database.TORTOISE_ORM", "migrate"], check=True
    )
    asyncio.run(pg_prepare(settings, after_migration=True))


if __name__ == "__main__":
    main()
