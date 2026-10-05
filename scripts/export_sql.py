"""Generate dialect DDL without connecting to either database."""

import asyncio
from pathlib import Path

from tortoise import Tortoise

from app.database import build_db_config, close_database
from app.settings import Settings


async def main():
    settings = Settings(
        _env_file=None,
        business_db_url="mysql://rag:unused@127.0.0.1/rag_business",
        vector_db_url="postgres://rag:unused@127.0.0.1/rag_vectors",
    )
    await Tortoise.init(config=build_db_config(settings), init_connections=False)
    try:
        for alias in ("business", "vectors"):
            connection = Tortoise.get_connection(alias)
            generator = connection.schema_generator(connection)
            sql = generator.get_create_schema_sql(safe=True)
            if alias == "vectors":
                sql = (
                    "CREATE EXTENSION IF NOT EXISTS vector;\n"
                    + sql
                    + (
                        "\nCREATE INDEX IF NOT EXISTS chunk_vector_hnsw_idx "
                        "ON chunk_vector USING hnsw (embedding vector_cosine_ops);\n"
                    )
                )
            Path(f"infra/{alias}_schema.sql").write_text(sql, encoding="utf-8")
            print(f"Generated infra/{alias}_schema.sql")
    finally:
        await close_database()


if __name__ == "__main__":
    asyncio.run(main())
