"""Optional real database integration; never treat SQLite as a PGVector test."""

import os
from uuid import uuid4

import pytest

from app.database import close_database, init_database
from app.models import ParentChunk
from app.schemas import ChildRecord
from app.settings import Settings
from app.vector_store import VectorStore


@pytest.mark.integration
async def test_real_mysql_pgvector_roundtrip(tmp_path):
    mysql_url = os.environ.get("TEST_BUSINESS_DB_URL")
    pg_url = os.environ.get("TEST_VECTOR_DB_URL")
    if not mysql_url or not pg_url:
        pytest.skip("Set TEST_BUSINESS_DB_URL and TEST_VECTOR_DB_URL; apply migrations first")
    settings = Settings(
        _env_file=None,
        app_mode="production",
        model_provider="demo",
        business_db_url=mysql_url,
        vector_db_url=pg_url,
        jwt_secret=uuid4().hex + uuid4().hex,
        upload_dir=tmp_path,
    )
    await init_database(settings)
    owner, kb, doc, parent, child = (str(uuid4()) for _ in range(5))
    store = VectorStore(settings)
    vector = [1.0] + [0.0] * 1023
    try:
        await ParentChunk.create(
            id=parent,
            owner_id=owner,
            kb_id=kb,
            doc_id=doc,
            ordinal=0,
            content="MySQL父块",
            metadata={"page": 2},
        )
        await store.replace_document(
            doc,
            [
                ChildRecord(
                    id=child,
                    owner_id=owner,
                    kb_id=kb,
                    doc_id=doc,
                    parent_id=parent,
                    content="PGVector子块",
                    metadata={"page": 2},
                    embedding_fingerprint=settings.embedding_fingerprint,
                    embedding=vector,
                )
            ],
        )
        matches = await store.dense(owner, kb, [doc], settings.embedding_fingerprint, vector, 20)
        assert matches[0].id == child and matches[0].cosine_score == pytest.approx(1)
        assert matches[0].metadata["page"] == 2
        assert (
            await store.dense(str(uuid4()), kb, [doc], settings.embedding_fingerprint, vector, 20)
            == []
        )
        stored = await ParentChunk.get(id=matches[0].parent_id)
        assert stored.content == "MySQL父块"
    finally:
        await store.delete_document(doc)
        await ParentChunk.filter(id=parent).delete()
        await close_database()
