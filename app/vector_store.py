import asyncio
import json
from collections.abc import Sequence

from tortoise.transactions import in_transaction

from app.schemas import Candidate, ChildRecord
from app.settings import Settings
from app.vector_models import ChunkVector, validate_vector


class VectorStore:
    def __init__(self, settings: Settings):
        self.postgres = settings.vector_db_url.startswith(("postgres://", "postgresql://"))

    async def replace_document(self, doc_id: str, records: Sequence[ChildRecord]):
        # One PG/SQLite transaction. MySQL publication happens only afterwards.
        async with in_transaction("vectors") as connection:
            await ChunkVector.filter(doc_id=doc_id).using_db(connection).delete()
            if records:
                await ChunkVector.bulk_create(
                    [ChunkVector(**r.model_dump()) for r in records],
                    using_db=connection,
                    batch_size=100,
                )

    async def replace_parent(self, parent_id: str, records: Sequence[ChildRecord]):
        async with in_transaction("vectors") as connection:
            await ChunkVector.filter(parent_id=parent_id).using_db(connection).delete()
            if records:
                await ChunkVector.bulk_create(
                    [ChunkVector(**r.model_dump()) for r in records],
                    using_db=connection,
                    batch_size=100,
                )

    async def delete_document(self, doc_id: str):
        await ChunkVector.filter(doc_id=doc_id).delete()

    async def corpus(
        self, owner_id: str, kb_id: str, doc_ids: list[str], fingerprint: str, limit: int
    ):
        if not doc_ids:
            return []
        query = ChunkVector.filter(
            owner_id=owner_id, kb_id=kb_id, doc_id__in=doc_ids, embedding_fingerprint=fingerprint
        )
        if await query.count() > limit:
            raise ValueError("BM25 语料超过 SPARSE_MAX_CHUNKS；请部署独立稀疏索引服务")
        rows = await query.order_by("id").values(
            "id",
            "owner_id",
            "kb_id",
            "doc_id",
            "parent_id",
            "content",
            "metadata",
            "embedding_fingerprint",
        )
        return [Candidate(**row) for row in rows]

    async def dense(
        self,
        owner_id: str,
        kb_id: str,
        doc_ids: list[str],
        fingerprint: str,
        vector: list[float],
        top_k: int,
    ):
        validate_vector(vector)
        if not doc_ids:
            return []
        if self.postgres:
            async with in_transaction("vectors") as connection:
                # ANN filtering is applied after scanning. Iterative scanning
                # prevents small tenant/KB scopes from starving the candidate set.
                await connection.execute_query("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                _, rows = await connection.execute_query(
                    "SELECT id, owner_id, kb_id, doc_id, parent_id, content, metadata, "
                    "embedding_fingerprint, 1 - (embedding <=> $1::vector) AS cosine_score "
                    "FROM chunk_vector WHERE owner_id=$2 AND kb_id=$3 "
                    "AND doc_id=ANY($4::varchar[]) AND embedding_fingerprint=$5 "
                    "ORDER BY embedding <=> $1::vector LIMIT $6",
                    [json.dumps(vector), owner_id, kb_id, doc_ids, fingerprint, top_k],
                )
            return [
                Candidate(
                    **{
                        **row,
                        "metadata": json.loads(row["metadata"])
                        if isinstance(row["metadata"], str)
                        else row["metadata"],
                    }
                )
                for row in rows
            ]
        # Exact cosine demo adapter, bounded by the corpus budget in the retriever.
        rows = await ChunkVector.filter(
            owner_id=owner_id, kb_id=kb_id, doc_id__in=doc_ids, embedding_fingerprint=fingerprint
        ).all()

        def rank():
            import numpy as np

            if not rows:
                return []
            matrix = np.asarray([row.embedding for row in rows], dtype=float)
            scores = (
                matrix
                @ np.asarray(vector)
                / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))
            )
            indices = np.argsort(-scores, kind="stable")[:top_k]
            return [
                Candidate(
                    **{
                        k: getattr(rows[i], k)
                        for k in (
                            "id",
                            "owner_id",
                            "kb_id",
                            "doc_id",
                            "parent_id",
                            "content",
                            "metadata",
                            "embedding_fingerprint",
                        )
                    },
                    cosine_score=float(scores[i]),
                )
                for i in indices
            ]

        return await asyncio.to_thread(rank)
