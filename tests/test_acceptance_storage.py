"""Real storage acceptance, including an observed filtered HNSW execution plan."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from tortoise import Tortoise
from tortoise.transactions import in_transaction

from app.models import Document
from app.schemas import ChildRecord
from app.vector_models import ChunkVector
from tests import test_api, test_management
from tests.conftest import account, knowledge_base, uploaded


@pytest.mark.integration
async def test_real_upload_replace_rebuild_delete(dual_client):
    await test_management.test_file_replacement_publishes_new_revision_and_removes_old_vectors(
        dual_client
    )
    doc = await Document.all().first()
    from app.models import User
    from app.security import issue_token

    user = await User.get(id=doc.owner_id)
    headers = {"Authorization": "Bearer " + issue_token(user.id, dual_client.app.state.settings)}
    response = await dual_client.post(f"/api/documents/{doc.id}/reindex", headers=headers)
    assert response.status_code == 202
    await dual_client.app.state.worker.tick()
    await doc.refresh_from_db()
    assert doc.status == "ready" and doc.index_revision == 3
    response = await dual_client.delete(f"/api/documents/{doc.id}", headers=headers)
    assert response.status_code == 200
    assert not await Document.filter(id=doc.id).exists()
    assert not await ChunkVector.filter(doc_id=doc.id).exists()


@pytest.mark.integration
async def test_real_permissions_and_metadata(dual_client):
    await test_api.test_acl_every_resource_and_metadata_filter(dual_client)


@pytest.mark.integration
async def test_filtered_hnsw_and_server_dimension_constraint(dual_client, tmp_path):
    settings = dual_client.app.state.settings
    headers = await account(dual_client)
    kb = await knowledge_base(dual_client, headers)
    doc_id = await uploaded(dual_client, headers, kb)
    doc = await Document.get(id=doc_id)
    records = []
    relevant_ids = []
    for i in range(260):
        target = i >= 240
        child_id = str(uuid4())
        if target:
            relevant_ids.append(child_id)
        records.append(
            ChildRecord(
                id=child_id,
                owner_id=doc.owner_id if target else "other-tenant",
                kb_id=kb,
                doc_id=doc_id,
                parent_id=str(uuid4()),
                content=f"controlled vector {i}",
                embedding=[1.0, 0.01 + (i - 240) * 0.001 if target else (i + 1) * 0.000001]
                + [0.0] * 1022,
                embedding_fingerprint=settings.embedding_fingerprint,
                metadata={"fixture": True},
            )
        )
    await dual_client.app.state.store.replace_document(doc_id, records)
    pg = Tortoise.get_connection("vectors")
    sql = (
        "SELECT id FROM chunk_vector WHERE owner_id=$2 AND kb_id=$3 "
        "AND doc_id=ANY($4::varchar[]) AND embedding_fingerprint=$5 "
        "ORDER BY embedding <=> $1::vector LIMIT $6"
    )
    vector = [1.0] + [0.0] * 1023
    params = [json.dumps(vector), doc.owner_id, kb, [doc_id], settings.embedding_fingerprint, 10]
    _, default_plan = await pg.execute_query("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params)
    async with in_transaction("vectors") as conn:
        await conn.execute_script(
            "SET LOCAL enable_seqscan=off; SET LOCAL enable_bitmapscan=off; SET LOCAL enable_sort=off; SET LOCAL hnsw.iterative_scan=strict_order;"
        )
        _, plan = await conn.execute_query("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params)
        matches = await dual_client.app.state.store.dense(
            doc.owner_id, kb, [doc_id], settings.embedding_fingerprint, vector, 10
        )
    plan_text = json.dumps(plan, default=str)
    assert "chunk_vector_hnsw_idx" in plan_text
    assert len(matches) == 10
    assert [c.id for c in matches] == relevant_ids[:10]
    assert all(c.owner_id == doc.owner_id and c.doc_id == doc_id for c in matches)
    with pytest.raises(Exception, match="1024|dimensions"):
        await pg.execute_query(
            "UPDATE chunk_vector SET embedding='[1,2]'::vector WHERE id=$1", [relevant_ids[0]]
        )
    evidence = {
        "rows": 260,
        "other_tenant_nearer_rows": 240,
        "matching_rows": 20,
        "returned": len(matches),
        "ordered_target_ids_match": True,
        "production_default_plan": default_plan,
        "forced_hnsw_plan": plan,
        "dimension_1024_server_rejection": True,
        "note": "Small-table planner may choose scope btree + sort. Session-only planner settings force actual HNSW for this test; application defaults unchanged.",
    }
    path = Path(os.environ.get("ACCEPTANCE_EVIDENCE_DIR", str(tmp_path))) / "hnsw.json"
    if path.parent.exists():
        path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
