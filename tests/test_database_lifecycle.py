"""Database lifetimes, SQL-cache teardown, and environment isolation; no models called."""

import asyncio
from uuid import uuid4

import pytest
from tortoise.backends.base.executor import EXECUTOR_CACHE

from app import database
from app.database import close_database, init_database
from app.models import AgentRun, Conversation, User
from app.settings import Settings
from app.vector_models import ChunkVector
from tests.conftest import account, knowledge_base, uploaded


def profile(tmp_path, name):
    return Settings(
        _env_file=None,
        business_db_url=f"sqlite://{tmp_path}/{name}/business.sqlite3",
        vector_db_url=f"sqlite://{tmp_path}/{name}/vectors.sqlite3",
        upload_dir=tmp_path / name / "uploads",
    )


async def test_sql_cache_from_another_dialect_is_discarded_between_lifetimes(tmp_path):
    # Initialize model metadata before deriving the real cache key.
    await init_database(profile(tmp_path, "warmup"), create_schema=True)
    await close_database()
    stale = ("business", None, User._meta.db_table)
    foreign = ("unrelated_application", None, "unrelated_table")
    # A MySQL-style cached insert was the actual failing SQLite SQL.
    poisoned = ([], "INSERT INTO user VALUES (%s)", [], "", "", {})
    EXECUTOR_CACHE[stale] = poisoned
    EXECUTOR_CACHE[foreign] = poisoned
    try:
        await init_database(profile(tmp_path, "one"), create_schema=True)
        try:
            user = await User.create(username="cache_user", password_hash="marker")
            assert (await User.get(id=user.id)).username == "cache_user"
            assert EXECUTOR_CACHE[stale][1] != poisoned[1]
            assert EXECUTOR_CACHE[foreign] is poisoned
        finally:
            await close_database()
        assert stale not in EXECUTOR_CACHE
        assert EXECUTOR_CACHE[foreign] is poisoned
    finally:
        EXECUTOR_CACHE.pop(foreign, None)


async def test_sequential_environments_keep_business_and_vector_rows_separate(tmp_path):
    user_ids, vector_ids = {}, {}
    for name in ("one", "two", "one", "two"):
        await init_database(profile(tmp_path, name), create_schema=True)
        try:
            if name not in user_ids:
                user = await User.create(username="same_name", password_hash=f"hash_{name}")
                user_ids[name] = user.id
                vector_ids[name] = str(uuid4())
                await ChunkVector.create(
                    id=vector_ids[name],
                    owner_id=user.id,
                    kb_id=str(uuid4()),
                    doc_id=str(uuid4()),
                    parent_id=str(uuid4()),
                    content=f"only_{name}",
                    metadata={},
                    embedding_fingerprint="demo:test:1024",
                    embedding=[1.0] + [0.0] * 1023,
                )
            assert await User.all().count() == 1
            assert (await User.get(username="same_name")).id == user_ids[name]
            assert (await User.get(id=user_ids[name])).password_hash == f"hash_{name}"
            assert await ChunkVector.all().count() == 1
            assert (await ChunkVector.get(id=vector_ids[name])).content == f"only_{name}"
            other = "two" if name == "one" else "one"
            if other in user_ids:
                assert await User.get_or_none(id=user_ids[other]) is None
                assert await ChunkVector.get_or_none(id=vector_ids[other]) is None
        finally:
            await close_database()


async def test_second_live_environment_is_rejected_before_rebinding(tmp_path):
    await init_database(profile(tmp_path, "one"), create_schema=True)
    try:
        user = await User.create(username="first_owner", password_hash="first_database")
        cached = dict(EXECUTOR_CACHE)
        with pytest.raises(RuntimeError, match="independent processes"):
            await init_database(profile(tmp_path, "two"), create_schema=True)
        assert EXECUTOR_CACHE == cached
        assert (await User.get(id=user.id)).password_hash == "first_database"
        assert not (tmp_path / "two").exists()
        with pytest.raises(RuntimeError, match="independent processes"):
            await asyncio.create_task(init_database(profile(tmp_path, "two"), create_schema=True))
        assert (await User.get(id=user.id)).password_hash == "first_database"
    finally:
        await close_database()


async def test_concurrent_requests_keep_the_existing_profile_and_users(tmp_path):
    await init_database(profile(tmp_path, "one"), create_schema=True)
    try:
        users = [
            await User.create(username=f"owner_{i}", password_hash=f"marker_{i}") for i in range(2)
        ]

        async def read_owned(user, marker):
            for _ in range(20):
                row = await User.get(id=user.id)
                assert row.password_hash == marker
                await asyncio.sleep(0)

        await asyncio.gather(*(read_owned(u, f"marker_{i}") for i, u in enumerate(users)))
        assert await User.all().count() == 2
    finally:
        await close_database()


async def test_failed_initialization_releases_lifetime_for_a_clean_retry(tmp_path, monkeypatch):
    original = database.Tortoise.generate_schemas

    async def failure(*args, **kwargs):
        raise RuntimeError("injected schema failure")

    monkeypatch.setattr(database.Tortoise, "generate_schemas", failure)
    with pytest.raises(RuntimeError, match="injected schema failure"):
        await init_database(profile(tmp_path, "failed"), create_schema=True)
    monkeypatch.setattr(database.Tortoise, "generate_schemas", original)
    await init_database(profile(tmp_path, "retry"), create_schema=True)
    try:
        user = await User.create(username="after_failure", password_hash="retry_profile")
        assert (await User.get(id=user.id)).password_hash == "retry_profile"
    finally:
        await close_database()


@pytest.mark.integration
async def test_real_dual_db_concurrent_api_users_do_not_share_data(dual_client):
    """Normal authenticated requests, real MySQL/PG, deterministic Demo models."""
    profiles = []
    for i in range(2):
        name = f"isolated_owner_{i}"
        headers = await account(dual_client, name)
        kb = await knowledge_base(dual_client, headers, {"hybrid": True, "rerank": False})
        marker = f"仓储审批规则：本库负责人为运营组{i}，专用标识MARKER_{i}。"
        doc = await uploaded(dual_client, headers, kb, marker, filename=f"owner_{i}.txt")
        user = await User.get(username=name)
        conversation = await Conversation.create(owner_id=user.id, kb_id=kb, title=marker)
        run = await AgentRun.create(
            owner_id=user.id,
            kb_id=kb,
            conversation_id=conversation.id,
            query="仓储审批规则",
            status="done",
            answer=marker,
        )
        profiles.append((headers, kb, doc, run.id, marker))

    async def requests_for(index):
        headers, kb, doc, run, marker = profiles[index]
        _, other_kb, other_doc, other_run, _ = profiles[1 - index]
        for _ in range(3):
            own = await dual_client.get(f"/api/knowledge-bases/{kb}/documents", headers=headers)
            assert own.status_code == 200
            assert [d["id"] for d in own.json()["data"]] == [doc]
            search = await dual_client.post(
                "/api/retrieval/debug",
                headers=headers,
                json={"kb_id": kb, "query": "仓储审批规则"},
            )
            assert search.status_code == 200
            candidates = search.json()["data"]["candidates"]
            assert candidates and all(c["doc_id"] == doc for c in candidates)
            assert all(marker in c["content"] for c in candidates)
            trace = await dual_client.get(f"/api/runs/{run}", headers=headers)
            assert trace.status_code == 200 and trace.json()["data"]["answer"] == marker
            forbidden = await asyncio.gather(
                dual_client.get(f"/api/knowledge-bases/{other_kb}/documents", headers=headers),
                dual_client.delete(f"/api/documents/{other_doc}", headers=headers),
                dual_client.get(f"/api/runs/{other_run}", headers=headers),
                dual_client.post(
                    "/api/retrieval/debug",
                    headers=headers,
                    json={"kb_id": other_kb, "query": "仓储审批规则"},
                ),
            )
            assert all(response.status_code == 404 for response in forbidden)

    await asyncio.gather(requests_for(0), requests_for(1))
