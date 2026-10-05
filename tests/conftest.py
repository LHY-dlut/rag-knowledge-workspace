import httpx
import pytest
import pytest_asyncio

from app.main import create_app
from app.settings import Settings


@pytest_asyncio.fixture
async def client(tmp_path):
    settings = Settings(
        _env_file=None,
        business_db_url=f"sqlite://{tmp_path}/business.sqlite3",
        vector_db_url=f"sqlite://{tmp_path}/vectors.sqlite3",
        upload_dir=tmp_path / "uploads",
        model_provider="demo",
        app_mode="demo",
        worker_poll_seconds=0.1,
    )
    app = create_app(settings, start_worker=False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            c.app = app
            yield c


async def account(client, name="tester"):
    response = await client.post(
        "/api/auth/register", json={"username": name, "password": "test-password-123"}
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['data']['access_token']}"}


async def knowledge_base(client, headers, config=None):
    body = {"name": "测试知识库"}
    if config:
        body["config"] = config
    response = await client.post("/api/knowledge-bases", headers=headers, json=body)
    assert response.status_code == 201, response.text
    return response.json()["data"]["id"]


async def uploaded(
    client,
    headers,
    kb_id,
    text="公司差旅政策规定，员工出差后必须在7天内提交报销申请。\n\n报销需要发票和审批单。",
    filename="policy.md",
):
    response = await client.post(
        f"/api/knowledge-bases/{kb_id}/documents",
        headers=headers,
        files={"file": (filename, text.encode(), "text/markdown")},
        data={"tags": "制度"},
    )
    assert response.status_code == 202, response.text
    payload = response.json()["data"]
    await client.app.state.worker.tick()
    status = await client.get(f"/api/jobs/{payload['job_id']}", headers=headers)
    assert status.json()["data"]["status"] == "done", status.text
    return payload["document"]["id"]


def parse_events(response):
    normalized = response.text.replace("\r\n", "\n")
    import json

    result = []
    for frame in normalized.split("\n\n"):
        name, data = None, []
        for line in frame.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name and data:
            result.append((name, json.loads("\n".join(data))))
    return result


@pytest_asyncio.fixture
async def dual_client(tmp_path):
    import os
    from uuid import uuid4

    from tortoise import Tortoise

    from app.models import AgentRun, AgentStep, RunEvent, User
    from app.vector_models import ChunkVector

    mysql = os.environ.get("TEST_BUSINESS_DB_URL")
    postgres = os.environ.get("TEST_VECTOR_DB_URL")
    if not mysql or not postgres or os.environ.get("ACCEPTANCE_ISOLATED_DATABASES") != "1":
        pytest.skip("Needs real MySQL/PG and explicit ACCEPTANCE_ISOLATED_DATABASES=1")
    settings = Settings(
        _env_file=None,
        app_mode="production",
        model_provider="demo",
        business_db_url=mysql,
        vector_db_url=postgres,
        jwt_secret=uuid4().hex + uuid4().hex,
        upload_dir=tmp_path / "uploads",
        worker_poll_seconds=0.1,
        enable_api_workers=False,
    )
    app = create_app(settings, start_worker=False)
    async with app.router.lifespan_context(app):
        if await User.all().count():
            pytest.fail("Acceptance DB must start empty; refusing to use existing user data")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                client.app = app
                yield client
        finally:
            owner_ids = await User.all().values_list("id", flat=True)
            run_ids = await AgentRun.filter(owner_id__in=owner_ids).values_list("id", flat=True)
            await RunEvent.filter(run_id__in=run_ids).delete()
            await AgentStep.filter(run_id__in=run_ids).delete()
            for model in Tortoise.apps["business"].values():
                if "owner_id" in model._meta.fields_map:
                    await model.filter(owner_id__in=owner_ids).delete()
            await ChunkVector.filter(owner_id__in=owner_ids).delete()
            await User.filter(id__in=owner_ids).delete()
