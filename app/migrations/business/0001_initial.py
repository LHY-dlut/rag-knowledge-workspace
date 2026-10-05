from orjson import loads
from tortoise import fields, migrations
from tortoise.fields.data import JSON_DUMPS
from tortoise.migrations import operations as ops

from app.models import new_id


class Migration(migrations.Migration):
    initial = True

    operations = [
        ops.CreateModel(
            name="AgentRun",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("kb_id", fields.CharField(max_length=36)),
                ("conversation_id", fields.CharField(max_length=36)),
                ("query", fields.TextField(unique=False)),
                ("status", fields.CharField(default="running", max_length=20)),
                ("answer", fields.TextField(default="", unique=False)),
                ("citations", fields.JSONField(default=list, encoder=JSON_DUMPS, decoder=loads)),
                ("grade_retries", fields.IntField(default=0)),
                ("check_retries", fields.IntField(default=0)),
                (
                    "config_snapshot",
                    fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads),
                ),
                ("error", fields.TextField(default="", unique=False)),
                ("elapsed_ms", fields.IntField(default=0)),
            ],
            options={"table": "agentrun", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="AgentStep",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("run_id", fields.CharField(db_index=True, max_length=36)),
                ("node", fields.CharField(max_length=30)),
                ("ordinal", fields.IntField()),
                ("status", fields.CharField(max_length=20)),
                (
                    "input_summary",
                    fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads),
                ),
                (
                    "output_summary",
                    fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads),
                ),
                ("elapsed_ms", fields.IntField(default=0)),
            ],
            options={"table": "agentstep", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="ChatMessage",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("conversation_id", fields.CharField(db_index=True, max_length=36)),
                ("run_id", fields.CharField(max_length=36)),
                ("role", fields.CharField(max_length=12)),
                ("content", fields.TextField(unique=False)),
                ("citations", fields.JSONField(default=list, encoder=JSON_DUMPS, decoder=loads)),
                ("accepted", fields.BooleanField(default=True)),
            ],
            options={"table": "chatmessage", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="Conversation",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("kb_id", fields.CharField(max_length=36)),
                ("title", fields.CharField(max_length=120)),
                ("active_run_id", fields.CharField(default="", max_length=36)),
                (
                    "active_run_until",
                    fields.DatetimeField(null=True, auto_now=False, auto_now_add=False),
                ),
            ],
            options={"table": "conversation", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="Document",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("kb_id", fields.CharField(db_index=True, max_length=36)),
                ("filename", fields.CharField(max_length=255)),
                ("file_type", fields.CharField(max_length=10)),
                ("storage_path", fields.TextField(unique=False)),
                ("sha256", fields.CharField(max_length=64)),
                ("status", fields.CharField(default="queued", db_index=True, max_length=20)),
                ("error", fields.TextField(default="", unique=False)),
                ("tags", fields.JSONField(default=list, encoder=JSON_DUMPS, decoder=loads)),
                ("metadata", fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads)),
                ("embedding_fingerprint", fields.CharField(default="", max_length=160)),
                ("child_count", fields.IntField(default=0)),
                ("parent_count", fields.IntField(default=0)),
                ("updated_at", fields.DatetimeField(auto_now=True, auto_now_add=False)),
            ],
            options={
                "table": "document",
                "app": "business",
                "unique_together": (("kb_id", "sha256"),),
                "pk_attr": "id",
            },
            bases=["Base"],
        ),
        ops.CreateModel(
            name="EvaluationRun",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("kb_id", fields.CharField(max_length=36)),
                ("status", fields.CharField(default="running", max_length=20)),
                ("results", fields.JSONField(default=list, encoder=JSON_DUMPS, decoder=loads)),
                ("metrics", fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads)),
                ("provider", fields.CharField(max_length=30)),
                ("error", fields.TextField(default="", unique=False)),
            ],
            options={"table": "evaluationrun", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="IngestJob",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(max_length=36)),
                ("doc_id", fields.CharField(db_index=True, max_length=36)),
                ("kind", fields.CharField(default="ingest", max_length=20)),
                ("payload", fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads)),
                ("status", fields.CharField(default="queued", db_index=True, max_length=20)),
                ("attempts", fields.IntField(default=0)),
                ("lease_owner", fields.CharField(default="", max_length=36)),
                (
                    "lease_until",
                    fields.DatetimeField(null=True, auto_now=False, auto_now_add=False),
                ),
                ("error", fields.TextField(default="", unique=False)),
            ],
            options={"table": "ingestjob", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="KnowledgeBase",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("name", fields.CharField(max_length=120)),
                ("description", fields.TextField(default="", unique=False)),
                ("config", fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads)),
                ("revision", fields.IntField(default=0)),
            ],
            options={"table": "knowledgebase", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="ParentChunk",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("kb_id", fields.CharField(db_index=True, max_length=36)),
                ("doc_id", fields.CharField(db_index=True, max_length=36)),
                ("ordinal", fields.IntField()),
                ("content", fields.TextField(unique=False)),
                ("metadata", fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads)),
                ("edited", fields.BooleanField(default=False)),
            ],
            options={"table": "parentchunk", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="PromptTemplate",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(max_length=36)),
                ("name", fields.CharField(max_length=40)),
                ("content", fields.TextField(unique=False)),
            ],
            options={
                "table": "prompttemplate",
                "app": "business",
                "unique_together": (("owner_id", "name"),),
                "pk_attr": "id",
            },
            bases=["Base"],
        ),
        ops.CreateModel(
            name="ToolDefinition",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(max_length=36)),
                ("name", fields.CharField(max_length=64)),
                ("enabled", fields.BooleanField(default=True)),
            ],
            options={
                "table": "tooldefinition",
                "app": "business",
                "unique_together": (("owner_id", "name"),),
                "pk_attr": "id",
            },
            bases=["Base"],
        ),
        ops.CreateModel(
            name="User",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("username", fields.CharField(unique=True, max_length=64)),
                ("password_hash", fields.CharField(max_length=256)),
                ("is_active", fields.BooleanField(default=True)),
            ],
            options={"table": "user", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
    ]
