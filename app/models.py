from uuid import uuid4

from tortoise import fields
from tortoise.models import Model


def new_id() -> str:
    return str(uuid4())


class Base(Model):
    id = fields.CharField(max_length=36, primary_key=True, default=new_id)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        abstract = True


class User(Base):
    username = fields.CharField(max_length=64, unique=True)
    password_hash = fields.CharField(max_length=256)
    is_active = fields.BooleanField(default=True)


class KnowledgeBase(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    name = fields.CharField(max_length=120)
    description = fields.TextField(default="")
    config = fields.JSONField(default=dict)
    revision = fields.IntField(default=0)


class Document(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36, db_index=True)
    filename = fields.CharField(max_length=255)
    file_type = fields.CharField(max_length=10)
    storage_path = fields.TextField()
    sha256 = fields.CharField(max_length=64)
    status = fields.CharField(max_length=20, default="queued", db_index=True)
    error = fields.TextField(default="")
    tags = fields.JSONField(default=list)
    metadata = fields.JSONField(default=dict)
    embedding_fingerprint = fields.CharField(max_length=160, default="")
    child_count = fields.IntField(default=0)
    parent_count = fields.IntField(default=0)
    index_revision = fields.IntField(default=0)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        unique_together = (("kb_id", "sha256"),)


class ParentChunk(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36, db_index=True)
    doc_id = fields.CharField(max_length=36, db_index=True)
    ordinal = fields.IntField()
    content = fields.TextField()
    metadata = fields.JSONField(default=dict)
    edited = fields.BooleanField(default=False)


class IngestJob(Base):
    owner_id = fields.CharField(max_length=36)
    doc_id = fields.CharField(max_length=36, db_index=True)
    kind = fields.CharField(max_length=20, default="ingest")
    payload = fields.JSONField(default=dict)
    status = fields.CharField(max_length=20, default="queued", db_index=True)
    attempts = fields.IntField(default=0)
    lease_owner = fields.CharField(max_length=36, default="")
    lease_until = fields.DatetimeField(null=True)
    error = fields.TextField(default="")


class Conversation(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36)
    title = fields.CharField(max_length=120)
    application_id = fields.CharField(max_length=36, default="", db_index=True)
    active_run_id = fields.CharField(max_length=36, default="")
    active_run_until = fields.DatetimeField(null=True)


class ChatMessage(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    conversation_id = fields.CharField(max_length=36, db_index=True)
    run_id = fields.CharField(max_length=36)
    role = fields.CharField(max_length=12)
    content = fields.TextField()
    citations = fields.JSONField(default=list)
    accepted = fields.BooleanField(default=True)


class AgentRun(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36)
    conversation_id = fields.CharField(max_length=36)
    query = fields.TextField()
    status = fields.CharField(max_length=20, default="running")
    answer = fields.TextField(default="")
    citations = fields.JSONField(default=list)
    grade_retries = fields.IntField(default=0)
    check_retries = fields.IntField(default=0)
    config_snapshot = fields.JSONField(default=dict)
    error = fields.TextField(default="")
    elapsed_ms = fields.IntField(default=0)
    request_snapshot = fields.JSONField(default=dict)
    lease_owner = fields.CharField(max_length=36, default="")
    lease_until = fields.DatetimeField(null=True)
    event_sequence = fields.IntField(default=0)


class RunEvent(Base):
    run_id = fields.CharField(max_length=36, db_index=True)
    sequence = fields.IntField()
    name = fields.CharField(max_length=30)
    data = fields.JSONField(default=dict)

    class Meta:
        unique_together = (("run_id", "sequence"),)


class AgentStep(Base):
    run_id = fields.CharField(max_length=36, db_index=True)
    node = fields.CharField(max_length=30)
    ordinal = fields.IntField()
    status = fields.CharField(max_length=20)
    input_summary = fields.JSONField(default=dict)
    output_summary = fields.JSONField(default=dict)
    elapsed_ms = fields.IntField(default=0)


class ToolDefinition(Base):
    owner_id = fields.CharField(max_length=36)
    name = fields.CharField(max_length=64)
    enabled = fields.BooleanField(default=True)

    class Meta:
        unique_together = (("owner_id", "name"),)


class PromptTemplate(Base):
    owner_id = fields.CharField(max_length=36)
    name = fields.CharField(max_length=40)
    content = fields.TextField()

    class Meta:
        unique_together = (("owner_id", "name"),)


class EvaluationRun(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36)
    status = fields.CharField(max_length=20, default="running")
    results = fields.JSONField(default=list)
    metrics = fields.JSONField(default=dict)
    provider = fields.CharField(max_length=30)
    error = fields.TextField(default="")


class Application(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36, db_index=True)
    name = fields.CharField(max_length=120)
    description = fields.TextField(default="")
    strategy = fields.CharField(max_length=20, default="hybrid")
    mode = fields.CharField(max_length=12, default="rag")
    welcome = fields.TextField(default="")
    fallback = fields.TextField(default="资料库中暂无足够依据回答这个问题。")
    prompt = fields.TextField(default="")
    enabled = fields.BooleanField(default=True)


class Feedback(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    run_id = fields.CharField(max_length=36, db_index=True)
    helpful = fields.BooleanField()
    comment = fields.TextField(default="")

    class Meta:
        unique_together = (("owner_id", "run_id"),)


class EvaluationDataset(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36, db_index=True)
    name = fields.CharField(max_length=120)
    origin = fields.CharField(max_length=30, default="manual")
    cases = fields.JSONField(default=list)


class ProviderBudget(Model):
    id = fields.CharField(primary_key=True, max_length=10)
    dispatches = fields.IntField(default=0)


class GeneratedArtifact(Base):
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36, db_index=True)
    run_id = fields.CharField(max_length=36, db_index=True)
    kind = fields.CharField(max_length=16)
    answer_sha256 = fields.CharField(max_length=64)
    payload = fields.JSONField(default=dict)
    storage_path = fields.CharField(max_length=255)
    file_sha256 = fields.CharField(max_length=64)

    class Meta:
        unique_together = (("run_id", "kind", "answer_sha256"),)
