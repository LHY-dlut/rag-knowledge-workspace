from orjson import loads
from tortoise import fields, migrations
from tortoise.fields.data import JSON_DUMPS
from tortoise.migrations import operations as ops

from app.vector_models import VectorField


class Migration(migrations.Migration):
    initial = True

    operations = [
        ops.CreateModel(
            name="ChunkVector",
            fields=[
                (
                    "id",
                    fields.CharField(primary_key=True, unique=True, db_index=True, max_length=36),
                ),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("kb_id", fields.CharField(db_index=True, max_length=36)),
                ("doc_id", fields.CharField(db_index=True, max_length=36)),
                ("parent_id", fields.CharField(db_index=True, max_length=36)),
                ("content", fields.TextField(unique=False)),
                ("metadata", fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads)),
                ("embedding_fingerprint", fields.CharField(max_length=160)),
                ("embedding", VectorField(generated=False, null=False, unique=False)),
            ],
            options={"table": "chunk_vector", "app": "vectors", "pk_attr": "id"},
            bases=["Model"],
        ),
    ]
