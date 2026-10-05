from tortoise import fields, migrations
from tortoise.migrations import operations as ops

from app.models import new_id


class Migration(migrations.Migration):
    dependencies = [("business", "0005_provider_budget")]
    initial = False
    operations = [
        ops.CreateModel(
            name="GeneratedArtifact",
            fields=[
                ("id", fields.CharField(primary_key=True, max_length=36, default=new_id)),
                ("created_at", fields.DatetimeField(auto_now_add=True)),
                ("owner_id", fields.CharField(max_length=36, db_index=True)),
                ("kb_id", fields.CharField(max_length=36, db_index=True)),
                ("run_id", fields.CharField(max_length=36, db_index=True)),
                ("kind", fields.CharField(max_length=16)),
                ("answer_sha256", fields.CharField(max_length=64)),
                ("payload", fields.JSONField(default=dict)),
                ("storage_path", fields.CharField(max_length=255)),
                ("file_sha256", fields.CharField(max_length=64)),
            ],
            options={
                "table": "generatedartifact",
                "app": "business",
                "pk_attr": "id",
                "unique_together": (("run_id", "kind", "answer_sha256"),),
            },
            bases=[],
        )
    ]
