from orjson import loads
from tortoise import fields, migrations
from tortoise.fields.data import JSON_DUMPS
from tortoise.migrations import operations as ops

from app.models import new_id


class Migration(migrations.Migration):
    dependencies = [("business", "0003_document_index_revision")]

    initial = False

    operations = [
        ops.CreateModel(
            name="RunEvent",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("run_id", fields.CharField(db_index=True, max_length=36)),
                ("sequence", fields.IntField()),
                ("name", fields.CharField(max_length=30)),
                ("data", fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads)),
            ],
            options={
                "table": "runevent",
                "app": "business",
                "unique_together": (("run_id", "sequence"),),
                "pk_attr": "id",
            },
            bases=["Base"],
        ),
        ops.AddField(
            model_name="AgentRun",
            name="event_sequence",
            field=fields.IntField(default=0),
        ),
        ops.AddField(
            model_name="AgentRun",
            name="lease_owner",
            field=fields.CharField(default="", max_length=36),
        ),
        ops.AddField(
            model_name="AgentRun",
            name="lease_until",
            field=fields.DatetimeField(null=True, auto_now=False, auto_now_add=False),
        ),
        ops.AddField(
            model_name="AgentRun",
            name="request_snapshot",
            field=fields.JSONField(default=dict, encoder=JSON_DUMPS, decoder=loads),
        ),
    ]
