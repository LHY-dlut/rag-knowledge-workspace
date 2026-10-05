from orjson import loads
from tortoise import fields, migrations
from tortoise.fields.data import JSON_DUMPS
from tortoise.migrations import operations as ops

from app.models import new_id


class Migration(migrations.Migration):
    dependencies = [("business", "0001_initial")]

    initial = False

    operations = [
        ops.CreateModel(
            name="Application",
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
                ("name", fields.CharField(max_length=120)),
                ("description", fields.TextField(default="", unique=False)),
                ("strategy", fields.CharField(default="hybrid", max_length=20)),
                ("mode", fields.CharField(default="rag", max_length=12)),
                ("welcome", fields.TextField(default="", unique=False)),
                (
                    "fallback",
                    fields.TextField(default="资料库中暂无足够依据回答这个问题。", unique=False),
                ),
                ("prompt", fields.TextField(default="", unique=False)),
                ("enabled", fields.BooleanField(default=True)),
            ],
            options={"table": "application", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="EvaluationDataset",
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
                ("name", fields.CharField(max_length=120)),
                ("origin", fields.CharField(default="manual", max_length=30)),
                ("cases", fields.JSONField(default=list, encoder=JSON_DUMPS, decoder=loads)),
            ],
            options={"table": "evaluationdataset", "app": "business", "pk_attr": "id"},
            bases=["Base"],
        ),
        ops.CreateModel(
            name="Feedback",
            fields=[
                (
                    "id",
                    fields.CharField(
                        primary_key=True, default=new_id, unique=True, db_index=True, max_length=36
                    ),
                ),
                ("created_at", fields.DatetimeField(auto_now=False, auto_now_add=True)),
                ("owner_id", fields.CharField(db_index=True, max_length=36)),
                ("run_id", fields.CharField(db_index=True, max_length=36)),
                ("helpful", fields.BooleanField()),
                ("comment", fields.TextField(default="", unique=False)),
            ],
            options={
                "table": "feedback",
                "app": "business",
                "unique_together": (("owner_id", "run_id"),),
                "pk_attr": "id",
            },
            bases=["Base"],
        ),
        ops.AddField(
            model_name="Conversation",
            name="application_id",
            field=fields.CharField(default="", db_index=True, max_length=36),
        ),
    ]
