from tortoise import fields, migrations
from tortoise.migrations import operations as ops


class Migration(migrations.Migration):
    dependencies = [("business", "0004_durable_chat")]
    initial = False
    operations = [
        ops.CreateModel(
            name="ProviderBudget",
            fields=[
                ("id", fields.CharField(primary_key=True, max_length=10)),
                ("dispatches", fields.IntField(default=0)),
            ],
            options={"table": "providerbudget", "app": "business", "pk_attr": "id"},
            bases=[],
        )
    ]
