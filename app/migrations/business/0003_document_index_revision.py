from tortoise import fields, migrations
from tortoise.migrations import operations as ops


class Migration(migrations.Migration):
    dependencies = [("business", "0002_applications_feedback_datasets")]

    initial = False

    operations = [
        ops.AddField(
            model_name="Document",
            name="index_revision",
            field=fields.IntField(default=0),
        ),
    ]
