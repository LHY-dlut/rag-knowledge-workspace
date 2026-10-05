import json
import math
from typing import Any

from tortoise import fields
from tortoise.fields import Field
from tortoise.models import Model


def validate_vector(value: list[float]) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 1024
        or not all(type(x) in (int, float) and math.isfinite(x) for x in value)
    ):
        raise ValueError("Embedding must contain exactly 1024 finite values")
    norm_squared = sum(x * x for x in value)
    if not math.isfinite(norm_squared) or norm_squared == 0:
        raise ValueError("Embedding must have a finite nonzero squared norm")
    return value


class VectorField(Field[list[float]]):
    SQL_TYPE = "VECTOR(1024)"

    class _db_sqlite:
        SQL_TYPE = "TEXT"

    def to_db_value(self, value: list[float], instance: Any) -> str:
        return json.dumps(validate_vector(value), separators=(",", ":"))

    def to_python_value(self, value: Any) -> list[float]:
        return json.loads(value) if isinstance(value, str) else list(value)


class ChunkVector(Model):
    id = fields.CharField(max_length=36, primary_key=True)
    owner_id = fields.CharField(max_length=36, db_index=True)
    kb_id = fields.CharField(max_length=36, db_index=True)
    doc_id = fields.CharField(max_length=36, db_index=True)
    parent_id = fields.CharField(max_length=36, db_index=True)
    content = fields.TextField()
    metadata = fields.JSONField(default=dict)
    embedding_fingerprint = fields.CharField(max_length=160)
    embedding = VectorField()

    class Meta:
        table = "chunk_vector"
