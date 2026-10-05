import pytest
from pydantic import ValidationError

from app.schemas import GradeDecision


@pytest.mark.parametrize(
    "passed,support",
    [
        ("true", "supported_answer"),
        (1, "supported_answer"),
        ("false", "insufficient"),
        (0, "insufficient"),
        (None, "insufficient"),
        ([], "insufficient"),
        ({"value": True}, "insufficient"),
    ],
)
def test_grade_boolean_is_not_coerced_even_with_consistent_support(passed, support):
    payload = {
        "passed": passed,
        "support": support,
        "reason": "结构校验",
        "answer_scope": "仅说明营业时间",
        "evidence": [
            {
                "source_id": "S1",
                "quote": "门店营业至20时。",
                "subject": "门店",
                "attribute": "营业时间",
            }
        ],
    }
    with pytest.raises(ValidationError) as exc:
        GradeDecision.model_validate(payload)
    assert any(e["loc"] == ("passed",) and e["type"] == "bool_type" for e in exc.value.errors())
