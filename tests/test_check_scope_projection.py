import json

import httpx
import pytest
from pydantic import ValidationError

from app.check_scope import PREDICATES
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import CheckScopeAssessment, CheckScopeDecision
from tests.test_check_scope_focus import assessment
from tests.test_providers_parsers import remote_settings


@pytest.mark.parametrize(
    "changes",
    [
        {"source_modes": ["future"]},
        {"source_modes": ["asserted", "future"]},
        {"source_voices": ["opinion"]},
        {"source_voices": ["fact", "opinion"]},
        {"source_modes": ["undetermined"], "answer_modes": ["undetermined"]},
        {"source_voices": ["undetermined"], "answer_voices": ["undetermined"]},
    ],
)
def test_comparative_positive_cannot_override_projection_mismatch_or_unknown(changes):
    value = assessment()
    value["checks"][0]["projection"].update(changes)
    wire = CheckScopeAssessment.model_validate(value)
    assert all(getattr(wire.checks[0], p) == "preserved" for p in PREDICATES)
    check = wire.to_decision().checks[0]
    assert not all(getattr(check, p) for p in PREDICATES)
    assert "投影不一致或待定" in check.reason
    assert check.projection.model_dump() == value["checks"][0]["projection"]


def test_matching_multi_predicate_projections_preserve_valid_comparative_result():
    value = assessment()
    value["checks"][0]["projection"].update(
        source_modes=["asserted", "discussion"], answer_modes=["discussion", "asserted"]
    )
    check = CheckScopeAssessment.model_validate(value).to_decision().checks[0]
    assert all(getattr(check, p) for p in PREDICATES)


@pytest.mark.parametrize("predicate", PREDICATES)
def test_matching_projections_never_upgrade_any_original_negative_relation(predicate):
    wire = CheckScopeAssessment.model_validate(assessment(**{predicate: "changed"}))
    check = wire.to_decision().checks[0]
    assert not getattr(check, predicate)
    assert check.projection.preserves_modes() and check.projection.preserves_voices()


@pytest.mark.parametrize(
    "field", ["source_modes", "answer_modes", "source_voices", "answer_voices"]
)
def test_duplicate_projection_tags_are_invalid(field):
    value = assessment()
    tags = value["checks"][0]["projection"][field]
    tags.append(tags[0])
    with pytest.raises(ValidationError, match="must be unique"):
        CheckScopeAssessment.model_validate(value)


def test_missing_new_projection_is_rejected_not_assumed_matching():
    value = assessment()
    del value["checks"][0]["projection"]
    with pytest.raises(ValidationError):
        CheckScopeAssessment.model_validate(value)


async def test_actual_wire_response_is_restricted_by_backend_not_overall_model_verdict():
    value = assessment()
    value["checks"][0]["projection"]["source_modes"] = ["future"]

    def handle(request):
        body = json.loads(request.content)
        assert '"CheckScopeProjection"' in body["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        check = (await provider.structured(CheckScopeDecision, "check_scope_binding", {})).checks[0]
        assert check.subject_predicate_preserved and not check.modality_preserved
        assert check.projection.source_modes == ["future"]
    finally:
        await provider.close()


def test_backend_negative_explanation_keeps_original_input_and_hard_length_limit():
    value = assessment(reason="证" * 300)
    # The helper intentionally fixes its reason; set the raw input explicitly.
    value["checks"][0]["reason"] = "证" * 300
    value["checks"][0]["projection"]["source_modes"] = ["future"]
    wire = CheckScopeAssessment.model_validate(value)
    result = wire.to_decision().checks[0]
    assert wire.checks[0].reason == "证" * 300
    assert len(result.reason) == 300 and not result.modality_preserved


async def test_invalid_oversized_model_reason_still_causes_protocol_error():
    value = assessment()
    value["checks"][0]["reason"] = "证" * 301
    provider = DashScopeProvider(
        remote_settings(),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(value)}}]}
            )
        ),
    )
    try:
        with pytest.raises(CheckProtocolError):
            await provider.structured(CheckScopeDecision, "check_scope_binding", {})
    finally:
        await provider.close()
