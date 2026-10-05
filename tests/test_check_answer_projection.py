import json

import httpx
import pytest
from pydantic import ValidationError

from app.check_scope import PREDICATES, validate_check_scope
from app.providers import CheckProtocolError, DashScopeProvider
from app.schemas import CheckAnswerProjection, CheckScopeDecision, CheckScopeProjection
from tests.test_check_scope import ANSWER, SOURCE, scope_reply, upstream
from tests.test_providers_parsers import remote_settings


class ProjectionProvider:
    supports_check_scope_binding = True
    supports_check_answer_projection = True

    def __init__(self, answer_modes=None, source_modes=None, voices=None, failed=None):
        self.calls = []
        self.answer_modes = answer_modes or ["asserted"]
        self.source_modes = source_modes or ["future"]
        self.voices = voices or ["fact"]
        self.failed = failed

    async def structured(self, schema, task, payload):
        self.calls.append((task, payload))
        if task == "check_answer_projection":
            assert schema is CheckAnswerProjection
            assert set(payload) == {
                "answer",
                "focus_answer_span_id",
                "answer_spans",
                "answer_context_spans",
            }
            return CheckAnswerProjection(
                answer_span_id=payload["focus_answer_span_id"],
                modes=self.answer_modes,
                voices=self.voices,
                reason="仅读当前实际答案，不借原文补语气",
            )
        assert task == "check_scope_binding" and schema is CheckScopeDecision
        # Simulate the formerly observed joint-reading error: both projections
        # borrow the source modality, despite the actual answer asserting it.
        claim = scope_reply(payload, **({self.failed: False} if self.failed else {}))
        claim.checks[0].projection = CheckScopeProjection(
            answer_modes=self.source_modes,
            source_modes=self.source_modes,
            answer_voices=["opinion"],
            source_voices=["opinion"],
        )
        return claim


@pytest.mark.parametrize(
    "modes,voices",
    [
        (["asserted"], ["opinion"]),
        (["future"], ["fact"]),
        (["undetermined"], ["opinion"]),
        (["future"], ["undetermined"]),
    ],
)
async def test_joint_positive_cannot_borrow_missing_modality_or_attribution(modes, voices):
    provider = ProjectionProvider(answer_modes=modes, voices=voices)
    original = upstream()
    final, audit = await validate_check_scope(provider, original, ANSWER, [SOURCE], "私有问题")
    assert not final.passed and original.passed
    assert len(provider.calls) == audit["scope_model_requests"] == 2
    answer_input = json.dumps(provider.calls[0][1], ensure_ascii=False)
    assert "私有问题" not in answer_input and SOURCE["content"] not in answer_input
    assert audit["independent_answer_projections"][0]["modes"] == modes
    assert not audit["can_upgrade_rejection"]


@pytest.mark.parametrize("failed", PREDICATES)
async def test_matching_independent_reading_does_not_upgrade_any_scope_negative(failed):
    provider = ProjectionProvider(answer_modes=["future"], voices=["opinion"], failed=failed)
    final, audit = await validate_check_scope(
        provider, upstream(), ANSWER, [SOURCE], "资料如何说？"
    )
    assert not final.passed and not audit["checks"][0][failed]


async def test_matching_independent_reading_retains_existing_positive_and_positions():
    provider = ProjectionProvider(answer_modes=["future"], voices=["opinion"])
    original = upstream()
    final, audit = await validate_check_scope(provider, original, ANSWER, [SOURCE], "如何表述？")
    assert final.passed and final.evidence_protocol == original.evidence_protocol
    assert all(audit["checks"][0][name] for name in PREDICATES)
    assert provider.calls[1][1]["sources"][0] == SOURCE
    assert "independent_answer_projections" not in provider.calls[1][1]


@pytest.mark.parametrize(
    "change",
    [
        {"modes": []},
        {"modes": ["asserted", "asserted"]},
        {"modes": ["unknown"]},
        {"voices": ["fact", "fact"]},
        {"voices": [True]},
        {"reason": "证" * 301},
    ],
)
def test_invalid_independent_wire_cannot_be_repaired_to_a_success(change):
    with pytest.raises(ValidationError):
        CheckAnswerProjection.model_validate(
            {
                "answer_span_id": "A:E1",
                "modes": ["asserted"],
                "voices": ["fact"],
                "reason": "原文未输入",
                **change,
            }
        )


async def test_wrong_answer_id_stops_before_comparative_request():
    class Wrong(ProjectionProvider):
        async def structured(self, schema, task, payload):
            result = await super().structured(schema, task, payload)
            return result.model_copy(update={"answer_span_id": "A:E99"})

    provider = Wrong()
    with pytest.raises(CheckProtocolError, match="非当前片段"):
        await validate_check_scope(provider, upstream(), ANSWER, [SOURCE], "资料？")
    assert len(provider.calls) == 1


async def test_missing_source_projection_stays_protocol_failure():
    class Missing(ProjectionProvider):
        async def structured(self, schema, task, payload):
            result = await super().structured(schema, task, payload)
            if task == "check_scope_binding":
                result.checks[0].projection = None
            return result

    with pytest.raises(CheckProtocolError, match="缺少来源投影"):
        await validate_check_scope(Missing(), upstream(), ANSWER, [SOURCE], "资料？")


async def test_upstream_rejection_never_requests_projection_or_recovers_draft():
    provider = ProjectionProvider()
    original = upstream().model_copy(update={"passed": False})
    final, audit = await validate_check_scope(provider, original, ANSWER, [SOURCE], "资料？")
    assert final is original and audit is None and provider.calls == []


async def test_actual_provider_sends_only_answer_data_and_keeps_invalid_output_a_failure():
    payload = {
        "focus_answer_span_id": "A:E1",
        "answer_spans": [{"quote": ANSWER}],
        "answer_context_spans": [],
    }

    def handle(request):
        body = json.loads(request.content)
        assert json.loads(body["messages"][1]["content"]) == payload
        assert '"CheckAnswerProjection"' in body["messages"][0]["content"]
        value = {
            "answer_span_id": "A:E1",
            "modes": ["asserted", "asserted"],
            "voices": ["fact"],
            "reason": "重复无效",
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    try:
        with pytest.raises(CheckProtocolError):
            await provider.structured(CheckAnswerProjection, "check_answer_projection", payload)
    finally:
        await provider.close()


async def test_projection_transport_timeout_propagates_without_a_late_positive():
    class Timeout(ProjectionProvider):
        async def structured(self, *args):
            raise TimeoutError("synthetic provider timeout")

    with pytest.raises(TimeoutError):
        await validate_check_scope(Timeout(), upstream(), ANSWER, [SOURCE], "资料？")


async def test_answer_only_input_retains_original_paragraphs_and_whitespace():
    answer = "\t预约已成为服务发展的主要推动力。[S1]  \n\n"
    provider = ProjectionProvider(answer_modes=["future"], voices=["opinion"])
    original = upstream(answer)
    assert original.passed
    await validate_check_scope(provider, original, answer, [SOURCE], "问题不传入")
    assert provider.calls[0][1]["answer"] == answer
    assert provider.calls[0][1]["answer_spans"][0]["source_start"] == 1
