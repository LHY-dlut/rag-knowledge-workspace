import json

import httpx
import pytest

from app.atomic_scope import review_atomic_scope
from app.check_protocol import answer_spans
from app.check_scope import PREDICATES
from app.providers import PREDICATE_CLASSIFICATION_RULES, CheckProtocolError, DashScopeProvider
from app.schemas import PredicateScopeDecision
from tests.test_predicate_ranges import selection
from tests.test_providers_parsers import remote_settings


class RoleProvider:
    supports_check_predicate_parts = True

    def __init__(self, mode="permission", source_mode="permission", changed=None, invalid=None):
        self.mode, self.source_mode, self.changed, self.invalid = (
            mode,
            source_mode,
            changed,
            invalid,
        )
        self.calls = []

    async def structured(self, schema, task, payload):
        self.calls.append((task, payload))
        if task == "check_answer_predicate_parts":
            ids = [f["fragment_id"] for f in payload["answer_fragments"]]
            if self.invalid == "missing_intro":
                ids = ids[1:]
            return schema.model_validate(
                {
                    "answer_span_id": payload["focus_answer_span_id"],
                    "predicates": [selection(ids, mode=self.mode)],
                }
            )
        assert task == "check_predicate_parts_scope" and schema is PredicateScopeDecision
        return schema.model_validate(
            {
                "answer_span_id": payload["focus_answer_span_id"],
                "checks": [
                    {
                        "predicate_id": "P1",
                        "source_mode": self.source_mode,
                        "source_voice": "fact",
                        "reason": "核验实际对应关系",
                        "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                        **{n: "changed" if n == self.changed else "preserved" for n in PREDICATES},
                    }
                ],
            }
        )


async def review(provider, answer, content):
    spans = answer_spans(answer)
    source = {"source_id": "S1", "content": content}
    return await review_atomic_scope(
        provider, answer, spans[0], spans, [source], {"S1"}, "实际问题"
    )


@pytest.mark.parametrize(
    "intro", ["据所提供资料，", "在旧规所述期间，", "本段仅适用于已审核人员："]
)
async def test_modifier_is_positioned_with_content_and_not_a_new_claim(intro):
    answer = intro + "完成审核后可以办理。[S1]"
    provider = RoleProvider()
    result, _ = await review(provider, answer, intro + "完成审核后可以办理。")
    assert all(result[n] for n in PREDICATES)
    atoms = provider.calls[1][1]["answer_predicates"]
    assert len(atoms) == 1 and len(atoms[0]["answer_parts"]) == 2
    assert atoms[0]["answer_parts"][0]["quote"] == intro
    assert provider.calls[1][1]["answer"] == answer
    assert "query" not in provider.calls[0][1] and "sources" not in provider.calls[0][1]


async def test_omitted_intro_remains_protocol_failure_without_automatic_attachment():
    provider = RoleProvider(invalid="missing_intro")
    with pytest.raises(CheckProtocolError, match="遗漏"):
        await review(provider, "旧规适用期内，审核后可以办理。[S1]", "审核后可以办理。")
    assert len(provider.calls) == 1


@pytest.mark.parametrize("failed", PREDICATES)
async def test_linked_scope_never_overrides_any_source_semantic_rejection(failed):
    result, _ = await review(
        RoleProvider(changed=failed), "在旧规期间，审核后可以办理。[S1]", "审核后可以办理。"
    )
    assert not result[failed]


@pytest.mark.parametrize(
    "answer_mode,source_mode",
    [
        ("document_limitation", "asserted"),
        ("permission", "asserted"),
        ("asserted", "document_limitation"),
        ("asserted", "planned"),
        ("asserted", "future"),
        ("permission", "possible"),
        ("undetermined", "undetermined"),
    ],
)
async def test_inconsistent_categories_are_not_silently_repaired_or_relaxed(
    answer_mode, source_mode
):
    result, _ = await review(
        RoleProvider(mode=answer_mode, source_mode=source_mode),
        "资料范围内，实际断言。[S1]",
        "实际断言。",
    )
    assert not result["modality_preserved"]


async def test_real_provider_uses_same_category_protocol_for_both_readings_without_editing_payload():
    captured = []

    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        payload = json.loads(body["messages"][1]["content"])
        if "answer_predicates" in payload:
            value = {
                "answer_span_id": "A:E1",
                "checks": [
                    {
                        "predicate_id": "P1",
                        "source_mode": "permission",
                        "source_voice": "fact",
                        "reason": "核对实际许可关系",
                        "evidence": [{"source_id": "S1", "span_id": "S1:E1"}],
                        **{n: "preserved" for n in PREDICATES},
                    }
                ],
            }
        else:
            value = {
                "answer_span_id": "A:E1",
                "predicates": [selection(["A:E1:F1", "A:E1:F2"], mode="permission")],
            }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}}]})

    provider = DashScopeProvider(remote_settings(), httpx.MockTransport(handle))
    # This existing fixture exercises the original two-reading protocol.
    # The paired source protocol has its own actual-provider regression.
    provider.supports_check_source_predicate_parts = False
    answer = "据所提供资料，审核后可以办理。[S1]"
    content = "审核后可以办理。"
    try:
        result, _ = await review(provider, answer, content)
        assert all(result[n] for n in PREDICATES) and len(captured) == 2
        for body in captured:
            assert body["messages"][0]["content"].endswith(PREDICATE_CLASSIFICATION_RULES)
            assert json.loads(body["messages"][1]["content"])["answer"] == answer
            assert body["max_tokens"] == provider.settings.model_max_output_tokens
        assert "sources" not in json.loads(captured[0]["messages"][1]["content"])
        assert captured[0]["model"] == captured[1]["model"] == provider.settings.generation_model
    finally:
        await provider.close()
