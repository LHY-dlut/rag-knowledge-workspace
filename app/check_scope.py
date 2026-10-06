"""Final scope review can only restrict a provenance-validated check."""

from difflib import SequenceMatcher

from app.check_protocol import answer_citation_manifest, answer_spans
from app.evidence_protocol import evidence_spans
from app.providers import CheckProtocolError
from app.schemas import BoundCheckJudgment, CheckAnswerProjection, CheckScopeDecision

PREDICATES = (
    "subject_predicate_preserved",
    "modality_preserved",
    "temporal_scope_preserved",
    "conditions_preserved",
    "attribution_preserved",
)


def literal_edit_view(answer_span: dict, evidence: dict) -> dict:
    """Positioned character edits are a reading aid, never an entailment score."""
    left, right = evidence["quote"], answer_span["quote"]
    return {
        "source_id": evidence["source_id"],
        "source_span_id": evidence["span_id"],
        "answer_span_id": answer_span["span_id"],
        "offset_unit": "original Unicode codepoint",
        "edits": [
            {
                "operation": tag,
                "source_start": evidence["source_start"] + i,
                "source_end": evidence["source_start"] + j,
                "source_text": left[i:j],
                "answer_start": answer_span["source_start"] + k,
                "answer_end": answer_span["source_start"] + m,
                "answer_text": right[k:m],
            }
            for tag, i, j, k, m in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
        ],
        "semantic_verdict": None,
        "original_text_unmodified": True,
    }


async def validate_check_scope(provider, judgment, answer: str, sources: list[dict], query: str):
    if not judgment.passed or not getattr(provider, "supports_check_scope_binding", False):
        return judgment, None
    if not isinstance(judgment, BoundCheckJudgment) or not judgment.evidence_protocol["passed"]:
        raise CheckProtocolError("范围核验前缺少有效的原引用位置核验")
    registry = {s["span_id"]: s for s in answer_spans(answer)}
    manifest = answer_citation_manifest(answer)
    actual = {c["answer_span_id"]: set(c["cited_source_ids"]) for c in manifest}
    known = {s["source_id"]: s for s in sources}
    if not registry or any(not ids or not ids <= known.keys() for ids in actual.values()):
        raise CheckProtocolError("范围核验不能自动补全不存在或缺失的引用")
    references = []
    for claim in judgment.checks:
        quotes = []
        for ref in claim.evidence:
            span = (
                next(
                    (
                        s
                        for s in evidence_spans(known.get(ref.source_id, {}))
                        if s["span_id"] == ref.span_id
                    ),
                    None,
                )
                if ref.source_id in known
                else None
            )
            if span is None or ref.source_id not in actual.get(claim.answer_span_id, set()):
                raise CheckProtocolError("范围核验前引用位置不再匹配")
            quotes.append({"source_id": ref.source_id, **span})
        references.append({"answer_span_id": claim.answer_span_id, "literal_evidence": quotes})
    if (
        len(references) != len(registry)
        or {r["answer_span_id"] for r in references} != registry.keys()
    ):
        raise CheckProtocolError("范围核验前上游答案覆盖不完整")
    seen = set()
    reviews, reading_views, answer_projections = [], [], []
    source_projection_cache = {}
    for reference in references:
        focus = reference["answer_span_id"]
        edits = [literal_edit_view(registry[focus], q) for q in reference["literal_evidence"]]
        reading_views.append({"answer_span_id": focus, "literal_edit_view": edits})
        if getattr(provider, "supports_check_atomic_scope", False):
            from app.atomic_scope import review_atomic_scope

            claim, projection = await review_atomic_scope(
                provider,
                answer,
                registry[focus],
                list(registry.values()),
                sources,
                actual[focus],
                query,
                source_evidence=reference["literal_evidence"],
                source_projection_cache=source_projection_cache,
            )
            seen.add(focus)
            reviews.append(claim)
            answer_projections.append(projection)
            continue
        independent = None
        if getattr(provider, "supports_check_answer_projection", False):
            # This request deliberately has no query, sources or upstream verdicts.
            independent = await provider.structured(
                CheckAnswerProjection,
                "check_answer_projection",
                {
                    "answer": answer,
                    "focus_answer_span_id": focus,
                    "answer_spans": [registry[focus]],
                    "answer_context_spans": list(registry.values()),
                },
            )
            if independent.answer_span_id != focus:
                raise CheckProtocolError("独立答案投影返回非当前片段")
            answer_projections.append(independent.model_dump())
        review = await provider.structured(
            CheckScopeDecision,
            "check_scope_binding",
            {
                "query": query,
                "answer": answer,
                "focus_answer_span_id": focus,
                "answer_spans": [registry[focus]],
                "answer_citation_manifest": [c for c in manifest if c["answer_span_id"] == focus],
                "sources": [s for s in sources if s["source_id"] in actual[focus]],
                "verified_literal_evidence": [reference],
                "answer_context_spans": list(registry.values()),
            },
        )
        if len(review.checks) != 1 or review.checks[0].answer_span_id != focus:
            raise CheckProtocolError("范围核验重复、遗漏或返回非当前答案片段")
        claim = review.checks[0]
        if claim.answer_span_id in seen:
            raise CheckProtocolError("范围核验重复答案片段")
        seen.add(claim.answer_span_id)
        if (
            len(set(claim.source_ids)) != len(claim.source_ids)
            or set(claim.source_ids) != actual[claim.answer_span_id]
        ):
            raise CheckProtocolError("范围核验来源与答案实际引用不一致")
        if independent is not None:
            if claim.projection is None:
                raise CheckProtocolError("独立答案投影核对缺少来源投影")
            modes, voices = independent.matches_source(
                claim.projection,
                no_upgrade=getattr(
                    getattr(provider, "settings", None), "scope_no_upgrade_relaxation", False
                ),
            )
            claim = claim.model_copy(
                update={
                    "modality_preserved": claim.modality_preserved and modes,
                    "attribution_preserved": claim.attribution_preserved and voices,
                    "reason": claim.reason
                    if modes and voices
                    else ("独立答案投影与原命题不一致或待定；" + claim.reason)[:300],
                }
            )
        reviews.append(claim)
    if seen != registry.keys():
        raise CheckProtocolError("范围核验遗漏答案片段")
    atomic = bool(getattr(provider, "supports_check_atomic_scope", False))
    checks = reviews if atomic else [c.model_dump() for c in reviews]
    failed = [c for c in checks if not all(c[name] for name in PREDICATES)]
    audit = {
        "passed": not failed,
        "checks": checks,
        "reading_views": reading_views,
        "literal_edits_only_in_audit_not_model_input": True,
        "scope_model_requests": sum(c.get("scope_model_requests", 2) for c in checks)
        if atomic
        else len(reviews) + len(answer_projections),
        "independent_answer_projections": answer_projections,
        "answer_projection_has_question_or_sources": False,
        "wire_rule": "five strict relations and literal predicate-bound mode/voice pairs; unknown or changed cannot retain true"
        if atomic
        else "enum per predicate plus independent modality/voice projections; only matching known projections can retain true",
        "rule": "all actual answer spans and cited source IDs; five strict semantic predicates",
        "semantic_verdict_source": "independent real model scope review; literal attribution is not entailment",
        "previous_check_preserved": True,
        "can_upgrade_rejection": False,
        "per_predicate_scope_binding": atomic,
    }
    if failed:
        reason = ";".join(c["reason"] for c in failed)[:1000]
        judgment = judgment.model_copy(
            update={"passed": False, "reason": "答案范围核验未通过：" + reason[:980]}
        )
    return judgment, audit
