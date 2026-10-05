"""Literal source predicates, projected without questions or generated answers."""

from pydantic import Field, model_validator

from app.schemas import (
    AnswerRangePredicate,
    AnswerRangePredicates,
    PredicateScopeAssessment,
    PredicateScopeDecision,
    StrictModel,
)


class SourcePredicateParts(StrictModel):
    source_id: str = Field(min_length=1, max_length=30)
    source_span_id: str = Field(min_length=1, max_length=60)
    predicates: list[AnswerRangePredicate] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_predicates(self):
        ids = [p.predicate_id for p in self.predicates]
        if len(ids) != len(set(ids)):
            raise ValueError("Source predicate IDs must be unique")
        return self


class PairedPredicateScopeAssessment(PredicateScopeAssessment):
    source_predicate_ids: list[str] = Field(min_length=1, max_length=20)
    context_source_predicate_ids: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def unique_source_predicates(self):
        ids = self.source_predicate_ids + self.context_source_predicate_ids
        if len(ids) != len(set(ids)):
            raise ValueError("Paired fact and context predicate IDs must be unique and disjoint")
        return self


class PairedPredicateScopeDecision(PredicateScopeDecision):
    checks: list[PairedPredicateScopeAssessment] = Field(min_length=1, max_length=30)


def bind_source_parts(decision: SourcePredicateParts, source: dict, focus: dict) -> list[dict]:
    from app.answer_parts import bind_answer_parts
    from app.providers import CheckProtocolError

    if decision.source_id != source["source_id"] or decision.source_span_id != focus["span_id"]:
        raise CheckProtocolError("来源谓词投影返回其他文档或原文片段")
    projected = AnswerRangePredicates(
        answer_span_id=decision.source_span_id, predicates=decision.predicates
    )
    bound = bind_answer_parts(projected, focus, source["content"])
    return [
        {
            "source_predicate_id": f"{source['source_id']}:{focus['span_id']}:{p['predicate_id']}",
            "source_id": source["source_id"],
            "span_id": focus["span_id"],
            "source_parts": p["answer_parts"],
            "mode": p["mode"],
            "voice": p["voice"],
            "reason": p["reason"],
            "literal_binding": {
                **{k: v for k, v in p["literal_binding"].items() if k != "focus_answer_span_id"},
                "source_id": source["source_id"],
                "focus_source_span_id": focus["span_id"],
                "offset_unit": "unchanged full source Unicode codepoint",
            },
        }
        for p in bound
    ]


def validate_source_pairings(checks: list, registry: list[dict]) -> list[dict]:
    from app.evidence_roles import bind_evidence_roles

    audits = []
    for claim in checks:
        selected, context = bind_evidence_roles(claim, registry)
        modes = all(
            p["mode"] != "undetermined" and p["mode"] == claim.source_mode for p in selected
        )
        voices = all(
            p["voice"] != "undetermined" and p["voice"] == claim.source_voice for p in selected
        )
        # Never rewrite a model category to make it match the answer. This is
        # another veto, followed by the unchanged answer/source comparison.
        if not modes:
            claim.modality_preserved = "changed"
        if not voices:
            claim.attribution_preserved = "changed"
        audits.append(
            {
                "predicate_id": claim.predicate_id,
                "positioned_source_predicates": selected,
                "positioned_scope_context_predicates": context,
                "source_mode_matches_independent_parts": modes,
                "source_voice_matches_independent_parts": voices,
            }
        )
    return audits
