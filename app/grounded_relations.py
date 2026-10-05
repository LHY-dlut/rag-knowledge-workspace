"""Relations are judged separately from independent source classifications."""

from typing import Literal

from pydantic import Field, model_validator

from app.schemas import CheckEvidence, StrictModel
from app.source_parts import PairedPredicateScopeDecision

Relation = Literal["preserved", "changed", "undetermined"]


class GroundedRelation(StrictModel):
    predicate_id: str = Field(min_length=1, max_length=60)
    evidence: list[CheckEvidence] = Field(min_length=1, max_length=20)
    source_predicate_ids: list[str] = Field(min_length=1, max_length=20)
    context_source_predicate_ids: list[str] = Field(default_factory=list, max_length=20)
    subject_predicate_preserved: Relation
    modality_preserved: Relation
    temporal_scope_preserved: Relation
    conditions_preserved: Relation
    attribution_preserved: Relation
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def unique_references(self):
        ids = self.source_predicate_ids + self.context_source_predicate_ids
        if len(ids) != len(set(ids)):
            raise ValueError("Grounded fact and context predicates must be unique and disjoint")
        refs = [(r.source_id, r.span_id) for r in self.evidence]
        if len(refs) != len(set(refs)):
            raise ValueError("Grounded evidence positions must be unique")
        return self


class GroundedRelations(StrictModel):
    answer_span_id: str = Field(min_length=1, max_length=60)
    checks: list[GroundedRelation] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_checks(self):
        ids = [c.predicate_id for c in self.checks]
        if len(ids) != len(set(ids)):
            raise ValueError("Grounded answer predicates must be unique")
        return self


def bind_grounded_relations(
    decision: GroundedRelations, registry: list[dict]
) -> PairedPredicateScopeDecision:
    from app.evidence_roles import bind_evidence_roles

    checks = []
    for claim in decision.checks:
        selected, _ = bind_evidence_roles(claim, registry)
        modes = {p["mode"] for p in selected}
        voices = {p["voice"] for p in selected}
        # Fact categories come only from independently projected primary evidence.
        # Scope context retains exact positions and all five semantic vetoes;
        # it cannot alone support the fact or rewrite the primary classification.
        mode = next(iter(modes)) if len(modes) == 1 else "undetermined"
        voice = next(iter(voices)) if len(voices) == 1 else "undetermined"
        checks.append({**claim.model_dump(), "source_mode": mode, "source_voice": voice})
    return PairedPredicateScopeDecision(answer_span_id=decision.answer_span_id, checks=checks)
