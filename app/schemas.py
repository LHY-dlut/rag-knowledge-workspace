from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    computed_field,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalConfig(StrictModel):
    # Seven independent retrieval switches; original toggle names are not legible.
    query_rewrite: bool = False
    multi_query: bool = False
    hyde: bool = False
    hybrid: bool = True
    rerank: bool = False
    parent_retrieval: bool = True
    metadata_filter: bool = True
    top_k: int = Field(8, ge=1, le=100)
    candidate_k: int = Field(16, ge=1, le=100)
    context_k: int = Field(8, ge=1, le=20)
    rerank_k: int = Field(4, ge=1, le=20)
    rrf_k: int = Field(60, ge=1, le=1000)
    rerank_threshold: float = Field(0.05, ge=0, le=1)
    cosine_threshold: float = Field(0.3, ge=-1, le=1)
    context_chars: int = Field(10000, ge=500, le=30000)
    chunk_strategy: Literal["recursive", "recursive_short", "parent_child"] = "parent_child"
    structure_mode: Literal["general", "laws", "qa"] = "general"
    recursive_size: int = Field(500, ge=100, le=5000)
    recursive_overlap: int = Field(50, ge=0, le=200)
    short_size: int = Field(250, ge=50, le=1000)
    short_overlap: int = Field(30, ge=0, le=200)
    parent_size: int = Field(1200, ge=200, le=5000)
    child_size: int = Field(300, ge=50, le=1000)
    child_overlap: int = Field(0, ge=0, le=200)
    grade_retry_limit: int = Field(3, ge=0, le=3)
    check_retry_limit: int = Field(2, ge=0, le=2)
    # 整链兜底：逐调用重试（protocol_retry_limit）之后仍协议失败时，
    # 对同一份未改动草稿再跑整条核验链的次数。
    check_protocol_retry_limit: int = Field(1, ge=0, le=3)

    @model_validator(mode="after")
    def validate_chunk_sizes(self):
        if self.child_size > self.parent_size or self.child_overlap >= self.child_size:
            raise ValueError("child_size <= parent_size and overlap < child_size required")
        if self.recursive_overlap >= self.recursive_size or self.short_overlap >= self.short_size:
            raise ValueError("Recursive overlap must be smaller than chunk size")
        return self


class Credentials(StrictModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class KBCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field("", max_length=1000)
    config: RetrievalConfig = Field(default_factory=RetrievalConfig)


class MetadataFilter(StrictModel):
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    file_types: list[Literal["pdf", "docx", "xlsx", "md", "txt"]] = Field(
        default_factory=list, max_length=5
    )
    tags: list[str] = Field(default_factory=list, max_length=20)


class QueryRequest(StrictModel):
    kb_id: str = Field(min_length=1, max_length=36)
    query: str = Field(min_length=1, max_length=2000)
    filters: MetadataFilter = Field(default_factory=MetadataFilter)
    strategy: Literal["dense", "hybrid", "full", "custom"] = "custom"


class ChatRequest(QueryRequest):
    conversation_id: str | None = Field(None, max_length=36)
    mode: Literal["agent", "rag"] = "agent"
    application_id: str | None = Field(None, max_length=36)
    output_type: Literal["answer", "chart", "report", "webpage"] = "answer"


class RewriteDecision(StrictModel):
    standalone_query: str = Field(min_length=1, max_length=2000)
    queries: list[str] = Field(default_factory=list, max_length=3)
    hypothetical_document: str = Field("", max_length=1000)


class Judgment(StrictModel):
    passed: bool
    reason: str = Field(max_length=1000)


class CheckEvidence(StrictModel):
    source_id: str = Field(min_length=1, max_length=30)
    span_id: str = Field(min_length=1, max_length=60)


class CheckClaim(StrictModel):
    answer_span_id: str = Field(min_length=1, max_length=60)
    verdict: Literal["supported", "unsupported", "contradicted", "condition_error", "scope_error"]
    evidence: list[CheckEvidence] = Field(default_factory=list, max_length=20)
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def supported_needs_evidence(self):
        if self.verdict == "supported" and not self.evidence:
            raise ValueError("A supported claim requires literal evidence references")
        return self


class CheckDecision(StrictModel):
    checks: list[CheckClaim] = Field(min_length=1, max_length=100)


ScopeMode = Literal[
    "asserted",
    "future",
    "planned",
    "possible",
    "permission",
    "discussion",
    "suggestion",
    "document_limitation",
    "undetermined",
]
ScopeVoice = Literal["fact", "opinion", "suggestion", "undetermined"]


class CheckAnswerProjection(StrictModel):
    """Read only the actual draft, without the question or any retrieved evidence."""

    answer_span_id: str = Field(min_length=1, max_length=60)
    reason: str = Field(min_length=1, max_length=300)
    modes: list[ScopeMode] = Field(
        min_length=1, max_length=9, json_schema_extra={"uniqueItems": True}
    )
    voices: list[ScopeVoice] = Field(
        min_length=1, max_length=4, json_schema_extra={"uniqueItems": True}
    )

    @model_validator(mode="after")
    def unique_tags(self):
        if len(self.modes) != len(set(self.modes)) or len(self.voices) != len(set(self.voices)):
            raise ValueError("Independent answer projection tags must be unique")
        return self

    def matches_source(
        self, projection: "CheckScopeProjection", no_upgrade: bool = False
    ) -> tuple[bool, bool]:
        modes = "undetermined" not in self.modes + projection.source_modes and set(
            self.modes
        ) == set(projection.source_modes)
        voices = "undetermined" not in self.voices + projection.source_voices and set(
            self.voices
        ) == set(projection.source_voices)
        if no_upgrade:
            modes = modes or _no_upgrade_preserved(self.modes, projection.source_modes)
            voices = voices or _no_upgrade_preserved(self.voices, projection.source_voices)
        return modes, voices


def _no_upgrade_preserved(answer_values: list[str], source_values: list[str]) -> bool:
    """答案不得比原文更绝对（只在显式启用时使用）。

    原判定要求答案与原文的类别集合完全相等。但设计意图是防止把预测/建议
    升级成已实现的事实；答案复述为同样谨慎的语气（甚至更保守）不应被拒。
    因此：原文非"断定/事实"时，答案出现"断定/事实"才算升级并拒绝；
    原文本身是断定/事实时，答案更保守视为通过。待定仍按原严格规则处理。
    """
    if "undetermined" in answer_values or "undetermined" in source_values:
        return False
    if set(answer_values) == set(source_values):
        return True
    if "asserted" in answer_values and "asserted" not in source_values:
        return False
    return not ("fact" in answer_values and "fact" not in source_values)


class CheckScopeProjection(StrictModel):
    answer_modes: list[ScopeMode] = Field(
        min_length=1,
        max_length=9,
        json_schema_extra={"uniqueItems": True},
        description="仅读当前实际答案的内容性谓词，使用系统唯一分类表；不从来源替答案补语气，不漏实际谓词，输出去重集合。",
    )
    answer_voices: list[ScopeVoice] = Field(
        min_length=1,
        max_length=4,
        json_schema_extra={"uniqueItems": True},
        description="按当前实际答案的观点/建议/事实性质，使用系统同一分类表，输出去重集合。",
    )
    source_modes: list[ScopeMode] = Field(
        min_length=1,
        max_length=9,
        json_schema_extra={"uniqueItems": True},
        description="仅读实际来源中对应当前答案谓词的原命题，使用同一分类表；其他未被声称的并列事实不混入，输出去重集合。",
    )
    source_voices: list[ScopeVoice] = Field(
        min_length=1,
        max_length=4,
        json_schema_extra={"uniqueItems": True},
        description="按当前对应原命题的观点/建议/事实性质，使用系统同一分类表，输出去重集合。",
    )

    @model_validator(mode="after")
    def unique_projection_tags(self):
        for name in ("answer_modes", "answer_voices", "source_modes", "source_voices"):
            tags = getattr(self, name)
            if len(tags) != len(set(tags)):
                raise ValueError("Semantic projection tags must be unique")
        return self

    def preserves_modes(self) -> bool:
        return "undetermined" not in self.answer_modes + self.source_modes and set(
            self.answer_modes
        ) == set(self.source_modes)

    def preserves_voices(self) -> bool:
        return "undetermined" not in self.answer_voices + self.source_voices and set(
            self.answer_voices
        ) == set(self.source_voices)


class CheckScopeClaim(StrictModel):
    answer_span_id: str = Field(min_length=1, max_length=60)
    source_ids: list[str] = Field(min_length=1, max_length=20)
    subject_predicate_preserved: StrictBool
    modality_preserved: StrictBool
    temporal_scope_preserved: StrictBool
    conditions_preserved: StrictBool
    attribution_preserved: StrictBool
    reason: str = Field(min_length=1, max_length=300)

    projection: CheckScopeProjection | None = None


class CheckScopeDecision(StrictModel):
    checks: list[CheckScopeClaim] = Field(min_length=1, max_length=100)


class CheckScopeAssessmentClaim(StrictModel):
    answer_span_id: str = Field(min_length=1, max_length=60)
    source_ids: list[str] = Field(min_length=1, max_length=20)
    reason: str = Field(
        min_length=1,
        max_length=300,
        description="用不超过80个中文字符概述实际差异或支持关系；不得逐项长篇展开。",
    )
    projection: CheckScopeProjection
    subject_predicate_preserved: Literal["preserved", "changed", "undetermined"] = Field(
        description="按实际答案分句的主体、谓词、对象与实际引用配对；问题分类词不是答案断言。"
    )
    modality_preserved: Literal["preserved", "changed", "undetermined"] = Field(
        description="按答案实际谓词比较语气；讨论、计划或建议的宾语不是独立实施断言，除非答案实际改称已实现。"
    )
    temporal_scope_preserved: Literal["preserved", "changed", "undetermined"] = Field(
        description="比较当前断言的真实时间框架；同句历史导语限定后续叙述，不臆造现在。发布日期不代替事件、生效或失效时间，不能越出原文时间范围。"
    )
    conditions_preserved: Literal["preserved", "changed", "undetermined"] = Field(
        description="只核对当前命题由原文建立的必要限定，不把标题或省略的其他并列事实增设为前提；真实日期、地域、试点及程序限制不能扩大。"
    )
    attribution_preserved: Literal["preserved", "changed", "undetermined"] = Field(
        description="明示观点、预测或建议不能被客观化。普通有来源规则或事实不用复述无语义限定标题；实际观点性与提出者归属仍须保留。"
    )


class CheckScopeAssessment(StrictModel):
    checks: list[CheckScopeAssessmentClaim] = Field(min_length=1, max_length=100)

    def to_decision(self) -> CheckScopeDecision:
        checks = []
        for claim in self.checks:
            relations = {
                name: getattr(claim, name) == "preserved"
                for name in (
                    "subject_predicate_preserved",
                    "modality_preserved",
                    "temporal_scope_preserved",
                    "conditions_preserved",
                    "attribution_preserved",
                )
            }
            modes = claim.projection.preserves_modes()
            voices = claim.projection.preserves_voices()
            relations["modality_preserved"] = relations["modality_preserved"] and modes
            relations["attribution_preserved"] = relations["attribution_preserved"] and voices
            reason = claim.reason
            if not modes or not voices:
                reason = ("实际答案与原文的语气/归属投影不一致或待定；模型理由：" + reason)[:300]
            checks.append({**claim.model_dump(), **relations, "reason": reason})
        return CheckScopeDecision(checks=checks)


class AnswerPredicate(StrictModel):
    predicate_id: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z0-9:_-]+$")
    quote: str = Field(min_length=1, max_length=400)
    mode: ScopeMode
    voice: ScopeVoice
    reason: str = Field(min_length=1, max_length=300)


class AnswerPredicates(StrictModel):
    answer_span_id: str = Field(min_length=1, max_length=60)
    predicates: list[AnswerPredicate] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_predicates(self):
        ids = [p.predicate_id for p in self.predicates]
        if len(ids) != len(set(ids)):
            raise ValueError("Answer predicate IDs must be unique")
        return self


PredicateFragmentId = Annotated[
    str, Field(min_length=1, max_length=90, pattern=r"^[A-Za-z0-9:_-]+$")
]


class AnswerRangePredicate(StrictModel):
    predicate_id: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z0-9:_-]+$")
    fragment_ids: list[PredicateFragmentId] = Field(min_length=1, max_length=400)
    mode: ScopeMode
    voice: ScopeVoice
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def unique_fragments(self):
        if len(self.fragment_ids) != len(set(self.fragment_ids)):
            raise ValueError("Answer fragment IDs must be unique")
        return self


class AnswerRangePredicates(StrictModel):
    answer_span_id: str = Field(min_length=1, max_length=60)
    predicates: list[AnswerRangePredicate] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_predicates(self):
        ids = [p.predicate_id for p in self.predicates]
        if len(ids) != len(set(ids)):
            raise ValueError("Answer range predicate IDs must be unique")
        return self


class PredicateScopeAssessment(StrictModel):
    predicate_id: str = Field(min_length=1, max_length=60)
    evidence: list[CheckEvidence] = Field(min_length=1, max_length=20)
    source_mode: ScopeMode
    source_voice: ScopeVoice
    subject_predicate_preserved: Literal["preserved", "changed", "undetermined"]
    modality_preserved: Literal["preserved", "changed", "undetermined"]
    temporal_scope_preserved: Literal["preserved", "changed", "undetermined"]
    conditions_preserved: Literal["preserved", "changed", "undetermined"]
    attribution_preserved: Literal["preserved", "changed", "undetermined"]
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def unique_evidence(self):
        refs = [(e.source_id, e.span_id) for e in self.evidence]
        if len(refs) != len(set(refs)):
            raise ValueError("Predicate evidence references must be unique")
        return self


class PredicateScopeDecision(StrictModel):
    answer_span_id: str = Field(min_length=1, max_length=60)
    checks: list[PredicateScopeAssessment] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_checks(self):
        ids = [c.predicate_id for c in self.checks]
        if len(ids) != len(set(ids)):
            raise ValueError("Scope predicate IDs must be unique")
        return self


class BoundCheckJudgment(Judgment):
    checks: list[CheckClaim]
    evidence_protocol: dict[str, Any]


class GradeEvidence(StrictModel):
    source_id: str = Field(min_length=1, max_length=30)
    quote: str = Field("", max_length=5000)
    span_id: str = Field("", max_length=60)
    subject: str = Field(min_length=1, max_length=300)
    attribute: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def require_evidence_reference(self):
        if not self.quote and not self.span_id.strip():
            raise ValueError("Evidence requires a raw quote or a server-issued span ID")
        return self


class GradeDecision(Judgment):
    passed: StrictBool
    support: Literal["supported_answer", "supported_limitation", "insufficient"]
    evidence: list[GradeEvidence] = Field(default_factory=list, max_length=20)
    answer_scope: str = Field("", max_length=2000)

    @model_validator(mode="after")
    def require_supporting_evidence(self):
        supported = self.support != "insufficient"
        if self.passed != supported:
            raise ValueError("Grade passed must agree with support classification")
        if supported and (not self.evidence or not self.answer_scope.strip()):
            raise ValueError("A supported grade requires evidence and an explicit answer scope")
        return self


class GradeSpanEvidence(StrictModel):
    source_id: str = Field(min_length=1, max_length=30)
    span_id: str = Field(min_length=1, max_length=60)
    subject: str = Field(min_length=1, max_length=300)
    attribute: str = Field(min_length=1, max_length=300)


class GradeSpanDecision(GradeDecision):
    # Transport-only schema: the server binds selected IDs to the immutable raw
    # source. A model-supplied quote must not compete with that exact binding.
    evidence: list[GradeSpanEvidence] = Field(default_factory=list, max_length=20)


class GradeAssessment(StrictModel):
    """Transport classification: the domain passed boolean is computed once."""

    support: Literal["supported_answer", "supported_limitation", "insufficient"]
    evidence: list[GradeEvidence] = Field(default_factory=list, max_length=20)
    answer_scope: str = Field("", max_length=2000)
    reason: str = Field(max_length=1000)

    @model_validator(mode="after")
    def validate_domain_requirements(self):
        GradeDecision.model_validate(
            {**self.model_dump(), "passed": self.support != "insufficient"}
        )
        return self


class GradeSpanAssessment(GradeAssessment):
    evidence: list[GradeSpanEvidence] = Field(default_factory=list, max_length=20)


class GradeAnswerBindingDecision(StrictModel):
    subject_attribute_pairs_supported: StrictBool
    question_requirements_covered: StrictBool
    conditions_preserved: StrictBool
    scope_supported: StrictBool
    reason: str = Field(max_length=1000)

    @computed_field
    @property
    def passed(self) -> bool:
        return all(
            (
                self.subject_attribute_pairs_supported,
                self.question_requirements_covered,
                self.conditions_preserved,
                self.scope_supported,
            )
        )


class GradeBindingDecision(StrictModel):
    subject_matches: StrictBool
    attribute_matches: StrictBool
    explicit_limitation: StrictBool
    scope_supported: StrictBool
    reason: str = Field(max_length=1000)

    @computed_field
    @property
    def passed(self) -> bool:
        return all(
            (
                self.subject_matches,
                self.attribute_matches,
                self.explicit_limitation,
                self.scope_supported,
            )
        )


class ChildRecord(BaseModel):
    id: str
    owner_id: str
    kb_id: str
    doc_id: str
    parent_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_fingerprint: str
    embedding: list[float] = Field(default_factory=list)

    @field_validator("embedding", mode="before")
    @classmethod
    def strict_embedding_values(cls, value):
        if not isinstance(value, list) or any(type(x) not in (int, float) for x in value):
            raise ValueError("Embedding values must be numbers, not booleans or strings")
        return value


class Candidate(ChildRecord):
    cosine_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0
    rerank_score: float | None = None


class Citation(BaseModel):
    source_id: str
    document_id: str
    parent_id: str
    filename: str
    location: str
    content: str
    child_ids: list[str]
    edited: bool = False
    document_revision: int = 0


class RetrievalResult(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class ChunkEdit(StrictModel):
    content: str = Field(min_length=1, max_length=5000)


class EvaluationHistoryMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=5000)


class EvaluationCase(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    reference_answer: str = Field(min_length=1, max_length=5000)
    reference_facts: list[str] = Field(default_factory=list, max_length=20)
    answerable: bool = True
    relevant_document_ids: list[str] = Field(default_factory=list, max_length=100)
    provenance: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    relevant_child_ids: list[str] | None = Field(None, max_length=1000)
    annotation_method: Literal["human"] | None = None
    history: list[EvaluationHistoryMessage] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def require_human_labels(self):
        if self.answerable and not self.reference_facts:
            raise ValueError("Answerable evaluation cases require reference facts")
        if self.relevant_child_ids is not None and self.annotation_method != "human":
            raise ValueError("Relevant child labels require independent human annotation")
        return self


class EvaluationRequest(StrictModel):
    kb_id: str
    cases: list[EvaluationCase] = Field(min_length=1, max_length=50)
    # rag：单轮检索+生成核验，用于隔离检索策略差异；
    # agent：允许受限循环与工具路由（默认，保持既有行为）。
    mode: Literal["rag", "agent"] = "agent"


class PromptEdit(StrictModel):
    content: str = Field(min_length=1, max_length=8000)


StrategyName = Literal["dense", "hybrid", "full", "custom"]


class ApplicationInput(StrictModel):
    kb_id: str = Field(min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field("", max_length=1000)
    strategy: StrategyName = "hybrid"
    mode: Literal["rag", "agent"] = "rag"
    welcome: str = Field("你好，请提出与资料库有关的问题。", max_length=2000)
    fallback: str = Field("资料库中暂无足够依据回答这个问题。", min_length=1, max_length=2000)
    prompt: str = Field("用中文回答，先给结论，再列依据。", max_length=8000)
    enabled: bool = True


class FeedbackInput(StrictModel):
    helpful: bool
    comment: str = Field("", max_length=2000)


class DatasetInput(StrictModel):
    kb_id: str = Field(min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=120)
    cases: list[EvaluationCase] = Field(min_length=1, max_length=50)


class DatasetGenerate(StrictModel):
    kb_id: str = Field(min_length=1, max_length=36)
    name: str = Field("文档自动生成测评集", min_length=1, max_length=120)
    count: int = Field(5, ge=1, le=10)
    document_ids: list[str] = Field(default_factory=list, max_length=100)


class FeedbackDatasetInput(StrictModel):
    run_id: str = Field(min_length=1, max_length=36)
    name: str = Field("反馈人工订正测评集", min_length=1, max_length=120)
    reference_answer: str = Field(min_length=1, max_length=5000)
    reference_facts: list[str] = Field(min_length=1, max_length=20)


class GeneratedCase(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    reference_answer: str = Field(min_length=1, max_length=5000)
    reference_facts: list[str] = Field(min_length=1, max_length=20)
    source_ids: list[str] = Field(min_length=1, max_length=10)


class GeneratedCases(StrictModel):
    cases: list[GeneratedCase] = Field(min_length=1, max_length=10)


class ComparisonRequest(EvaluationRequest):
    cases: list[EvaluationCase] = Field(min_length=1, max_length=10)
