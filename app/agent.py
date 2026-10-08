import json
import time
from typing import Any, Literal, TypedDict

from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.check_protocol import answer_citation_coverage, answer_year_binding, check_retry_feedback
from app.check_scope import validate_check_scope
from app.evidence_protocol import bind_decision, evidence_spans, prepare_citations
from app.models import AgentStep, KnowledgeBase, PromptTemplate, ToolDefinition
from app.providers import CheckProtocolError, GradeProtocolError, ModelProvider
from app.retrieval import AdvancedRetrieverPipeline
from app.schemas import (
    GradeAnswerBindingDecision,
    GradeBindingDecision,
    GradeDecision,
    Judgment,
    MetadataFilter,
    RetrievalConfig,
    RetrievalResult,
    RewriteDecision,
)
from app.strategies import expanded_queries
from app.tools import (
    CALCULATOR_TOOL,
    CLOCK_TOOL,
    SEARCH_TOOL,
    CalculatorArguments,
    ClockArguments,
    SearchArguments,
    calculate,
    current_time,
)

FALLBACK_ANSWER = (
    "资料库中暂无足够依据回答这个问题。请补充相关文档，或把问题限定到已上传资料的范围。"
)


class GraphState(TypedDict, total=False):
    run_id: str
    owner_id: str
    kb: KnowledgeBase
    query: str
    standalone_query: str
    queries: list[str]
    hypothetical_document: str
    history: list[dict[str, str]]
    filters: MetadataFilter
    mode: Literal["agent", "rag"]
    route: Literal["knowledge", "calculator", "current_time"]
    tool_arguments: dict[str, Any]
    grade_retries: int
    check_retries: int
    check_protocol_retries: int
    generation_attempt: int
    retrieval: RetrievalResult
    grade: GradeDecision
    grade_binding: GradeBindingDecision | GradeAnswerBindingDecision
    check: Judgment
    draft: str
    answer: str
    feedback: str
    rejected: bool
    step_ordinal: int
    grade_retry_allowed: bool
    check_retry_allowed: bool
    check_protocol_retry_allowed: bool
    application_prompt: str
    fallback_answer: str
    grade_protocol: dict
    citation_protocol: dict
    protocol_feedback: str | dict


def citation_check(answer: str, sources: list[dict]) -> Judgment:
    _, audit = prepare_citations(answer, sources)
    passed = audit["passed"]
    return Judgment(
        passed=passed,
        reason="引用 ID 合法" if passed else "引用协议失败:" + ",".join(audit["failure_types"]),
    )


def bind_grade_evidence(decision: GradeDecision, sources: list[dict]) -> GradeDecision:
    """Validate attribution; semantic subject/property matching belongs to the model."""
    return bind_decision(decision, sources)[0]


async def validate_grade_support(
    provider, decision: GradeDecision, sources: list[dict], query: str
):
    decision = bind_grade_evidence(decision, sources)
    if not decision.passed:
        return decision, None
    if decision.support == "supported_answer":
        # Real providers advertise the additional semantic gate. Demo and
        # interface-only stubs retain their explicitly unverified simulation.
        if not getattr(provider, "supports_answer_binding", False):
            return decision, None
        schema, task = GradeAnswerBindingDecision, "grade_answer_binding"
    else:
        schema, task = GradeBindingDecision, "grade_binding"
    binding = await provider.structured(
        schema,
        task,
        {
            "query": query,
            "sources": sources,
            "evidence": [item.model_dump() for item in decision.evidence],
            "answer_scope": decision.answer_scope,
        },
    )
    if not binding.passed:
        decision = GradeDecision(passed=False, support="insufficient", reason=binding.reason)
    return decision, binding


class RAGAgent:
    def __init__(self, provider: ModelProvider, retriever: AdvancedRetrieverPipeline):
        self.provider, self.retriever = provider, retriever
        graph = StateGraph(GraphState)
        for name in (
            "route",
            "rewrite",
            "retrieve",
            "grade",
            "generate",
            "check",
            "fallback",
            "tool",
        ):
            graph.add_node(name, self._wrap(name, getattr(self, name)))
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route", lambda s: "rewrite" if s["route"] == "knowledge" else "tool"
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", self._after_grade)
        graph.add_edge("generate", "check")
        graph.add_conditional_edges("check", self._after_check)
        graph.add_edge("fallback", END)
        graph.add_edge("tool", END)
        self.graph = graph.compile()

    def _wrap(self, name, function):
        async def node(state: GraphState):
            writer = get_stream_writer()
            ordinal = state.get("step_ordinal", 0) + 1
            summary = {
                "query": state.get("standalone_query", state["query"]),
                "grade_retries": state.get("grade_retries", 0),
                "check_retries": state.get("check_retries", 0),
                "check_protocol_retries": state.get("check_protocol_retries", 0),
            }
            step = await AgentStep.create(
                run_id=state["run_id"],
                node=name,
                ordinal=ordinal,
                status="running",
                input_summary=summary,
            )
            writer(
                {
                    "event": "agent_step",
                    "data": {
                        "node": name,
                        "status": "running",
                        "input": summary,
                        "ordinal": ordinal,
                        "step_id": step.id,
                    },
                }
            )
            started = time.perf_counter()
            try:
                update = await function(state)
                output = self._summary(update)
                elapsed = round((time.perf_counter() - started) * 1000)
                await AgentStep.filter(id=step.id).update(
                    status="completed", output_summary=output, elapsed_ms=elapsed
                )
                writer(
                    {
                        "event": "agent_step",
                        "data": {
                            "node": name,
                            "ordinal": ordinal,
                            "step_id": step.id,
                            "status": "completed",
                            "output": output,
                            "elapsed_ms": elapsed,
                        },
                    }
                )
                return {**update, "step_ordinal": ordinal}
            except BaseException as exc:
                await AgentStep.filter(id=step.id).update(
                    status="failed",
                    elapsed_ms=round((time.perf_counter() - started) * 1000),
                    output_summary={"error_type": type(exc).__name__},
                )
                raise

        return node

    @staticmethod
    def _summary(update: dict) -> dict:
        output: dict[str, Any] = {}
        for key, value in update.items():
            if isinstance(value, RetrievalResult):
                output[key] = {
                    "candidate_count": len(value.candidates),
                    "source_count": len(value.sources),
                    **value.diagnostics,
                }
            elif isinstance(value, (Judgment, GradeBindingDecision, GradeAnswerBindingDecision)):
                output[key] = value.model_dump()
            elif key in {"draft", "answer"}:
                output[f"{key}_chars"] = len(value)
            elif isinstance(value, (str, bool, int, list, dict)):
                output[key] = value
        return output

    async def route(self, state: GraphState):
        if state["mode"] == "rag":
            return {"route": "knowledge"}
        disabled = set(
            await ToolDefinition.filter(owner_id=state["owner_id"], enabled=False).values_list(
                "name", flat=True
            )
        )
        tools = [SEARCH_TOOL] + [
            t for t in (CALCULATOR_TOOL, CLOCK_TOOL) if t["function"]["name"] not in disabled
        ]
        decision = await self.provider.route(state["query"], state.get("history", []), tools)
        calls = decision.get("tool_calls", [])
        if len(calls) != 1:
            # Conservative fact router: direct model content never bypasses grounding.
            return {"route": "knowledge"}
        fn = calls[0]["function"]
        arguments = json.loads(fn["arguments"])
        if fn["name"] == "calculator" and "calculator" not in disabled:
            arguments = CalculatorArguments.model_validate(arguments).model_dump()
            return {"route": "calculator", "tool_arguments": arguments}
        if fn["name"] == "current_time" and "current_time" not in disabled:
            arguments = ClockArguments.model_validate(arguments).model_dump()
            return {"route": "current_time", "tool_arguments": arguments}
        if fn["name"] == "search_knowledge":
            arguments = SearchArguments.model_validate(arguments)
            config = RetrievalConfig.model_validate(state["kb"].config)
            query = arguments.query if config.query_rewrite else state["query"]
            return {"route": "knowledge", "standalone_query": query}
        raise ValueError("模型选择了未注册的工具")

    async def tool(self, state: GraphState):
        if state["route"] == "current_time":
            return {
                "answer": "当前系统时间：" + current_time(**state["tool_arguments"]),
                "rejected": False,
            }
        return {
            "answer": "计算结果：" + calculate(state["tool_arguments"]["expression"]),
            "rejected": False,
        }

    async def rewrite(self, state: GraphState):
        config = RetrievalConfig.model_validate(state["kb"].config)
        query = (
            state.get("standalone_query", state["query"])
            if config.query_rewrite
            else state["query"]
        )
        if not config.query_rewrite and not config.multi_query and not config.hyde:
            return {"standalone_query": query, "queries": [query], "hypothetical_document": ""}
        decision = await self.provider.structured(
            RewriteDecision,
            "rewrite",
            {
                "query": query,
                "history": state.get("history", []),
                "feedback": state.get("feedback", ""),
                "resolve_coreference": config.query_rewrite,
                "multi_query": config.multi_query,
                "generate_hyde": config.hyde,
            },
        )
        standalone = decision.standalone_query if config.query_rewrite else query
        queries = expanded_queries(state["query"], standalone, decision.queries, config.multi_query)
        return {
            "standalone_query": standalone,
            "queries": queries,
            "hypothetical_document": decision.hypothetical_document if config.hyde else "",
        }

    async def retrieve(self, state: GraphState):
        async def search_knowledge(query: str) -> dict:
            result = await self.retriever.retrieve(
                owner_id=state["owner_id"],
                kb=state["kb"],
                original_query=query,
                queries=state["queries"],
                hypothetical_document=state.get("hypothetical_document", ""),
                filters=state["filters"],
            )
            return result.model_dump()

        tool = StructuredTool.from_function(
            coroutine=search_knowledge,
            name="search_knowledge",
            description="检索当前授权知识库，返回候选子块与父块证据",
            args_schema=SearchArguments,
        )
        result = await tool.ainvoke({"query": state.get("standalone_query", state["query"])})
        return {"retrieval": RetrievalResult.model_validate(result)}

    async def grade(self, state: GraphState):
        sources = [s.model_dump() for s in state["retrieval"].sources]
        invalid_response = False
        if not any(s["content"].strip() for s in sources):
            judgment = GradeDecision(
                passed=False, support="insufficient", reason="没有检索到可用证据"
            )
        else:
            try:
                judgment = await self.provider.structured(
                    GradeDecision,
                    "grade",
                    {
                        "query": state.get("standalone_query", state["query"]),
                        "sources": [{**s, "evidence_spans": evidence_spans(s)} for s in sources],
                        **(
                            {"protocol_feedback": state["protocol_feedback"]}
                            if state.get("protocol_feedback")
                            else {}
                        ),
                    },
                )
            except GradeProtocolError:
                invalid_response = True
                judgment = GradeDecision(
                    passed=False,
                    support="insufficient",
                    reason="相关性判断结构无效，未获得有效证据判断；重新核对主体、属性及原文。",
                )
        judgment, protocol = bind_decision(judgment, sources)
        if invalid_response:
            protocol["failure_types"] = ["grade_response_schema_invalid"]
            protocol["grade_response"] = {
                "passed": False,
                "failure_type": "grade_response_schema_invalid",
                "error_type": "GradeProtocolError",
            }
        judgment, binding = await validate_grade_support(
            self.provider, judgment, sources, state.get("standalone_query", state["query"])
        )
        update = {
            "grade": judgment,
            "feedback": "" if judgment.passed else judgment.reason,
            "grade_protocol": protocol,
            "protocol_feedback": protocol.get("retry_feedback", judgment.reason)
            if not protocol["passed"] and protocol.get("evidence")
            else "",
        }
        if invalid_response:
            update["protocol_feedback"] = {
                "failure_type": "grade_response_schema_invalid",
                "instruction": (
                    "上轮grade结构无效，不得沿用其结论。按本轮原文重新判断。"
                    "只返回support分类；supported_answer或supported_limitation须提供真实证据和范围；"
                    "insufficient表示缺依据，passed由后端计算。不能仅为满足结构而声称有事实支持。"
                ),
            }
        if binding is not None:
            update["grade_binding"] = binding
        limit = RetrievalConfig.model_validate(state["kb"].config).grade_retry_limit
        allowed = (
            not judgment.passed
            and state["mode"] == "agent"
            and state.get("grade_retries", 0) < limit
        )
        update["grade_retry_allowed"] = allowed
        if allowed:
            update["grade_retries"] = state.get("grade_retries", 0) + 1
        return update

    def _after_grade(self, state: GraphState):
        if state["grade"].passed:
            return "generate"
        return "rewrite" if state["grade_retry_allowed"] else "fallback"

    async def generate(self, state: GraphState):
        writer = get_stream_writer()
        attempt = state.get("generation_attempt", 0) + 1
        writer({"event": "draft_start", "data": {"attempt": attempt, "verified": False}})
        template = await PromptTemplate.get_or_none(owner_id=state["owner_id"], name="generation")
        parts = []
        characters = 0
        assessment = state["grade"].model_dump()
        if state["grade"].support == "supported_answer":
            # For a direct answer, gate prose is not new evidence or a requested
            # answer. Limitation answers still receive their validated scope.
            assessment.pop("reason", None)
            assessment.pop("answer_scope", None)
        async for token in self.provider.stream_answer(
            state.get("standalone_query", state["query"]),
            [s.model_dump() for s in state["retrieval"].sources],
            json.dumps(
                {
                    "evidence_assessment": assessment,
                    "revision_feedback": state.get("feedback", ""),
                },
                ensure_ascii=False,
            ),
            state.get("application_prompt", template.content if template else ""),
        ):
            parts.append(token)
            characters += len(token)
            if characters > 20000:
                raise ValueError("生成答案超过字符上限")
            writer(
                {
                    "event": "draft_token",
                    "data": {"text": token, "attempt": attempt, "verified": False},
                }
            )
        return {"draft": "".join(parts), "generation_attempt": attempt}

    async def check(self, state: GraphState):
        sources = [s.model_dump() for s in state["retrieval"].sources]
        rendered, protocol = prepare_citations(state["draft"], sources)
        judgment = citation_check(state["draft"], sources)
        if judgment.passed:
            coverage = answer_citation_coverage(rendered)
            protocol["answer_citation_coverage"] = coverage
            if not coverage["passed"]:
                protocol["passed"] = False
                protocol["failure_types"] = coverage["failure_types"]
                excerpts = [s["quote"] for s in coverage["uncited_answer_spans"]]
                judgment = Judgment(
                    passed=False,
                    reason=(
                        "存在未被现有引用覆盖的答案片段："
                        + json.dumps(excerpts, ensure_ascii=False)[:700]
                        + "。引用不能跨空行；删除非必要引言或为陈述提供真实相邻引用后重新生成。"
                        "若引用后还留有动作或结论，应重新生成完整陈述并把对应真实引用放在该陈述末尾，"
                        "不要只在数字或名词后提前标记引用。"
                        "不得补造引用；引用完整后仍需独立事实核验。"
                    ),
                )
        if judgment.passed:
            binding = answer_year_binding(rendered, sources)
            protocol["answer_year_binding"] = binding
            if not binding["passed"]:
                protocol["passed"] = False
                protocol["failure_types"] = binding["failure_types"]
                years = json.dumps(
                    [
                        {"year": v["year"], "cited": v["cited_source_ids"]}
                        for v in binding["unsupported_years"]
                    ],
                    ensure_ascii=False,
                )
                judgment = Judgment(
                    passed=False,
                    reason=(
                        "答案断言的年份在被引用的原文中找不到："
                        + years[:700]
                        + "。原文未标明年份时不得把问题里的年份回写进答案。"
                        "请删去无原文依据的年份、改用原文真正写出的时间表述，"
                        "或为含年份的陈述提供确实包含该年份的引用来源；"
                        "若证据本身无法回答该年份的问题，应当拒答而不是补一个年份。"
                    ),
                )
        if judgment.passed:
            try:
                judgment = await self.provider.structured(
                    Judgment,
                    "check",
                    {
                        "query": state.get("standalone_query", state["query"]),
                        "answer": rendered,
                        "sources": sources,
                        "evidence_assessment": state["grade"].model_dump(),
                    },
                )
                judgment, scope_review = await validate_check_scope(
                    self.provider,
                    judgment,
                    rendered,
                    sources,
                    state.get("standalone_query", state["query"]),
                )
                if scope_review is not None:
                    protocol["scope_review"] = scope_review
            except CheckProtocolError:
                # An invalid check is neither evidence nor permission to publish.
                # It is also not evidence against the draft: a protocol failure
                # says nothing about support, so the unchanged draft is verified
                # again instead of being replaced or rejected. Network/budget
                # failures remain ProviderError and keep their terminal handling.
                limit = RetrievalConfig.model_validate(
                    state["kb"].config
                ).check_protocol_retry_limit
                attempted = state.get("check_protocol_retries", 0)
                allowed = attempted < limit
                protocol["check_response"] = {
                    "passed": False,
                    "failure_type": "check_response_schema_invalid",
                    "error_type": "CheckProtocolError",
                    "attempts": attempted + 1,
                    "protocol_retry_allowed": allowed,
                }
                if allowed:
                    return {
                        "citation_protocol": protocol,
                        "check_protocol_retries": attempted + 1,
                        "check_protocol_retry_allowed": True,
                    }
                judgment = Judgment(
                    passed=False,
                    reason="核验结构无效，草稿不可发布；重新生成有明确出处的事实陈述后再核验。",
                )
        if judgment.passed:
            return {
                "check": judgment,
                "answer": rendered,
                "draft": rendered,
                "citation_protocol": protocol,
                "check_protocol_retry_allowed": False,
                "rejected": False,
            }
        if protocol["passed"]:
            protocol["semantic_support"] = {
                "passed": False,
                "failure_type": "check_response_schema_invalid"
                if "check_response" in protocol
                else "citation_fact_support_failed",
                "reason": judgment.reason,
            }
        get_stream_writer()({"event": "draft_reset", "data": {"reason": judgment.reason}})
        limit = RetrievalConfig.model_validate(state["kb"].config).check_retry_limit
        allowed = state["mode"] == "agent" and state.get("check_retries", 0) < limit
        return {
            "check": judgment,
            "citation_protocol": protocol,
            "check_retries": state.get("check_retries", 0) + int(allowed),
            "check_retry_allowed": allowed,
            "check_protocol_retry_allowed": False,
            "feedback": check_retry_feedback(judgment),
            "draft": "",
        }

    def _after_check(self, state: GraphState):
        check = state.get("check")
        if check is not None and check.passed:
            return END
        if state.get("check_protocol_retry_allowed"):
            # Re-verify the same untouched draft; never regenerate for a
            # protocol failure, which is not a verdict about support.
            return "check"
        return "generate" if state["check_retry_allowed"] else "fallback"

    async def fallback(self, state: GraphState):
        get_stream_writer()({"event": "draft_reset", "data": {"reason": "证据校验未通过"}})
        return {
            "answer": state.get("fallback_answer", FALLBACK_ANSWER),
            "draft": "",
            "rejected": True,
        }
