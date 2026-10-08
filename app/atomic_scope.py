"""Pair scope checks with literal predicates, never sentence-wide category sets."""

from types import SimpleNamespace

from app.answer_parts import FragmentCoverageError
from app.check_protocol import citation_scopes
from app.evidence_protocol import evidence_spans, match_evidence
from app.providers import CheckProtocolError
from app.schemas import AnswerPredicates, AnswerRangePredicates, PredicateScopeDecision
from app.scope_feedback import atomic_scope_feedback

PREDICATES = (
    "subject_predicate_preserved",
    "modality_preserved",
    "temporal_scope_preserved",
    "conditions_preserved",
    "attribution_preserved",
)


async def project_parts(provider, schema, task: str, payload: dict, bind):
    """逐谓词片段投影；模型漏选实质片段时按缺失清单定向重问。

    只重述被漏掉的真实片段并要求重新归类；不替模型补全、不改写类别、
    不放宽覆盖要求。重问次数用尽后仍按原样抛出，由上层决定是否拒答。
    """
    limit = getattr(getattr(provider, "settings", None), "part_coverage_retry_limit", 0)
    correction = ""
    for attempt in range(limit + 1):
        # 首次保持原样调用；只有确实查到漏选片段时才带上纠正指令重问。
        decision = (
            await provider.structured(schema, task, payload, correction=correction)
            if correction
            else await provider.structured(schema, task, payload)
        )
        try:
            return decision, bind(decision)
        except FragmentCoverageError as exc:
            if attempt >= limit:
                raise
            correction = exc.correction
    raise AssertionError("不可达：循环内必然返回或抛出")


def bind_answer_predicates(decision: AnswerPredicates, focus: dict, answer: str) -> list[dict]:
    if decision.answer_span_id != focus["span_id"]:
        raise CheckProtocolError("逐谓词投影返回非当前答案片段")
    marker_positions, _ = citation_scopes(answer)
    required = {
        i
        for i in range(focus["source_start"], focus["source_end"])
        if answer[i].isalnum() and i not in marker_positions
    }
    covered, rows = set(), []
    for predicate in decision.predicates:
        first = focus["quote"].find(predicate.quote)
        if first >= 0 and focus["quote"].find(predicate.quote, first + 1) >= 0:
            raise CheckProtocolError("逐谓词投影文本位置有歧义，不能自动选择第一次出现")
        bound = match_evidence(
            SimpleNamespace(span_id="", quote=predicate.quote),
            {"source_id": "A", "content": focus["quote"]},
        )
        if not bound["passed"]:
            raise CheckProtocolError("逐谓词投影文本不在当前真实答案中或位置有歧义")
        start = focus["source_start"] + bound["source_start"]
        end = focus["source_start"] + bound["source_end"]
        positions = set(range(start, end)) & required
        if not positions:
            raise CheckProtocolError("逐谓词投影只有引用或标点，不能核验事实")
        covered.update(positions)
        # match_evidence reads a bounded focus substring. Persist all positions
        # in the coordinate system of the unchanged full answer.
        offset = focus["source_start"]
        bound = {
            **bound,
            "source_start": start,
            "source_end": end,
            "normalization_map": [
                (left + offset, right + offset) for left, right in bound["normalization_map"]
            ],
            "removed_whitespace_spans": [
                (left + offset, right + offset) for left, right in bound["removed_whitespace_spans"]
            ],
            "offset_unit": "unchanged full answer Unicode codepoint",
            "focus_answer_span_id": focus["span_id"],
        }
        rows.append(
            {
                **predicate.model_dump(),
                "source_start": start,
                "source_end": end,
                "original_quote": answer[start:end],
                "literal_binding": bound,
            }
        )
    if covered != required:
        raise CheckProtocolError("逐谓词投影遗漏答案内容或范围限定")
    return rows


async def review_atomic_scope(
    provider,
    answer: str,
    focus: dict,
    context: list[dict],
    sources: list[dict],
    actual_source_ids: set[str],
    query: str,
    source_evidence: list[dict] | None = None,
    source_projection_cache: dict | None = None,
):
    payload = {
        "answer": answer,
        "focus_answer_span_id": focus["span_id"],
        "answer_spans": [focus],
        "answer_context_spans": context,
    }
    if getattr(provider, "supports_check_context_without_foreign_ids", False):
        payload["answer_context_spans"] = [
            {key: value for key, value in span.items() if key != "span_id"} for span in context
        ]
    use_parts = getattr(provider, "supports_check_predicate_parts", False)
    if use_parts:
        from app.answer_parts import bind_answer_parts
        from app.answer_ranges import answer_fragments

        payload["answer_fragments"] = answer_fragments(focus, answer)
        if (
            getattr(provider, "supports_check_explicit_projection_focus", False)
            and getattr(provider, "supports_check_source_predicate_parts", False)
            and getattr(provider, "supports_check_grounded_relations", False)
        ):
            payload = {
                "focus_answer_span": focus,
                **payload,
            }
        independent, atoms = await project_parts(
            provider,
            AnswerRangePredicates,
            "check_answer_predicate_parts",
            payload,
            lambda decision: bind_answer_parts(decision, focus, answer),
        )
    elif getattr(provider, "supports_check_predicate_ranges", False):
        from app.answer_ranges import answer_fragments, bind_answer_ranges

        payload["answer_fragments"] = answer_fragments(focus, answer)
        independent = await provider.structured(
            AnswerRangePredicates, "check_answer_predicate_ranges", payload
        )
        atoms = bind_answer_ranges(independent, focus, answer)
    else:
        independent = await provider.structured(
            AnswerPredicates, "check_answer_predicates", payload
        )
        atoms = bind_answer_predicates(independent, focus, answer)
    # Hide independent classifications/reasons from the comparative request.
    # The source classification must come from actual cited evidence.
    projection_input = (
        [{"predicate_id": p["predicate_id"], "answer_parts": p["answer_parts"]} for p in atoms]
        if use_parts
        else [
            {k: p[k] for k in ("predicate_id", "original_quote", "source_start", "source_end")}
            for p in atoms
        ]
    )
    source_payload = {
        "query": query,
        "answer": answer,
        "focus_answer_span_id": focus["span_id"],
        "answer_context_spans": context,
        "answer_predicates": projection_input,
        "sources": [
            {**s, "evidence_spans": evidence_spans(s)}
            for s in sources
            if s["source_id"] in actual_source_ids
        ],
        "actual_cited_source_ids": sorted(actual_source_ids),
    }
    single = use_parts and getattr(provider, "supports_check_single_predicate_scope", False)
    paired = single and getattr(provider, "supports_check_source_predicate_parts", False)
    grounded = paired and getattr(provider, "supports_check_grounded_relations", False)
    source_atoms, source_calls, pair_audits = [], 0, []
    if paired:
        from app.answer_ranges import answer_fragments
        from app.source_parts import SourcePredicateParts, bind_source_parts

        if not source_evidence:
            raise CheckProtocolError("来源谓词核验前缺少上游已核验的真实证据位置")
        cache = source_projection_cache if source_projection_cache is not None else {}
        known_sources = {s["source_id"]: s for s in sources}
        if len(known_sources) != len(sources):
            raise CheckProtocolError("来源谓词核验发现重复文档ID，不能覆盖原文")
        for evidence in source_evidence:
            source = known_sources.get(evidence["source_id"])
            if source is None or evidence["source_id"] not in actual_source_ids:
                raise CheckProtocolError("来源谓词核验引用不是答案实际引用")
            registered = next(
                (s for s in evidence_spans(source) if s["span_id"] == evidence["span_id"]), None
            )
            if registered is None or any(
                registered[n] != evidence[n] for n in ("quote", "source_start", "source_end")
            ):
                raise CheckProtocolError("来源谓词核验原文位置不再匹配")
        # Basic check references are validated above, but are not an exhaustive
        # index of all support in an actual cited parent. A later independent
        # scope review may legitimately need a different span of that source.
        projection_evidence = (
            [
                {"source_id": s["source_id"], **span}
                for s in sources
                if s["source_id"] in actual_source_ids
                for span in evidence_spans(s)
            ]
            if grounded
            else source_evidence
        )
        use_windows = grounded and getattr(provider, "supports_check_source_windows", False)
        if use_windows:
            from app.source_windows import project_source_windows

            source_atoms, source_calls = await project_source_windows(
                provider, [s for s in sources if s["source_id"] in actual_source_ids], cache
            )
        for evidence in [] if use_windows else projection_evidence:
            source = known_sources[evidence["source_id"]]
            registered = next(
                s for s in evidence_spans(source) if s["span_id"] == evidence["span_id"]
            )
            key = (source["source_id"], source["content"], registered["span_id"])
            if key not in cache:
                source_payload_independent = {
                    "source_id": source["source_id"],
                    "source_text": source["content"],
                    "focus_source_span": registered,
                    "source_fragments": answer_fragments(registered, source["content"]),
                }
                _, cache[key] = await project_parts(
                    provider,
                    SourcePredicateParts,
                    "check_source_predicate_parts",
                    source_payload_independent,
                    lambda decision: bind_source_parts(decision, source, registered),
                )
                source_calls += 1
            source_atoms.extend(cache[key])
        source_atoms = list({p["source_predicate_id"]: p for p in source_atoms}.values())
        source_payload["source_predicates"] = [
            {
                **{
                    k: p[k] for k in ("source_predicate_id", "source_id", "span_id", "source_parts")
                },
                **(
                    {"source_window_id": p["source_window_id"], "evidence_refs": p["evidence_refs"]}
                    if "source_window_id" in p
                    else {}
                ),
            }
            for p in source_atoms
        ]
    if single:
        from app.grounded_relations import GroundedRelations, bind_grounded_relations
        from app.source_parts import PairedPredicateScopeDecision, validate_source_pairings

        checks = []
        for part in projection_input:
            request_payload = {
                **source_payload,
                "answer_predicates": [part],
                "focus_predicate_id": part["predicate_id"],
            }
            if grounded and getattr(provider, "supports_check_explicit_predicate_focus", False):
                request_payload = {
                    "focus_answer_predicate": part,
                    "focus_predicate_id": part["predicate_id"],
                    "focus_answer_span_id": focus["span_id"],
                    "query": query,
                    "actual_cited_source_ids": source_payload["actual_cited_source_ids"],
                    "source_predicates": source_payload["source_predicates"],
                    "sources": source_payload["sources"],
                    "answer_context_spans": context,
                }
                if getattr(provider, "supports_check_source_windows", False):
                    request_payload["focus_answer_reading_span"] = focus
            one = await provider.structured(
                GroundedRelations
                if grounded
                else (PairedPredicateScopeDecision if paired else PredicateScopeDecision),
                "check_grounded_relations"
                if grounded
                else (
                    "check_paired_predicate_scope"
                    if paired
                    else "check_single_predicate_parts_scope"
                ),
                request_payload,
            )
            if (
                one.answer_span_id != focus["span_id"]
                or len(one.checks) != 1
                or one.checks[0].predicate_id != part["predicate_id"]
            ):
                raise CheckProtocolError("单谓词核验遗漏、重复或返回其他命题")
            if grounded:
                one = bind_grounded_relations(one, source_atoms)
            if paired:
                pair_audits.extend(validate_source_pairings(one.checks, source_atoms))
            checks.extend(one.checks)
        review = PredicateScopeDecision(answer_span_id=focus["span_id"], checks=checks)
    else:
        review = await provider.structured(
            PredicateScopeDecision,
            "check_predicate_parts_scope" if use_parts else "check_predicate_scope",
            source_payload,
        )
    if review.answer_span_id != focus["span_id"] or {c.predicate_id for c in review.checks} != {
        p["predicate_id"] for p in atoms
    }:
        raise CheckProtocolError("逐谓词范围核验遗漏或返回其他命题")
    known = {s["source_id"]: s for s in sources}
    parsed = {p["predicate_id"]: p for p in atoms}
    verified_sources, details = set(), []
    for claim in review.checks:
        atom, bindings = parsed[claim.predicate_id], []
        for ref in claim.evidence:
            if ref.source_id not in actual_source_ids:
                raise CheckProtocolError("逐谓词范围核验引用不是答案实际引用")
            bound = match_evidence(
                SimpleNamespace(span_id=ref.span_id, quote=""), known.get(ref.source_id)
            )
            if not bound["passed"]:
                raise CheckProtocolError("逐谓词范围核验原文位置不存在")
            bindings.append(bound)
            verified_sources.add(ref.source_id)
        # 没有 settings 的 Provider 走保守回退（严格相等），与同一函数里
        # part_coverage_retry_limit 的回退 0 同一约定：未声明配置即按最严处理。
        no_upgrade = getattr(
            getattr(provider, "settings", None), "scope_no_upgrade_relaxation", False
        )
        flags = {name: getattr(claim, name) == "preserved" for name in PREDICATES}
        modes = (
            atom["mode"] != "undetermined"
            and claim.source_mode != "undetermined"
            and (atom["mode"] == claim.source_mode or (no_upgrade and atom["mode"] != "asserted"))
        )
        voices = (
            atom["voice"] != "undetermined"
            and claim.source_voice != "undetermined"
            and (atom["voice"] == claim.source_voice or (no_upgrade and atom["voice"] != "fact"))
        )
        flags["modality_preserved"] = flags["modality_preserved"] and modes
        flags["attribution_preserved"] = flags["attribution_preserved"] and voices
        details.append(
            {
                **claim.model_dump(),
                **flags,
                "answer_predicate": atom,
                "literal_evidence": bindings,
                "mode_matches_this_predicate": modes,
                "voice_matches_this_predicate": voices,
            }
        )
    if verified_sources != actual_source_ids:
        raise CheckProtocolError("逐谓词范围核验遗漏答案实际引用")
    flags = {name: all(d[name] for d in details) for name in PREDICATES}
    return {
        "answer_span_id": focus["span_id"],
        "source_ids": sorted(actual_source_ids),
        **flags,
        "reason": atomic_scope_feedback(details),
        "predicate_checks": details,
        **({"scope_model_requests": 1 + source_calls + len(projection_input)} if single else {}),
        **(
            {
                "independent_source_predicates": source_atoms,
                "source_pairing_checks": pair_audits,
                "source_projection_has_question_or_answer": False,
                **(
                    {"source_category_origin": "independent_literal_source_predicates_only"}
                    if grounded
                    else {}
                ),
            }
            if paired
            else {}
        ),
    }, independent.model_dump()
