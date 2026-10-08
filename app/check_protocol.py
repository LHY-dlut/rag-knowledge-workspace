"""Bind semantic check output to the actual answer and immutable source positions.

Literal binding establishes provenance, not entailment. Model semantic rejection
is never upgraded by this protocol, and incomplete check coverage fails closed.
"""

import json
import re
from types import SimpleNamespace

from app.evidence_protocol import MARKER, evidence_spans, match_evidence
from app.schemas import BoundCheckJudgment, CheckDecision


def check_retry_feedback(judgment) -> str:
    """Explain missing provenance checks; never change verdicts or references."""
    audit = getattr(judgment, "evidence_protocol", None) or {}
    if judgment.passed or not audit.get("claims"):
        return judgment.reason
    missing = []
    for claim in audit.get("claims", []):
        actual = set(claim.get("actual_cited_source_ids", []))
        verified = {e["source_id"] for e in claim.get("evidence", []) if e.get("passed")}
        unverified = sorted(actual - verified)
        if unverified:
            missing.append(
                {
                    "answer_span_id": claim.get("answer_binding", {}).get("selected_span_id"),
                    "unverified_cited_source_ids": unverified,
                }
            )
    if not missing:
        return judgment.reason
    return (
        judgment.reason
        + "。下面列出答案实际引用但尚未获得有效证据绑定的来源；这不表示事实已通过核验。"
        "重新依据原文生成完整陈述，只引用确实支持该事实的必要来源；"
        "同一事实的相互一致重复来源无需全部堆叠，跨来源事实仍须分别有依据。"
        "若来源冲突，保留差异和范围，不得静默挑选有利来源。"
        "不得制造或自动补全引用；新草稿的每个实际引用仍须独立核验。"
        + json.dumps(missing, ensure_ascii=False)
    )


def citation_scopes(answer: str) -> tuple[set[int], list[dict]]:
    """Group existing adjacent markers; bind preceding uncited paragraph text.

    A sentence-final shared marker can cover multiple clauses. A marker cannot
    reach across a blank paragraph or across a preceding citation group. These
    are rendering/position rules only; the model must verify each cited source.
    """
    markers, groups = list(MARKER.finditer(answer)), []
    marker_positions = {i for m in markers for i in range(m.start(), m.end())}
    for marker in markers:
        if groups and not answer[groups[-1]["marker_end"] : marker.start()].strip():
            groups[-1]["source_ids"].add(marker[1])
            groups[-1]["marker_end"] = marker.end()
        else:
            previous = groups[-1]["marker_end"] if groups else 0
            breaks = list(re.finditer(r"\r?\n\s*\r?\n", answer[previous : marker.start()]))
            start = previous + breaks[-1].end() if breaks else previous
            groups.append(
                {
                    "text_start": start,
                    "text_end": marker.start(),
                    "marker_end": marker.end(),
                    "source_ids": {marker[1]},
                }
            )
    return marker_positions, groups


def answer_spans(answer: str) -> list[dict]:
    markers, _ = citation_scopes(answer)
    return [
        span
        for span in evidence_spans({"source_id": "A", "content": answer})
        if any(
            answer[i].isalnum() and i not in markers
            for i in range(span["source_start"], span["source_end"])
        )
    ]


def answer_citation_manifest(answer: str) -> list[dict]:
    """Annotate unchanged answer spans with their actual citation positions.

    This does not select supporting evidence, insert references, or change
    coverage rules. The semantic model still checks every asserted fact.
    """
    marker_positions, scopes = citation_scopes(answer)
    result = []
    for span in answer_spans(answer):
        natural = {
            i
            for i in range(span["source_start"], span["source_end"])
            if answer[i].isalnum() and i not in marker_positions
        }
        segments, cited = [], set()
        for scope in scopes:
            if natural & set(range(scope["text_start"], scope["text_end"])):
                ids = sorted(scope["source_ids"])
                cited.update(ids)
                segments.append(
                    {
                        "text_start": max(span["source_start"], scope["text_start"]),
                        "text_end": min(span["source_end"], scope["text_end"]),
                        "source_ids": ids,
                    }
                )
        result.append(
            {
                "answer_span_id": span["span_id"],
                "cited_source_ids": sorted(cited),
                "citation_segments": segments,
            }
        )
    return result


def answer_citation_coverage(answer: str) -> dict:
    """Apply existing citation position rules before paying for semantic check.

    No marker is added and no text is treated as nonfactual. This is only a
    position preflight; a fully cited answer still requires semantic checking.
    """
    marker_positions, scopes = citation_scopes(answer)
    required = {i for i, c in enumerate(answer) if c.isalnum()} - marker_positions
    covered = set()
    for scope in scopes:
        covered.update(range(scope["text_start"], scope["text_end"]))
    missing = required - covered
    fragments = [
        span
        for span in answer_spans(answer)
        if missing & set(range(span["source_start"], span["source_end"]))
    ]
    return {
        "passed": bool(required) and not missing,
        "failure_types": ["check_claim_citation_missing"] if missing or not required else [],
        "uncovered_answer_positions": sorted(missing),
        "uncited_answer_spans": fragments,
        "rule": "existing citation groups cover preceding text in the same paragraph only",
        "semantic_support_checked": False,
        "references_added": False,
    }


def answer_year_binding(answer: str, sources: list[dict]) -> dict:
    """答案里出现的年份必须能在它实际引用的来源里找到。

    确定性规则，不调用模型，与 answer_citation_coverage 同属引用位置预检。
    防的是"把问题里的年份回写进答案"：问题问 2021 年、语料是 2023 年新闻，
    答案写成"2021年7月，某指标为49.3%[S1]"，而 S1 原文只有"7月份"未标年份。

    只按该处实际引用到的来源比对，不取全部来源的并集——否则无关来源里偶然
    出现的同一年份会让检查失效。
    """
    known = {s["source_id"]: (s.get("content") or "") for s in sources}
    _, scopes = citation_scopes(answer)
    violations = []
    for match in re.finditer(r"(?:19|20)\d{2}", answer):
        cited = {
            source_id
            for scope in scopes
            if scope["text_start"] <= match.start() < scope["text_end"]
            for source_id in scope["source_ids"]
        }
        if not cited:
            # 未被任何引用覆盖：由 answer_citation_coverage 负责，这里不重复报
            continue
        if any(match.group(0) in known.get(source_id, "") for source_id in cited):
            continue
        violations.append(
            {
                "year": match.group(0),
                "position": match.start(),
                "cited_source_ids": sorted(cited),
            }
        )
    return {
        "passed": not violations,
        "failure_types": ["check_year_not_in_cited_source"] if violations else [],
        "unsupported_years": violations,
        "rule": "a year stated in the answer must appear in a source cited at that position",
        "semantic_support_checked": False,
    }


def bind_check(decision: CheckDecision, answer: str, sources: list[dict]) -> BoundCheckJudgment:
    known = {s["source_id"]: s for s in sources}
    marker_positions, scopes = citation_scopes(answer)
    required = {i for i, c in enumerate(answer) if c.isalnum()} - marker_positions
    registry = {s["span_id"]: s for s in answer_spans(answer)}
    covered, rows, errors, seen = set(), [], [], set()
    for claim in decision.checks:
        claim_match = match_evidence(
            SimpleNamespace(span_id=claim.answer_span_id, quote=""),
            {"source_id": "A", "content": answer},
        )
        evidence, positions = [], set()
        if not claim_match["passed"] or claim.answer_span_id not in registry:
            errors.append("check_claim_not_in_answer")
        else:
            positions.update(range(claim_match["source_start"], claim_match["source_end"]))
            covered.update(positions)
        if claim.answer_span_id in seen:
            errors.append("check_answer_span_duplicate")
        seen.add(claim.answer_span_id)
        natural = positions & required
        cited, cited_positions = set(), set()
        for scope in scopes:
            overlap = natural & set(range(scope["text_start"], scope["text_end"]))
            if overlap:
                cited.update(scope["source_ids"])
                cited_positions.update(overlap)
        for ref in claim.evidence:
            bound = match_evidence(
                SimpleNamespace(span_id=ref.span_id, quote=""), known.get(ref.source_id)
            )
            evidence.append(bound)
            if not bound["passed"]:
                errors.append(bound["failure_type"])
            if claim.verdict == "supported" and ref.source_id not in cited:
                errors.append("check_evidence_not_cited_by_claim")
        if claim.verdict == "supported" and (not cited or natural - cited_positions):
            errors.append("check_claim_citation_missing")
        if claim.verdict == "supported" and cited - {r.source_id for r in claim.evidence}:
            errors.append("check_cited_source_not_verified")
        rows.append(
            {
                "answer_binding": claim_match,
                "actual_cited_source_ids": sorted(cited),
                "evidence": evidence,
            }
        )
    missing = sorted(required - covered)
    if missing or not required or set(registry) - seen:
        errors.append("check_answer_coverage_incomplete")
    audit = {
        "passed": not errors,
        "failure_types": sorted(set(errors)),
        "claims": rows,
        "uncovered_answer_positions": missing,
        "normalization": "server-issued answer/source span IDs retain exact raw positions; no semantic similarity",
        "citation_scope": "existing adjacent marker groups cover preceding uncited text in the same paragraph; never cross blank lines",
        "semantic_verdict_source": "model claim-by-claim judgment; provenance is not factual support",
    }
    semantic_passed = all(c.verdict == "supported" for c in decision.checks)
    reason = (
        ";".join(c.reason for c in decision.checks if c.verdict != "supported")[:1000]
        or "所有答案片段的事实与范围均获实际引用证据支持"
    )
    return BoundCheckJudgment(
        passed=semantic_passed and audit["passed"],
        reason=reason if audit["passed"] else "核验协议失败：" + ",".join(audit["failure_types"]),
        checks=decision.checks,
        evidence_protocol=audit,
    )
