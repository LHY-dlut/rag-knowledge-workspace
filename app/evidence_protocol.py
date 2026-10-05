"""Deterministic syntax/attribution checks; these never establish factual support."""

import re

MARKER = re.compile(r"[\[［]\s*(S\d+)\s*[\]］]")
MARKER_START = re.compile(r"[\[［(（]\s*[Ss](?:\d|[-_])")
SPAN_END = re.compile(r"[。！？；;\r\n]+[”’」』）】]*")


def evidence_spans(source: dict) -> list[dict]:
    """Offer literal, bounded fragments, retaining positions in the unchanged source.

    Never interpret an ellipsis, splice text, normalize punctuation or infer support.
    A long sentence is split at 400 Unicode codepoints; legacy continuous quotes
    can still cross these presentation boundaries inside the same actual source.
    """
    text = source["content"]
    boundaries = [m.end() for m in SPAN_END.finditer(text)] + [len(text)]
    result, start = [], 0
    for boundary in boundaries:
        while start < boundary:
            end = min(start + 400, boundary)
            left, right = start, end
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
            if left < right:
                result.append(
                    {
                        "span_id": f"{source['source_id']}:E{len(result) + 1}",
                        "source_start": left,
                        "source_end": right,
                        "quote": text[left:right],
                    }
                )
            start = end
    return result


def grade_reading_source(source: dict) -> dict:
    """Offer full original reading order alongside the canonical span registry.

    This is presentation only, never an extracted fact or a support decision.
    Regenerate IDs and quotes from the unchanged source; do not trust a caller's
    stale reading field or registry. Every byte-budget fallback stays lossless.
    """
    if "evidence_spans" not in source or "content" not in source:
        return dict(source)
    return {
        **{
            k: v
            for k, v in source.items()
            if k not in {"content", "reading_text", "evidence_spans"}
        },
        "reading_text": source["content"],
        "evidence_spans": evidence_spans(source),
    }


def match_evidence(evidence, source: dict | None) -> dict:
    if not evidence.span_id:
        return match_quote(evidence.quote, source)
    audit = {
        "model_quote": evidence.quote,
        "selected_span_id": evidence.span_id,
        "offset_unit": "source-relative Unicode codepoint",
    }
    if source is None:
        return {**audit, "passed": False, "failure_type": "citation_id_unknown"}
    audit.update({k: source.get(k) for k in ["source_id", "document_id", "parent_id"]})
    # Recompute from the actual retrieval source, never from model data or an
    # attached registry supplied by a client/model.
    span = next((s for s in evidence_spans(source) if s["span_id"] == evidence.span_id), None)
    if span is None:
        return {**audit, "passed": False, "failure_type": "evidence_span_unknown"}
    if evidence.quote:
        literal = match_quote(evidence.quote, {**source, "content": span["quote"]})
        if (
            not literal["passed"]
            or literal["source_start"] != 0
            or literal["source_end"] != len(span["quote"])
        ):
            return {**audit, "passed": False, "failure_type": "evidence_span_quote_conflict"}
    start, end = span["source_start"], span["source_end"]
    return {
        **audit,
        "passed": True,
        "match_mode": "server_span",
        "source_start": start,
        "source_end": end,
        "original_quote": source["content"][start:end],
        "normalization_map": [(i, i + 1) for i in range(start, end)],
        "removed_whitespace_spans": [],
    }


def prepare_citations(answer: str, sources: list[dict]) -> tuple[str, dict]:
    known = {s["source_id"] for s in sources}
    matches = list(MARKER.finditer(answer))
    spans = [(m.start(), m.end()) for m in matches]
    unknown = sorted({m[1] for m in matches} - known)
    malformed = [
        m.start()
        for m in MARKER_START.finditer(answer)
        if not any(start == m.start() for start, _ in spans)
    ]
    errors = []
    if unknown:
        errors.append("citation_id_unknown")
    if malformed or not matches or not answer.strip():
        errors.append("citation_format_invalid")
    rendered = MARKER.sub(lambda m: f"[{m[1]}]", answer) if not errors else answer
    return rendered, {
        "passed": not errors,
        "failure_types": errors,
        "unknown_ids": unknown,
        "malformed_positions": malformed,
        "raw_answer": answer,
        "rendered_answer": rendered,
        "citations": [
            {"source_id": m[1], "raw_start": m.start(), "raw_end": m.end()} for m in matches
        ],
        "normalization": "bracket whitespace/fullwidth square brackets only; no ID insertion",
    }


def whitespace_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Map whitespace runs and display space before existing ellipses to raw offsets.

    Other whitespace retains one space so word and numeric boundaries remain intact.
    """
    chars, positions = [], []
    i = 0
    while i < len(text):
        start = i
        if text[i].isspace():
            while i < len(text) and text[i].isspace():
                i += 1
            # Presentation whitespace before an existing literal ellipsis.
            # Never create '.', merge separated dots, or remove word boundaries.
            if text.startswith("...", i) or text.startswith("…", i):
                continue
            chars.append(" ")
        else:
            chars.append(text[i])
            i += 1
        positions.append((start, i))
    return "".join(chars), positions


def match_quote(quote: str, source: dict | None) -> dict:
    result = {"model_quote": quote, "offset_unit": "source-relative Unicode codepoint"}
    if source is None:
        return {**result, "passed": False, "failure_type": "citation_id_unknown"}
    text = source["content"]
    result.update({k: source.get(k) for k in ["source_id", "document_id", "parent_id"]})
    if not quote.strip() or not text.strip():
        return {**result, "passed": False, "failure_type": "context_evidence_missing"}
    start = text.find(quote)
    if start >= 0:
        end = start + len(quote)
        mode = "exact"
        positions = [(i, i + 1) for i in range(start, end)]
    else:
        canonical, mapping = whitespace_map(text)
        needle, _ = whitespace_map(quote)
        needle = needle.strip()
        hits = []
        offset = 0
        while needle and (found := canonical.find(needle, offset)) >= 0:
            hits.append(found)
            offset = found + 1
        if len(hits) != 1:
            return {
                **result,
                "passed": False,
                "failure_type": "evidence_text_mismatch",
                "reason": "no unique continuous match after whitespace normalization",
                "normalized_match_count": len(hits),
            }
        positions = mapping[hits[0] : hits[0] + len(needle)]
        start, end = positions[0][0], positions[-1][1]
        mode = "whitespace_runs"
    original = text[start:end]
    assert original == text[start:end]
    return {
        **result,
        "passed": True,
        "match_mode": mode,
        "source_start": start,
        "source_end": end,
        "original_quote": original,
        "normalization_map": positions,
        "removed_whitespace_spans": [
            (left[1], right[0])
            for left, right in zip(positions, positions[1:])
            if left[1] < right[0]
        ],
    }


def bind_decision(decision, sources: list[dict]):
    from app.schemas import GradeDecision

    if not decision.passed:
        return decision, {"passed": False, "failure_types": ["semantic_support_insufficient"]}
    by_id = {s["source_id"]: s for s in sources}
    matches = [match_evidence(e, by_id.get(e.source_id)) for e in decision.evidence]
    errors = sorted({m["failure_type"] for m in matches if not m["passed"]})
    if not matches:
        errors.append("context_evidence_missing")
    audit = {"passed": not errors, "failure_types": errors, "evidence": matches}
    if errors:
        audit["retry_feedback"] = {
            "instruction": (
                "上轮引用协议校验未通过。仅修复下面的错误摘录，不改变事实支持标准。"
                "从本轮sources中重新选择真实ID及短的连续原文quote；不可改写或跨来源拼接。"
                "缩短摘录时直接结束，不要添加原文不存在的...或省略号。"
                "同一来源的不相邻句子必须分成多条evidence，不能拼成一条quote。"
                "可选择本轮sources中真实span_id并省略quote；span只确认原文位置，不证明事实支持。"
                "subject、attribute、answer_scope仍须由原文支持；无法支持就返回insufficient。"
            ),
            "invalid_evidence": [
                {
                    "source_id": e.source_id,
                    "model_quote": e.quote,
                    "failure_type": m["failure_type"],
                    **({"span_id": e.span_id} if e.span_id else {}),
                }
                for e, m in zip(decision.evidence, matches, strict=True)
                if not m["passed"]
            ],
        }
        return GradeDecision(
            passed=False,
            support="insufficient",
            reason="证据协议失败:"
            + ",".join(errors)
            + "；每条quote须为对应ID原文的短连续摘录，不得改写、省略拼接或跨来源拼接。",
        ), audit
    evidence = [
        e.model_copy(update={"quote": m["original_quote"]})
        for e, m in zip(decision.evidence, matches, strict=True)
    ]
    return decision.model_copy(update={"evidence": evidence}), audit
