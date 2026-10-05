"""Literal answer range selection; never decide semantic support or add content."""

import re

from app.check_protocol import citation_scopes
from app.providers import CheckProtocolError
from app.schemas import AnswerRangePredicates

FRAGMENT_END = re.compile(r"[，,、:：]+")


def answer_fragments(focus: dict, answer: str) -> list[dict]:
    offset, limit = focus["source_start"], focus["source_end"]
    if answer[offset:limit] != focus["quote"]:
        raise CheckProtocolError("答案片段注册表与当前原答案不一致")
    text = focus["quote"]
    boundaries = [m.end() for m in FRAGMENT_END.finditer(text)] + [len(text)]
    start, rows = 0, []
    for end in boundaries:
        if end <= start:
            continue
        rows.append(
            {
                "fragment_id": f"{focus['span_id']}:F{len(rows) + 1}",
                "source_start": offset + start,
                "source_end": offset + end,
                "quote": answer[offset + start : offset + end],
            }
        )
        start = end
    return rows


def bind_answer_ranges(decision: AnswerRangePredicates, focus: dict, answer: str) -> list[dict]:
    if decision.answer_span_id != focus["span_id"]:
        raise CheckProtocolError("逐谓词范围选择返回非当前答案片段")
    fragments = answer_fragments(focus, answer)
    known = {f["fragment_id"]: i for i, f in enumerate(fragments)}
    marker_positions, _ = citation_scopes(answer)
    required = {
        i
        for i in range(focus["source_start"], focus["source_end"])
        if answer[i].isalnum() and i not in marker_positions
    }
    covered, rows = set(), []
    for predicate in decision.predicates:
        if any(fid not in known for fid in predicate.fragment_ids):
            raise CheckProtocolError("逐谓词范围选择包含未知或其他片段ID")
        indices = [known[fid] for fid in predicate.fragment_ids]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise CheckProtocolError("逐谓词范围必须按原文顺序连续，不得重排或跳块拼接")
        start = fragments[indices[0]]["source_start"]
        end = fragments[indices[-1]]["source_end"]
        natural = set(range(start, end)) & required
        if not natural:
            raise CheckProtocolError("逐谓词范围只有标点或引用，不能核验事实")
        covered.update(natural)
        rows.append(
            {
                **predicate.model_dump(),
                "source_start": start,
                "source_end": end,
                "original_quote": answer[start:end],
                "literal_binding": {
                    "passed": True,
                    "match_mode": "server_fragment_range",
                    "source_id": "A",
                    "focus_answer_span_id": focus["span_id"],
                    "selected_fragment_ids": predicate.fragment_ids,
                    "source_start": start,
                    "source_end": end,
                    "original_quote": answer[start:end],
                    "normalization_map": [(i, i + 1) for i in range(start, end)],
                    "removed_whitespace_spans": [],
                    "offset_unit": "unchanged full answer Unicode codepoint",
                    "semantic_support_checked": False,
                },
            }
        )
    if covered != required:
        raise CheckProtocolError("逐谓词范围选择遗漏答案内容或范围限定")
    return rows
