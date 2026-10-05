"""Positioned answer parts; never splice noncontiguous text into a quotation."""

from app.answer_ranges import answer_fragments
from app.check_protocol import citation_scopes
from app.providers import CheckProtocolError
from app.schemas import AnswerRangePredicates


def required_fragment_ids(fragments: list[dict], answer: str) -> list[str]:
    markers, _ = citation_scopes(answer)
    return [
        part["fragment_id"]
        for part in fragments
        if any(
            answer[i].isalnum() and i not in markers
            for i in range(part["source_start"], part["source_end"])
        )
    ]


def bind_answer_parts(decision: AnswerRangePredicates, focus: dict, answer: str) -> list[dict]:
    if decision.answer_span_id != focus["span_id"]:
        raise CheckProtocolError("逐谓词片段选择返回非当前答案片段")
    fragments = answer_fragments(focus, answer)
    known = {f["fragment_id"]: i for i, f in enumerate(fragments)}
    required = set(required_fragment_ids(fragments, answer))
    covered, rows = set(), []
    for predicate in decision.predicates:
        if any(fid not in known for fid in predicate.fragment_ids):
            raise CheckProtocolError("逐谓词片段选择包含未知或其他片段ID")
        indices = [known[fid] for fid in predicate.fragment_ids]
        if indices != sorted(indices):
            raise CheckProtocolError("逐谓词片段必须按原文顺序，不得重排")
        natural = set(predicate.fragment_ids) & required
        if not natural:
            raise CheckProtocolError("逐谓词片段只有标点或引用，不能核验事实")
        covered.update(natural)
        parts = [dict(fragments[i]) for i in indices]
        rows.append(
            {
                **predicate.model_dump(),
                "answer_parts": parts,
                "literal_binding": {
                    "passed": True,
                    "match_mode": "server_positioned_parts",
                    "source_id": "A",
                    "focus_answer_span_id": focus["span_id"],
                    "parts": [
                        {
                            **part,
                            "normalization_map": [
                                (i, i + 1) for i in range(part["source_start"], part["source_end"])
                            ],
                        }
                        for part in parts
                    ],
                    "offset_unit": "unchanged full answer Unicode codepoint",
                    "semantic_support_checked": False,
                    "composite_quote_created": False,
                },
            }
        )
    if covered != required:
        raise CheckProtocolError("逐谓词片段选择遗漏答案内容或范围限定")
    return rows


def constrain_parts_wire(schema: dict, payload: dict) -> dict:
    """Constrain this request's IDs, without changing schemas or repairing output."""
    fragments = payload.get("answer_fragments")
    if not fragments:
        return schema
    ids = [part["fragment_id"] for part in fragments]
    schema["$defs"]["AnswerRangePredicate"]["properties"]["fragment_ids"]["items"]["enum"] = ids
    schema["properties"]["answer_span_id"]["enum"] = [payload["focus_answer_span_id"]]
    schema["allOf"] = [
        {
            "properties": {
                "predicates": {
                    "contains": {
                        "properties": {"fragment_ids": {"contains": {"const": fid}}},
                        "required": ["fragment_ids"],
                    }
                }
            }
        }
        for fid in required_fragment_ids(fragments, payload["answer"])
    ]
    return schema
