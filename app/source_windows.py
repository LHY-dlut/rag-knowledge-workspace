"""Independent source propositions across bounded contiguous reading windows.

Each selected part retains its canonical evidence span, text and full-source
offsets. A window is a reading aid, never a new citation or a factual verdict.
"""

from pydantic import Field, model_validator

from app.schemas import AnswerRangePredicate, StrictModel


class SourceWindowParts(StrictModel):
    source_id: str = Field(min_length=1, max_length=30)
    source_window_id: str = Field(min_length=1, max_length=60)
    predicates: list[AnswerRangePredicate] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def unique_predicates(self):
        ids = [p.predicate_id for p in self.predicates]
        if len(ids) != len(set(ids)):
            raise ValueError("Source window predicate IDs must be unique")
        return self


def source_windows(source: dict) -> list[dict]:
    from app.evidence_protocol import evidence_spans

    spans = evidence_spans(source)
    groups, current = [], []
    for span in spans:
        if current and (span["source_end"] - current[0]["source_start"] > 800 or len(current) >= 8):
            groups.append(current)
            current = []
        current.append(span)
    if current:
        groups.append(current)
    return [
        {
            "window_id": f"{source['source_id']}:W{i + 1}",
            "source_start": group[0]["source_start"],
            "source_end": group[-1]["source_end"],
            "quote": source["content"][group[0]["source_start"] : group[-1]["source_end"]],
            "evidence_spans": group,
        }
        for i, group in enumerate(groups)
    ]


def window_fragments(source: dict, window: dict) -> list[dict]:
    from app.answer_ranges import answer_fragments

    return [
        {**part, "span_id": span["span_id"]}
        for span in window["evidence_spans"]
        for part in answer_fragments(span, source["content"])
    ]


def bind_source_window(decision: SourceWindowParts, source: dict, window: dict) -> list[dict]:
    from app.answer_parts import required_fragment_ids
    from app.providers import CheckProtocolError

    actual = next(
        (w for w in source_windows(source) if w["window_id"] == window["window_id"]), None
    )
    if (
        actual != window
        or decision.source_id != source["source_id"]
        or decision.source_window_id != window["window_id"]
    ):
        raise CheckProtocolError("来源窗口ID、原文或位置不匹配")
    fragments = window_fragments(source, window)
    known = {p["fragment_id"]: (i, p) for i, p in enumerate(fragments)}
    required = set(required_fragment_ids(fragments, source["content"]))
    covered, rows = set(), []
    for predicate in decision.predicates:
        if any(key not in known for key in predicate.fragment_ids):
            raise CheckProtocolError("来源窗口选择了其他窗口或不存在的原文部分")
        positions = [known[key][0] for key in predicate.fragment_ids]
        if positions != sorted(positions):
            raise CheckProtocolError("来源窗口部分不得重排")
        selected = set(predicate.fragment_ids) & required
        if not selected:
            raise CheckProtocolError("来源窗口谓词缺少真实事实文本")
        covered.update(selected)
        parts = [dict(known[key][1]) for key in predicate.fragment_ids]
        refs = list(dict.fromkeys((source["source_id"], p["span_id"]) for p in parts))
        rows.append(
            {
                "source_predicate_id": f"{source['source_id']}:{window['window_id']}:{predicate.predicate_id}",
                "source_id": source["source_id"],
                "span_id": parts[0]["span_id"],
                "source_window_id": window["window_id"],
                "source_parts": parts,
                "evidence_refs": [{"source_id": sid, "span_id": span} for sid, span in refs],
                "mode": predicate.mode,
                "voice": predicate.voice,
                "reason": predicate.reason,
                "literal_binding": {
                    "passed": True,
                    "match_mode": "server_positioned_source_window_parts",
                    "offset_unit": "unchanged full source Unicode codepoint",
                    "semantic_support_checked": False,
                    "composite_quote_created": False,
                    "source_id": source["source_id"],
                    "window_id": window["window_id"],
                    "parts": parts,
                },
            }
        )
    if covered != required:
        raise CheckProtocolError("来源窗口投影遗漏事实或范围限定")
    return rows


def predicate_refs(part: dict) -> set[tuple[str, str]]:
    """Window predicates can reference several existing canonical spans."""
    if "source_window_id" not in part:
        return {(part["source_id"], part["span_id"])}
    from app.providers import CheckProtocolError

    refs = {(part["source_id"], p["span_id"]) for p in part["source_parts"]}
    supplied = {(r["source_id"], r["span_id"]) for r in part["evidence_refs"]}
    if not refs or supplied != refs or len(supplied) != len(part["evidence_refs"]):
        raise CheckProtocolError("来源窗口的真实证据位置注册表不一致")
    return refs


async def project_source_windows(
    provider, sources: list[dict], cache: dict
) -> tuple[list[dict], int]:
    rows, calls = [], 0
    for source in sources:
        for window in source_windows(source):
            key = ("source_window", source["source_id"], source["content"], window["window_id"])
            if key not in cache:
                decision = await provider.structured(
                    SourceWindowParts,
                    "check_source_predicate_window",
                    {
                        "source_id": source["source_id"],
                        "source_text": source["content"],
                        "focus_source_window": window,
                        "source_fragments": window_fragments(source, window),
                    },
                )
                cache[key] = bind_source_window(decision, source, window)
                calls += 1
            rows.extend(cache[key])
    return rows, calls
