"""Resolve selected registry IDs; never infer or repair model-supplied evidence."""

import json
from copy import deepcopy

from pydantic import Field, create_model

from app.grounded_relations import GroundedRelation, GroundedRelations
from app.schemas import StrictModel

Selection = create_model(
    "GroundedSelection",
    __base__=StrictModel,
    **{
        name: (field.annotation, deepcopy(field))
        for name, field in GroundedRelation.model_fields.items()
        if name != "evidence"
    },
)
Selections = create_model(
    "GroundedSelections",
    __base__=StrictModel,
    answer_span_id=(str, deepcopy(GroundedRelations.model_fields["answer_span_id"])),
    checks=(list[Selection], Field(min_length=1, max_length=30)),
)


def selection_wire(schema: dict) -> dict:
    result = deepcopy(schema)
    definition = result["$defs"]["GroundedRelation"]
    definition["properties"].pop("evidence")
    definition["required"].remove("evidence")
    return result


def parse_grounded_selection(response_text: str, payload: dict) -> GroundedRelations:
    from app.answer_ranges import answer_fragments
    from app.evidence_protocol import evidence_spans
    from app.providers import CheckProtocolError
    from app.source_windows import predicate_refs

    value = json.loads(response_text)
    # A legacy explicit response is validated as-is. Never discard or fill an
    # invalid supplied position, even when the selected predicate ID is valid.
    if isinstance(value, dict) and isinstance(value.get("checks"), list):
        supplied = [isinstance(c, dict) and "evidence" in c for c in value["checks"]]
        if any(supplied):
            if not all(supplied):
                raise CheckProtocolError("不可混用显式位置与ID绑定协议")
            return GroundedRelations.model_validate(value)
    selection = Selections.model_validate(value)
    if (
        selection.answer_span_id != payload["focus_answer_span_id"]
        or len(selection.checks) != 1
        or selection.checks[0].predicate_id != payload["focus_predicate_id"]
    ):
        raise CheckProtocolError("ID绑定核验返回非当前唯一答案谓词")
    registry = {p["source_predicate_id"]: p for p in payload["source_predicates"]}
    sources = {s["source_id"]: s for s in payload["sources"]}
    if len(registry) != len(payload["source_predicates"]) or len(sources) != len(
        payload["sources"]
    ):
        raise CheckProtocolError("ID绑定注册表存在重复来源或谓词")
    actual = set(payload["actual_cited_source_ids"])
    checks = []
    for claim in selection.checks:
        ids = claim.source_predicate_ids + claim.context_source_predicate_ids
        if not claim.source_predicate_ids or len(ids) != len(set(ids)):
            raise CheckProtocolError("ID绑定事实证据不能为空或跨角色重复")
        refs = []
        for key in ids:
            part = registry.get(key)
            if part is None or part["source_id"] not in actual:
                raise CheckProtocolError("ID绑定引用未知谓词或答案未引用来源")
            source = sources.get(part["source_id"])
            if "source_window_id" in part:
                from app.source_windows import source_windows, window_fragments

                window = (
                    next(
                        (
                            w
                            for w in source_windows(source)
                            if w["window_id"] == part["source_window_id"]
                        ),
                        None,
                    )
                    if source
                    else None
                )
                if window is None:
                    raise CheckProtocolError("ID绑定的来源窗口不存在")
                fragments = {p["fragment_id"]: p for p in window_fragments(source, window)}
                if not part["source_parts"]:
                    raise CheckProtocolError("ID绑定来源窗口缺少真实原文部分")
                for positioned in part["source_parts"]:
                    registered = fragments.get(positioned["fragment_id"])
                    if registered is None or any(
                        type(positioned[n]) is not type(registered[n])
                        or positioned[n] != registered[n]
                        for n in ("source_start", "source_end", "quote", "span_id")
                    ):
                        raise CheckProtocolError("ID绑定来源窗口的文本或位置不匹配")
                for sid, span_id in sorted(predicate_refs(part)):
                    ref = {"source_id": sid, "span_id": span_id}
                    if ref not in refs:
                        refs.append(ref)
                continue
            span = (
                next((s for s in evidence_spans(source) if s["span_id"] == part["span_id"]), None)
                if source
                else None
            )
            if span is None:
                raise CheckProtocolError("ID绑定的原文位置不存在")
            fragments = {f["fragment_id"]: f for f in answer_fragments(span, source["content"])}
            if not part["source_parts"]:
                raise CheckProtocolError("ID绑定缺少真实原文部分")
            for positioned in part["source_parts"]:
                registered = fragments.get(positioned["fragment_id"])
                if (
                    type(positioned["source_start"]) is not int
                    or type(positioned["source_end"]) is not int
                    or type(positioned["quote"]) is not str
                    or registered is None
                    or any(
                        registered[n] != positioned[n]
                        for n in ("source_start", "source_end", "quote")
                    )
                ):
                    raise CheckProtocolError("ID绑定的原文文本与字符位置不匹配")
            ref = {"source_id": part["source_id"], "span_id": part["span_id"]}
            if ref not in refs:
                refs.append(ref)
        checks.append({**claim.model_dump(), "evidence": refs})
    return GroundedRelations.model_validate(
        {"answer_span_id": selection.answer_span_id, "checks": checks}
    )
