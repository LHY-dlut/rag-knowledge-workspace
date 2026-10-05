"""Bind fact evidence and scope context to the unchanged source registry."""


def bind_evidence_roles(claim, registry: list[dict]) -> tuple[list[dict], list[dict]]:
    from app.providers import CheckProtocolError
    from app.source_windows import predicate_refs

    known = {p["source_predicate_id"]: p for p in registry}
    if len(known) != len(registry):
        raise CheckProtocolError("证据角色注册表存在重复谓词ID")
    primary_ids = claim.source_predicate_ids
    context_ids = claim.context_source_predicate_ids
    ids = primary_ids + context_ids
    if not primary_ids or len(ids) != len(set(ids)):
        raise CheckProtocolError("事实证据不能为空，且不能与范围上下文重复")
    evidence = {(r.source_id, r.span_id) for r in claim.evidence}
    selected = []
    for key in ids:
        part = known.get(key)
        if part is None or not predicate_refs(part).issubset(evidence):
            raise CheckProtocolError("证据角色所选谓词不属于当前真实证据位置")
        selected.append(part)
    if set().union(*(predicate_refs(p) for p in selected)) != evidence:
        raise CheckProtocolError("证据角色所选谓词遗漏当前真实证据位置")
    return selected[: len(primary_ids)], selected[len(primary_ids) :]
