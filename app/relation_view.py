"""A reversible wire view of one relation; no semantic decision or response repair."""

import hashlib
import json
from copy import deepcopy


def relation_wire_view(payload: dict, schema: dict) -> tuple[dict, dict, dict]:
    from app.providers import CheckProtocolError

    registry = payload["source_predicates"]
    ids = [p["source_predicate_id"] for p in registry]
    if len(ids) != len(set(ids)):
        raise CheckProtocolError("比较输入包含重复来源谓词ID")
    aliases = {"R" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]: key for key in ids}
    if len(aliases) != len(ids):
        raise CheckProtocolError("比较输入ID别名冲突")
    reverse = {key: alias for alias, key in aliases.items()}
    parts = []
    for predicate in registry:
        item = {
            k: deepcopy(predicate[k])
            for k in (
                "source_predicate_id",
                "source_id",
                "span_id",
                "source_parts",
                "source_window_id",
                "evidence_refs",
            )
            if k in predicate
        }
        item["source_predicate_id"] = reverse[predicate["source_predicate_id"]]
        parts.append(item)
    view = {
        # Context comes first; the unique current relation comes last. All
        # quotes and offsets remain unchanged. The user's question and other
        # model classifications are not facts for this independent comparison.
        "answer_context_spans": deepcopy(payload["answer_context_spans"]),
        "sources": deepcopy(payload["sources"]),
        "source_predicates": parts,
        "actual_cited_source_ids": deepcopy(payload["actual_cited_source_ids"]),
        "focus_answer_span_id": payload["focus_answer_span_id"],
        "focus_predicate_id": payload["focus_predicate_id"],
        "focus_answer_predicate": deepcopy(payload["focus_answer_predicate"]),
    }
    document = deepcopy(schema)
    properties = document["$defs"]["GroundedRelation"]["properties"]
    for key in ("source_predicate_ids", "context_source_predicate_ids"):
        properties[key]["items"]["enum"] = list(aliases)
    return view, document, aliases


def parse_relation_view(response_text: str, aliases: dict, original_payload: dict):
    from app.grounded_selection import Selections, parse_grounded_selection
    from app.providers import CheckProtocolError

    # Validate the entire wire structure before translating identifiers.
    # Unknown, missing, boolean, duplicate or overlapping selections are never
    # repaired. The existing binder still validates roles, citations and every
    # real position against the original registry after translation.
    Selections.model_validate_json(response_text)
    value = json.loads(response_text)
    for check in value["checks"]:
        for field in ("source_predicate_ids", "context_source_predicate_ids"):
            if field not in check:
                continue
            actual = []
            for alias in check[field]:
                if alias not in aliases:
                    raise CheckProtocolError("比较返回未知来源谓词别名")
                actual.append(aliases[alias])
            check[field] = actual
    return parse_grounded_selection(json.dumps(value, ensure_ascii=False), original_payload)
