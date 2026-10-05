"""Expose existing output requirements; never validate or repair model results."""

from app.answer_parts import required_fragment_ids


def check_output_contract(task: str, payload: dict, schema: dict) -> dict:
    if task == "check_grounded_relations":
        fields = schema["$defs"]["GroundedRelation"]["properties"]
        return {
            "expected_answer_span_id": payload["focus_answer_span_id"],
            "expected_predicate_id": payload["focus_predicate_id"],
            "checks_count": 1,
            "relation_values": list(fields["subject_predicate_preserved"]["enum"]),
            "reason_max_characters": fields["reason"]["maxLength"],
            "reason_target_characters": 80,
            "semantic_support_not_prejudged": True,
            "fact_evidence_field": "source_predicate_ids",
            "scope_context_field": "context_source_predicate_ids",
            "allowed_source_predicate_ids": [
                p["source_predicate_id"] for p in payload["source_predicates"]
            ],
            "primary_evidence_required_roles_disjoint": True,
            "evidence_positions_bound_by_backend": True,
            "model_must_not_output_evidence_or_quote": True,
        }
    if task == "check_answer_predicate_parts":
        fragments, content = payload["answer_fragments"], payload["answer"]
        identity = {"expected_answer_span_id": payload["focus_answer_span_id"]}
    elif task in {"check_source_predicate_parts", "check_source_predicate_window"}:
        fragments, content = payload["source_fragments"], payload["source_text"]
        identity = {"expected_source_id": payload["source_id"]}
        if task == "check_source_predicate_window":
            identity["expected_source_window_id"] = payload["focus_source_window"]["window_id"]
        else:
            identity["expected_source_span_id"] = payload["focus_source_span"]["span_id"]
    else:
        raise ValueError("Unsupported output contract task")
    fields = schema["$defs"]["AnswerRangePredicate"]["properties"]
    return {
        **identity,
        "required_fragment_ids": required_fragment_ids(fragments, content),
        "mode_allowed_values": list(fields["mode"]["enum"]),
        "voice_allowed_values": list(fields["voice"]["enum"]),
        "mode_and_voice_are_separate_fields": True,
        "reason_max_characters": fields["reason"]["maxLength"],
        "reason_target_characters": 80,
        "semantic_support_not_prejudged": True,
    }
