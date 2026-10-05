"""Describe actual vetoes; raw model reasons remain in predicate_checks."""

FIELDS = {
    "subject_predicate_preserved": "主体事实",
    "modality_preserved": "语气",
    "temporal_scope_preserved": "时间范围",
    "conditions_preserved": "条件",
    "attribution_preserved": "归属",
}


def atomic_scope_feedback(details: list[dict]) -> str:
    messages = []
    for row in details:
        failed = {name for name in FIELDS if not row[name]}
        if not failed:
            continue
        parts = []
        atom = row["answer_predicate"]
        if not row["mode_matches_this_predicate"]:
            parts.append(
                f"独立语气类别不一致或待定：答案={atom['mode']}，原文={row['source_mode']}"
            )
            failed.discard("modality_preserved")
        if not row["voice_matches_this_predicate"]:
            parts.append(
                f"独立归属类别不一致或待定：答案={atom['voice']}，原文={row['source_voice']}"
            )
            failed.discard("attribution_preserved")
        if failed:
            names = "、".join(FIELDS[name] for name in FIELDS if name in failed)
            parts.append(f"模型关系核验未通过（{names}）；模型原始说明：{row['reason']}")
        else:
            parts.append("比较模型的原始理由保留在predicate_checks.reason，不代表后端通过")
        messages.append(f"{row['predicate_id']}：" + "；".join(parts))
    # Bound only the derived summary; never truncate/repair model responses.
    return "；".join(messages)[:300] or "所有逐谓词关系及独立语气归属均保留"
