import json

import pytest

from scripts import human_evaluation as protocol


def test_fair_arms_keep_query_processing_and_budgets_identical():
    plan = protocol.make_plan()
    arms = plan["retrieval_comparison"]["configs"]
    baseline = {k: v for k, v in arms["dense"].items() if k not in {"hybrid", "rerank"}}
    for cfg in arms.values():
        assert {k: v for k, v in cfg.items() if k not in {"hybrid", "rerank"}} == baseline
        assert not any(cfg[k] for k in ["query_rewrite", "multi_query", "hyde"])
    ablation = plan["coreference_ablation"]["configs"]
    changed = {
        k
        for k in ablation["rewrite_off"]
        if ablation["rewrite_off"][k] != ablation["rewrite_on"][k]
    }
    assert changed == {"query_rewrite"}
    assert plan["retrieval_comparison"]["questions"].endswith("history=[]")


def test_empty_gold_is_not_filled_with_distractors_or_a_perfect_score():
    assert protocol.prf([], ["distractor"]) == {
        "tp": 0,
        "returned": 1,
        "relevant": 0,
        "precision": 0.0,
        "recall": None,
        "f1": None,
    }
    assert protocol.prf([], [])["precision"] is None
    missed = protocol.prf(["gold"], [])
    assert missed["precision"] is None and missed["recall"] == 0 and missed["f1"] == 0


def test_stage_metrics_distinguish_recall_from_rerank_loss():
    result = {
        "diagnostics": {
            "rankings": [{"candidates": [{"id": "gold"}, {"id": "noise"}]}],
            "fused_candidates": [{"id": "gold"}, {"id": "noise"}],
        },
        "candidates": [{"id": "noise"}],
        "sources": [{"child_ids": ["noise"]}],
    }
    stage = protocol.stage_metrics({"relevant_child_ids": ["gold"]}, result)
    assert stage["raw_recall_union"]["recall"] == 1
    assert stage["fused"]["recall"] == 1
    assert stage["post_rank"]["recall"] == 0
    assert stage["context"]["recall"] == 0


def test_confirmation_is_bound_to_exact_bundle(tmp_path):
    bundle = {"snapshot_sha256": "frozen", "dataset_sha256": "human_labels", "plan_sha256": "fair"}
    (tmp_path / "review_bundle.json").write_text(json.dumps(bundle))
    with pytest.raises(FileNotFoundError):
        protocol.require_approval(tmp_path)
    (tmp_path / "human_confirmation.json").write_text(
        json.dumps(
            {
                "reviewer": "user",
                "confirmed_bundle": {**bundle, "snapshot_sha256": "changed"},
                "response_text": "synthetic unit-test confirmation",
            }
        )
    )
    with pytest.raises(ValueError, match="confirmation"):
        protocol.require_approval(tmp_path)


async def test_version_guard_rejects_mutation_before_any_model_request(tmp_path, monkeypatch):
    state = {"document_revision": 1, "child_ids": ["old"]}
    (tmp_path / "snapshot.json").write_text(
        json.dumps(
            {
                "state": {"kb_id": "test-kb"},
                "snapshot_sha256": protocol.digest(state),
            }
        )
    )

    async def changed(_):
        return {"document_revision": 2, "child_ids": ["new"]}

    monkeypatch.setattr(protocol, "snapshot", changed)
    with pytest.raises(ValueError, match="version changed"):
        await protocol.verify(tmp_path)


async def test_completed_run_is_rejected_before_any_dataset_write(tmp_path, monkeypatch):
    (tmp_path / "formal").mkdir()

    async def unexpected_write(**kwargs):
        pytest.fail("A completed evaluation must not create another dataset")

    monkeypatch.setattr(protocol.EvaluationDataset, "create", unexpected_write)
    with pytest.raises(FileExistsError, match="Formal results already exist"):
        await protocol.run(tmp_path, None)
