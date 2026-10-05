import math

import pytest

from app.agent import citation_check
from app.chunking import ParsedUnit, parent_child_chunks, split_spans
from app.retrieval import reciprocal_rank_fusion
from app.schemas import Candidate, RetrievalConfig
from app.tools import calculate
from app.vector_models import validate_vector


def candidate(id, score):
    return Candidate(
        id=id,
        owner_id="u",
        kb_id="k",
        doc_id="d",
        parent_id="p",
        content=id,
        embedding_fingerprint="demo",
        cosine_score=score,
    )


def test_rrf_uses_rank_not_absolute_score():
    a, b = candidate("a", 999), candidate("b", -999)
    first = reciprocal_rank_fusion([[a, b], [b, a]])
    second = reciprocal_rank_fusion([[a.model_copy(update={"cosine_score": -5}), b], [b, a]])
    assert [(c.id, c.rrf_score) for c in first] == [(c.id, c.rrf_score) for c in second]
    assert first[0].rrf_score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_deduplicates_one_route():
    a = candidate("a", 1)
    result = reciprocal_rank_fusion([[a, a], [a]], weights=[0.5, 0.5])
    assert result[0].rrf_score == pytest.approx(1 / 61)


def test_parent_child_offsets_cover_original():
    text = ("这是完整的一段，不能丢掉字符。\n\n" * 80) + "结尾"
    config = RetrievalConfig(parent_size=600, child_size=200, child_overlap=30)
    parents = parent_child_chunks("doc", [ParsedUnit(text, {"page": 2})], config)
    assert "".join(p.content for p in parents) == text
    assert parents == parent_child_chunks("doc", [ParsedUnit(text, {"page": 2})], config)
    for parent in parents:
        covered = set()
        for _, child, start, end in parent.children:
            assert child == parent.content[start:end]
            assert len(child) <= config.child_size
            covered.update(range(start, end))
        assert covered == set(range(len(parent.content)))
        assert parent.metadata["page"] == 2


def test_long_unpunctuated_paragraph_progresses():
    spans = split_spans("中" * 3000, 280, 40)
    assert spans[0] == (0, 280)
    assert spans[-1][1] == 3000
    assert all(a < b for a, b in spans)


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1)])
def test_invalid_chunk_configuration(size, overlap):
    with pytest.raises(ValueError):
        split_spans("test", size, overlap)


@pytest.mark.parametrize("vector", [[1.0], [0.0] * 1024, [math.nan] * 1024, [math.inf] * 1024])
def test_invalid_vectors(vector):
    with pytest.raises(ValueError):
        validate_vector(vector)


def test_citations_require_existing_id():
    assert citation_check("依据 [S1]", [{"source_id": "S1"}]).passed
    assert not citation_check("依据 [S2]", [{"source_id": "S1"}]).passed
    assert not citation_check("没有引用", [{"source_id": "S1"}]).passed


def test_safe_calculator():
    assert calculate("(2+3)*4") == "20"
    for expression in ("__import__('os').system('x')", "2**100000", "[1][0]", "1/0", "9e99"):
        with pytest.raises((ValueError, ZeroDivisionError)):
            calculate(expression)
