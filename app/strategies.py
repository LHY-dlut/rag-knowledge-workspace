"""Screenshot-derived presets. Chunking stays a per-KB indexing decision."""

from copy import copy

from app.models import KnowledgeBase
from app.schemas import RetrievalConfig

PRESETS = {
    "dense": {
        "query_rewrite": False,
        "multi_query": False,
        "hyde": False,
        "hybrid": False,
        "rerank": False,
        "top_k": 5,
        "candidate_k": 5,
        "context_k": 5,
        "cosine_threshold": 0.3,
    },
    "hybrid": {
        "query_rewrite": False,
        "multi_query": False,
        "hyde": False,
        "hybrid": True,
        "rerank": False,
        "top_k": 8,
        "candidate_k": 16,
        "context_k": 8,
        "rrf_k": 60,
    },
    "full": {
        "query_rewrite": True,
        "multi_query": True,
        "hyde": False,
        "hybrid": True,
        "rerank": True,
        "top_k": 10,
        "candidate_k": 10,
        "context_k": 4,
        "rerank_k": 4,
        "rrf_k": 60,
        "rerank_threshold": 0.05,
    },
}


def effective_kb(kb: KnowledgeBase, strategy: str = "custom") -> KnowledgeBase:
    result = copy(kb)
    config = RetrievalConfig.model_validate(kb.config)
    if strategy != "custom":
        if strategy not in PRESETS:
            raise ValueError("Unknown retrieval strategy")
        config = RetrievalConfig.model_validate({**config.model_dump(), **PRESETS[strategy]})
    result.config = config.model_dump()
    return result


def expanded_queries(original: str, standalone: str, variants: list[str], enabled: bool):
    # A single question produces the original plus at most three variants.
    # Coreference may add a standalone version inside the same four-query budget.
    return list(dict.fromkeys([original, standalone, *variants]))[:4] if enabled else [standalone]
