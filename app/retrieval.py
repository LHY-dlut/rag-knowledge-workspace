import asyncio
import time
from collections import OrderedDict

from rank_bm25 import BM25Okapi

from app.models import Document, KnowledgeBase, ParentChunk
from app.providers import ModelProvider, tokenize
from app.schemas import Candidate, Citation, MetadataFilter, RetrievalConfig, RetrievalResult
from app.settings import Settings
from app.vector_store import VectorStore


def reciprocal_rank_fusion(
    rankings: list[list[Candidate]], *, k: int = 60, weights: list[float] | None = None
) -> list[Candidate]:
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings) or k <= 0 or any(w < 0 for w in weights):
        raise ValueError("Invalid RRF arguments")
    fused: dict[str, Candidate] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen = set()
        for rank, candidate in enumerate(ranking, 1):
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            if candidate.id not in fused:
                fused[candidate.id] = candidate.model_copy(update={"rrf_score": 0})
            merged = fused[candidate.id]
            merged.rrf_score += weight / (k + rank)
            if candidate.cosine_score is not None:
                merged.cosine_score = max(candidate.cosine_score, merged.cosine_score or -1.0)
            if candidate.bm25_score is not None:
                merged.bm25_score = max(candidate.bm25_score, merged.bm25_score or float("-inf"))
    return sorted(fused.values(), key=lambda c: (-c.rrf_score, c.id))


class AdvancedRetrieverPipeline:
    def __init__(self, store: VectorStore, provider: ModelProvider, settings: Settings):
        self.store, self.provider, self.settings = store, provider, settings
        self.cache: OrderedDict[tuple, tuple[list[Candidate], list[set[str]], BM25Okapi]] = (
            OrderedDict()
        )

    async def _sparse_index(self, key: tuple, owner_id: str, kb_id: str, doc_ids: list[str]):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        corpus = await self.store.corpus(
            owner_id, kb_id, doc_ids, self.provider.fingerprint, self.settings.sparse_max_chunks
        )
        if not corpus:
            return None

        def build():
            tokenized = [tokenize(record.content) or ["__empty__"] for record in corpus]
            return corpus, [set(tokens) for tokens in tokenized], BM25Okapi(tokenized)

        index = await asyncio.to_thread(build)
        self.cache[key] = index
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return index

    async def retrieve(
        self,
        *,
        owner_id: str,
        kb: KnowledgeBase,
        original_query: str,
        queries: list[str],
        hypothetical_document: str = "",
        filters: MetadataFilter | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        config = RetrievalConfig.model_validate(kb.config)
        filters = filters or MetadataFilter()
        docs = await Document.filter(owner_id=owner_id, kb_id=kb.id, status="ready").all()
        if any(d.embedding_fingerprint != self.provider.fingerprint for d in docs):
            raise ValueError("Embedding 模型已改变，请重新索引全部文档后再检索")
        if config.metadata_filter:
            docs = [
                d
                for d in docs
                if (not filters.document_ids or d.id in filters.document_ids)
                and (not filters.file_types or d.file_type in filters.file_types)
                and (not filters.tags or set(filters.tags) <= set(d.tags))
            ]
        doc_ids = sorted(d.id for d in docs)
        document_map = {d.id: d for d in docs}
        if not doc_ids:
            return RetrievalResult(diagnostics={"reason": "no_ready_documents", "queries": queries})
        queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:4] or [
            original_query
        ]
        # HyDE is ONLY a dense document embedding; never feed it into generation evidence.
        dense_texts = queries + (
            [hypothetical_document] if config.hyde and hypothetical_document else []
        )
        vectors_task = asyncio.create_task(self.provider.embed(queries, "query"))
        sparse_task = (
            asyncio.create_task(
                self._sparse_index(
                    (owner_id, kb.id, kb.revision, tuple(doc_ids), self.provider.fingerprint),
                    owner_id,
                    kb.id,
                    doc_ids,
                )
            )
            if config.hybrid or not self.store.postgres
            else None
        )
        try:
            vectors = await vectors_task
            if len(dense_texts) > len(queries):
                vectors += await self.provider.embed([hypothetical_document], "document")
            sparse_index = await sparse_task if sparse_task else None
            dense_rankings = await asyncio.gather(
                *(
                    self.store.dense(
                        owner_id, kb.id, doc_ids, self.provider.fingerprint, v, config.top_k
                    )
                    for v in vectors
                )
            )
        except BaseException:
            for task in (vectors_task, sparse_task):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(t for t in (vectors_task, sparse_task) if t), return_exceptions=True
            )
            raise
        sparse_rankings: list[list[Candidate]] = []
        if config.hybrid and sparse_index:
            corpus, token_sets, index = sparse_index

            def sparse_rank(q: str):
                tokens = tokenize(q)
                scores = index.get_scores(tokens)
                # Okapi IDF can be zero/negative in tiny corpora. A matching token
                # remains a lexical hit; do not confuse score=0 with no match.
                ids = [i for i, terms in enumerate(token_sets) if terms & set(tokens)]
                ids.sort(key=lambda i: (-float(scores[i]), corpus[i].id))
                return [
                    corpus[i].model_copy(update={"bm25_score": float(scores[i])})
                    for i in ids[: config.top_k]
                ]

            sparse_rankings = await asyncio.gather(
                *(asyncio.to_thread(sparse_rank, q) for q in queries)
            )
        rankings = [*dense_rankings, *sparse_rankings]
        recall_elapsed = round((time.perf_counter() - started) * 1000)
        # Query expansion must not multiply one modality's total weight.
        weights = [1 / len(dense_rankings)] * len(dense_rankings)
        weights += [1 / len(sparse_rankings)] * len(sparse_rankings) if sparse_rankings else []
        fused = reciprocal_rank_fusion(rankings, k=config.rrf_k, weights=weights)[
            : config.candidate_k
        ]
        if config.rerank:
            reranked = await self.provider.rerank(original_query, [c.content for c in fused])
            winners = [
                fused[i].model_copy(update={"rerank_score": score})
                for i, score in reranked
                if score >= config.rerank_threshold
            ][: config.rerank_k]
        else:
            # Hybrid RRF scores are ranks, not similarities; this tier has no
            # cosine threshold. Dense-only uses its own cosine threshold.
            winners = (
                fused
                if config.hybrid
                else [
                    c
                    for c in fused
                    if c.cosine_score is not None and c.cosine_score >= config.cosine_threshold
                ]
            )
        parent_ids = list(dict.fromkeys(c.parent_id for c in winners))
        ranking_elapsed = round((time.perf_counter() - started) * 1000)
        parents = {
            p.id: p
            for p in await ParentChunk.filter(
                id__in=parent_ids, doc_id__in=doc_ids, kb_id=kb.id, owner_id=owner_id
            ).all()
        }
        sources: list[Citation] = []
        used, seen = 0, set()
        for c in winners:
            key = c.parent_id if config.parent_retrieval else c.id
            if key in seen or c.parent_id not in parents:
                continue
            parent = parents[c.parent_id]
            text = parent.content if config.parent_retrieval else c.content
            remaining = config.context_chars - used
            if remaining <= 0 or len(sources) >= config.context_k:
                break
            text = text[:remaining]
            seen.add(key)
            used += len(text)
            sources.append(
                Citation(
                    source_id=f"S{len(sources) + 1}",
                    document_id=c.doc_id,
                    parent_id=c.parent_id,
                    filename=document_map[c.doc_id].filename,
                    location=parent.metadata.get("location", "全文")
                    + ("（人工修正）" if parent.edited else ""),
                    content=text,
                    child_ids=[w.id for w in winners if w.parent_id == c.parent_id]
                    if config.parent_retrieval
                    else [c.id],
                    edited=parent.edited,
                    document_revision=document_map[c.doc_id].index_revision,
                )
            )
        # Recheck publication state after I/O: deletion/reindex during retrieval
        # must not make stale BM25 cache entries visible.
        live = dict(
            await Document.filter(id__in=doc_ids, owner_id=owner_id, status="ready").values_list(
                "id", "index_revision"
            )
        )
        sources = [s for s in sources if live.get(s.document_id) == s.document_revision]
        return RetrievalResult(
            candidates=[
                c for c in winners if live.get(c.doc_id) == document_map[c.doc_id].index_revision
            ],
            sources=sources,
            diagnostics={
                "queries": queries,
                "hyde_used": len(dense_texts) > len(queries),
                "dense_hits": [len(r) for r in dense_rankings],
                "sparse_hits": [len(r) for r in sparse_rankings],
                "fused_count": len(fused),
                "rankings": [
                    {
                        "origin": "vector" if index < len(dense_rankings) else "bm25",
                        "query_index": index
                        if index < len(dense_rankings)
                        else index - len(dense_rankings),
                        "candidates": [
                            dict(rank=rank, **c.model_dump(exclude={"embedding"}))
                            for rank, c in enumerate(ranking, 1)
                            if live.get(c.doc_id) == document_map[c.doc_id].index_revision
                        ],
                    }
                    for index, ranking in enumerate(rankings)
                ],
                "fused_candidates": [
                    dict(rank=rank, **c.model_dump(exclude={"embedding"}))
                    for rank, c in enumerate(fused, 1)
                    if live.get(c.doc_id) == document_map[c.doc_id].index_revision
                ],
                "recall_ms": recall_elapsed,
                "fusion_rerank_ms": ranking_elapsed - recall_elapsed,
                "context_chars": sum(len(s.content) for s in sources),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            },
        )
