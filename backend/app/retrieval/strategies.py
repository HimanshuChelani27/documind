"""Retrieval strategies. Each returns the final ranking plus every intermediate stage, so the UI
can show exactly how an answer's sources were found."""

import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from app.ingest.indexer import CollectionIndex
from app.retrieval import dense, fusion, rerank, sparse

Strategy = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
STRATEGIES: dict[str, str] = {
    "dense": "Dense (FAISS, bge-small)",
    "bm25": "BM25 (sparse keywords)",
    "hybrid": "Hybrid (dense + BM25, RRF)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}
CANDIDATES = 20  # first-stage depth fed into fusion and the reranker


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    filename: str
    page: int
    heading: str | None
    text: str
    score: float
    rank: int


class StageHit(BaseModel):
    chunk_id: str
    page: int
    filename: str
    score: float


@dataclass
class Retrieval:
    strategy: str
    hits: list[Hit]
    stages: dict[str, list[StageHit]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "hits": [h.model_dump() for h in self.hits],
            "stages": {k: [s.model_dump() for s in v] for k, v in self.stages.items()},
            "timings_ms": self.timings_ms,
        }


def _hit(index: CollectionIndex, pos: int, score: float, rank: int) -> Hit:
    c = index.chunks[pos]
    return Hit(
        chunk_id=c.id,
        doc_id=c.doc_id,
        doc_title=c.doc_title,
        filename=c.filename,
        page=c.page,
        heading=c.heading,
        text=c.text,
        score=round(score, 4),
        rank=rank,
    )


def _stage(index: CollectionIndex, hits, limit: int = 10) -> list[StageHit]:
    return [
        StageHit(
            chunk_id=index.chunks[p].id,
            page=index.chunks[p].page,
            filename=index.chunks[p].filename,
            score=round(s, 4),
        )
        for p, s in hits[:limit]
    ]


def retrieve(
    index: CollectionIndex,
    query: str,
    strategy: str = "hybrid_rerank",
    k: int = 5,
    doc_ids: list[str] | None = None,
    query_vec=None,
) -> Retrieval:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}")
    allowed = None
    if doc_ids:
        wanted = set(doc_ids)
        allowed = {i for i, c in enumerate(index.chunks) if c.doc_id in wanted}
    timings: dict[str, float] = {}
    stages: dict[str, list[StageHit]] = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        timings[name] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    depth = max(k, CANDIDATES)
    d_hits = s_hits = None
    if strategy in ("dense", "hybrid", "hybrid_rerank"):
        d_hits = timed("dense", lambda: dense.search(index, query, allowed, depth, query_vec))
        stages["dense"] = _stage(index, d_hits)
    if strategy in ("bm25", "hybrid", "hybrid_rerank"):
        s_hits = timed("bm25", lambda: sparse.search(index, query, allowed, depth))
        stages["bm25"] = _stage(index, s_hits)

    if strategy == "dense":
        final = d_hits
    elif strategy == "bm25":
        final = s_hits
    else:
        fused = timed("fusion", lambda: fusion.rrf([d_hits, s_hits])[:depth])
        stages["fused"] = _stage(index, fused)
        final = fused
        if strategy == "hybrid_rerank":
            final = timed("rerank", lambda: rerank.rerank(index, query, fused))
            stages["reranked"] = _stage(index, final)

    hits = [_hit(index, p, s, r) for r, (p, s) in enumerate(final[:k], start=1)]
    return Retrieval(strategy, hits, stages, timings)
