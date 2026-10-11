from app.ingest.embed import rerank_scores
from app.ingest.indexer import CollectionIndex

Hits = list[tuple[int, float]]


def rerank(index: CollectionIndex, query: str, candidates: Hits) -> Hits:
    """Cross-encoder rerank: reads query and passage together, so it judges relevance far better
    than either first-stage retriever, but is too slow to run over the whole collection."""
    if not candidates:
        return []
    scores = rerank_scores(query, [index.chunks[pos].search_text for pos, _ in candidates])
    ranked = sorted(zip((p for p, _ in candidates), scores, strict=True), key=lambda x: -x[1])
    return [(p, float(s)) for p, s in ranked]
