import numpy as np

from app.ingest.embed import embed_queries
from app.ingest.indexer import CollectionIndex

Hits = list[tuple[int, float]]  # (position in the collection index, score)


def search(
    index: CollectionIndex, query: str, allowed: set[int] | None, k: int, query_vec: np.ndarray | None = None
) -> Hits:
    if not len(index):
        return []
    q = query_vec if query_vec is not None else embed_queries([query])
    # Exact inner-product search over everything; filter afterwards (demo-scale collections).
    scores, ids = index.dense.search(q.reshape(1, -1), index.dense.ntotal)
    out = [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]
    if allowed is not None:
        out = [h for h in out if h[0] in allowed]
    return out[:k]
