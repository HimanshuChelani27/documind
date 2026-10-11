from app.ingest.indexer import CollectionIndex, tokenize

Hits = list[tuple[int, float]]


def search(index: CollectionIndex, query: str, allowed: set[int] | None, k: int) -> Hits:
    if index.bm25 is None or not len(index):
        return []
    tokens = tokenize([query])
    if not tokens.vocab:
        return []
    docs, scores = index.bm25.retrieve(tokens, k=len(index), show_progress=False)
    out = [(int(i), float(s)) for i, s in zip(docs[0], scores[0], strict=True) if s > 0]
    if allowed is not None:
        out = [h for h in out if h[0] in allowed]
    return out[:k]
