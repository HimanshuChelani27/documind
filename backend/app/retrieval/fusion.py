Hits = list[tuple[int, float]]


def rrf(rankings: list[Hits], k: int = 60) -> Hits:
    """Reciprocal-rank fusion: score(d) = Σ 1 / (k + rank_i(d)), ranks starting at 1.
    Uses only ranks, so dense cosine scores and BM25 scores never need to be calibrated."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (pos, _) in enumerate(ranking, start=1):
            scores[pos] = scores.get(pos, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
