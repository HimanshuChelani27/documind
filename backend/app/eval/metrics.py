"""Retrieval metrics against gold pages. A retrieved chunk is relevant when its (document, page)
is one of the question's gold pages; each gold page counts once, however many chunks hit it."""

import math


def relevance(retrieved: list[tuple[str, int]], gold: set[tuple[str, int]]) -> list[int]:
    """1 for the first chunk that hits each gold page, else 0 (so duplicates don't inflate scores)."""
    seen: set[tuple[str, int]] = set()
    rels = []
    for key in retrieved:
        if key in gold and key not in seen:
            seen.add(key)
            rels.append(1)
        else:
            rels.append(0)
    return rels


def hit_at_k(rels: list[int], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def recall_at_k(rels: list[int], k: int, n_gold: int) -> float:
    return sum(rels[:k]) / n_gold if n_gold else 0.0


def mrr(rels: list[int], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list[int], k: int, n_gold: int) -> float:
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(n_gold, k) + 1))
    return dcg / idcg if idcg else 0.0


def precision_at_k(retrieved: list[tuple[str, int]], gold: set[tuple[str, int]], k: int) -> float:
    """Context precision: share of the top-k chunks that come from a gold page."""
    top = retrieved[:k]
    return sum(1 for key in top if key in gold) / len(top) if top else 0.0


def retrieval_metrics(
    retrieved: list[tuple[str, int]], gold: set[tuple[str, int]], k: int
) -> dict[str, float]:
    rels = relevance(retrieved, gold)
    return {
        "hit": hit_at_k(rels, k),
        "recall": recall_at_k(rels, k, len(gold)),
        "mrr": mrr(rels, k),
        "ndcg": ndcg_at_k(rels, k, len(gold)),
        "context_precision": precision_at_k(retrieved, gold, k),
    }
