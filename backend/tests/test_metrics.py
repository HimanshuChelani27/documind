import math

import pytest

from app.answer.citations import cited_numbers, strip_invalid
from app.eval.metrics import retrieval_metrics
from app.retrieval.fusion import rrf


def test_rrf_toy_example():
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    sparse = [(3, 12.0), (1, 9.0), (4, 2.0)]
    fused = rrf([dense, sparse], k=60)
    # doc1: 1/61 + 1/62, doc3: 1/63 + 1/61, doc2: 1/62, doc4: 1/63
    assert [p for p, _ in fused] == [1, 3, 2, 4]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_retrieval_metrics_hand_computed():
    gold = {("a.pdf", 3), ("a.pdf", 7)}
    retrieved = [("a.pdf", 1), ("a.pdf", 3), ("a.pdf", 3), ("b.pdf", 7), ("a.pdf", 7)]
    m = retrieval_metrics(retrieved, gold, k=5)
    assert m["hit"] == 1.0
    assert m["recall"] == 1.0
    assert m["mrr"] == 0.5
    # relevant (deduplicated) at ranks 2 and 5
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg"] == pytest.approx(dcg / idcg)
    assert m["context_precision"] == pytest.approx(3 / 5)

    m3 = retrieval_metrics(retrieved, gold, k=1)
    assert m3 == {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "context_precision": 0.0}


def test_citation_parser_drops_unknown_sources():
    text = "Per diem is USD 85 [2]. Hotels [1, 7] and nothing [9]. Again [2]."
    assert cited_numbers(text, 3) == [2, 1]
    assert strip_invalid(text, 3) == "Per diem is USD 85 [2]. Hotels [1] and nothing . Again [2]."
