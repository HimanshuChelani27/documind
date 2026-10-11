from sqlmodel import Session

from app.db import engine
from app.eval.metrics import retrieval_metrics
from app.ingest.indexer import get_index
from app.retrieval.strategies import STRATEGIES, retrieve


def test_hybrid_rerank_beats_single_retrievers_on_gold(client, gold):
    with Session(engine) as session:
        index = get_index(session, "library")
    assert len(index) > 30
    table = {}
    for strategy in STRATEGIES:
        recalls, mrrs = [], []
        for g in gold:
            hits = retrieve(index, g["question"], strategy, 5).hits
            m = retrieval_metrics(
                [(h.filename, h.page) for h in hits], {(g["doc"], p) for p in g["gold_pages"]}, 5
            )
            recalls.append(m["recall"])
            mrrs.append(m["mrr"])
        table[strategy] = (sum(recalls) / len(gold), sum(mrrs) / len(gold))
    print("\nstrategy        recall@5  MRR")
    for s, (r, m) in table.items():
        print(f"{s:<15} {r:.3f}     {m:.3f}")
    assert table["hybrid_rerank"][0] >= table["dense"][0]
    assert table["hybrid_rerank"][0] >= table["bm25"][0]


def test_doc_filter_and_stages(client):
    with Session(engine) as session:
        index = get_index(session, "library")
    manual = next(c.doc_id for c in index.chunks if c.filename.startswith("aquaflow"))
    r = retrieve(index, "What does error code E03 mean?", "hybrid_rerank", 3, doc_ids=[manual])
    assert {h.doc_id for h in r.hits} == {manual}
    assert set(r.stages) == {"dense", "bm25", "fused", "reranked"}
    assert r.hits[0].page == 8
