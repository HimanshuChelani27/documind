"""Runs a dataset through several retrieval strategies, optionally generating and grading
answers, and stores per-item results plus an aggregate summary."""

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from statistics import mean

from sqlmodel import Session, col, select

from app.answer import llm
from app.answer.service import answer_once
from app.db import engine
from app.eval import judge
from app.eval.metrics import retrieval_metrics
from app.ingest.embed import embed_queries
from app.ingest.indexer import get_index
from app.memory import trim
from app.models import LIBRARY, EvalDataset, EvalItem, EvalResult, EvalRun
from app.retrieval.strategies import retrieve
from app.sse import Channel

logger = logging.getLogger("documind.eval")
ANSWER_ITEMS_LIMIT = 12  # answer generation + judging is capped to keep LLM usage modest
GOLD_ID = "gold"


def seed_gold(session: Session, gold_path) -> None:
    if session.get(EvalDataset, GOLD_ID) or not gold_path.is_file():
        return
    session.add(EvalDataset(id=GOLD_ID, name="Sample library gold set", collection=LIBRARY, builtin=True))
    for line in gold_path.read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            session.add(
                EvalItem(
                    id=g["id"],
                    dataset_id=GOLD_ID,
                    question=g["question"],
                    answer=g.get("answer"),
                    doc=g["doc"],
                    gold_pages=g["gold_pages"],
                    type=g.get("type"),
                )
            )
    session.commit()


def summarise(results: list[EvalResult], items: dict[str, EvalItem]) -> dict:
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    by_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_strategy[r.strategy].append(r.metrics)
        by_type[r.strategy][items[r.item_id].type or "other"].append(r.metrics)

    def avg(rows: list[dict]) -> dict:
        keys = {k for row in rows for k, v in row.items() if isinstance(v, int | float)}
        return {k: round(mean(row[k] for row in rows if k in row), 4) for k in sorted(keys)}

    return {
        "strategies": {s: avg(rows) for s, rows in by_strategy.items()},
        "by_type": {s: {t: avg(rows) for t, rows in types.items()} for s, types in by_type.items()},
    }


async def run_eval(run_id: str, channel: Channel | None = None) -> None:
    async def emit(event: dict, final: bool = False):
        if channel:
            await channel.publish(event, final)

    with Session(engine) as session:
        run = session.get(EvalRun, run_id)
        assert run is not None
        try:
            items = session.exec(select(EvalItem).where(EvalItem.dataset_id == run.dataset_id)).all()
            index = await asyncio.to_thread(get_index, session, run.collection)
            provider = llm.resolve_provider() if run.with_answers else llm.EXTRACTIVE
            total = len(items) * len(run.strategies)
            done = 0
            results: list[EvalResult] = []
            for n, item in enumerate(items):
                gold = {(item.doc, p) for p in item.gold_pages}
                t0 = time.perf_counter()
                qvec = (await asyncio.to_thread(embed_queries, [item.question]))[0]
                embed_ms = (time.perf_counter() - t0) * 1000  # charged to strategies that use it
                for strategy in run.strategies:
                    r = await asyncio.to_thread(retrieve, index, item.question, strategy, run.k, None, qvec)
                    metrics = retrieval_metrics([(h.filename, h.page) for h in r.hits], gold, run.k)
                    uses_dense = strategy != "bm25"
                    metrics["latency_ms"] = round(sum(r.timings_ms.values()) + embed_ms * uses_dense, 1)
                    answer = None
                    if run.with_answers and n < ANSWER_ITEMS_LIMIT:
                        state = await answer_once(item.question, r.hits, provider)
                        answer = state.text
                        if state.mode == "live":
                            try:
                                metrics |= await judge.judge_metrics(provider, item.question, answer, r.hits)
                            except (*llm.provider_errors(), ValueError) as e:
                                logger.warning("Judge failed: %s", e)
                        metrics |= await asyncio.to_thread(judge.proxy_metrics, item.question, answer, r.hits)
                    res = EvalResult(
                        run_id=run.id,
                        item_id=item.id,
                        strategy=strategy,
                        metrics=metrics,
                        answer=answer,
                        retrieved=[
                            {
                                "chunk_id": h.chunk_id,
                                "filename": h.filename,
                                "page": h.page,
                                "score": h.score,
                                "relevant": (h.filename, h.page) in gold,
                            }
                            for h in r.hits
                        ],
                    )
                    session.add(res)
                    results.append(res)
                    done += 1
                    await emit(
                        {
                            "type": "progress",
                            "done": done,
                            "total": total,
                            "item_id": item.id,
                            "strategy": strategy,
                            "metrics": metrics,
                        }
                    )
            run.summary = summarise(results, {i.id: i for i in items})
            run.status = "done"
            run.mode = "live" if run.with_answers and provider.live else "extractive"
            run.judge = f"{provider.name}/{provider.model}" if run.with_answers and provider.live else None
        except Exception as e:
            logger.exception("Eval run %s failed", run_id)
            run.status, run.error = "failed", str(e)[:300]
        trim()
        run.finished_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        await emit(
            {"type": "done", "status": run.status, "summary": run.summary, "error": run.error}, final=True
        )


async def ensure_baseline() -> None:
    """A shared retrieval-only run on the gold set, so the Evaluate page has data on first visit."""
    with Session(engine) as session:
        if not session.get(EvalDataset, GOLD_ID):
            return
        exists = session.exec(
            select(EvalRun).where(
                EvalRun.dataset_id == GOLD_ID, col(EvalRun.owner).is_(None), EvalRun.status == "done"
            )
        ).first()
        if exists:
            return
        n = len(session.exec(select(EvalItem).where(EvalItem.dataset_id == GOLD_ID)).all())
        run = EvalRun(
            dataset_id=GOLD_ID,
            collection=LIBRARY,
            owner=None,
            items=n,
            strategies=["dense", "bm25", "hybrid", "hybrid_rerank"],
            k=5,
        )
        session.add(run)
        session.commit()
        run_id = run.id
    await run_eval(run_id)
