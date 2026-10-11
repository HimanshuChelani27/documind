import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from app.answer.routes import router as chat_router
from app.config import get_settings
from app.db import engine, init_db
from app.documents.routes import router as documents_router
from app.eval.routes import router as eval_router
from app.eval.runner import ensure_baseline, seed_gold
from app.ingest.embed import warm_up
from app.ingest.pipeline import seed_library
from app.models import Document, EvalRun
from app.ratelimit import RateLimiter
from app.retrieval.routes import router as retrieval_router
from app.sse import prune
from app.workspaces import cleanup_loop

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("documind")


def _recover(session: Session) -> None:
    """Work interrupted by a restart can't resume; mark it failed so the UI doesn't wait forever."""
    for doc in session.exec(select(Document).where(Document.status == "processing")).all():
        doc.status, doc.error = "failed", "Interrupted by a server restart; please upload again"
        session.add(doc)
    for run in session.exec(select(EvalRun).where(EvalRun.status == "running")).all():
        run.status, run.error = "failed", "Interrupted by a server restart"
        session.add(run)
    session.commit()


async def _prune_loop(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(300)
        prune(app.state.jobs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    app.state.limiter = RateLimiter()
    app.state.jobs = {}
    app.state.tasks = set()
    await asyncio.to_thread(warm_up)
    with Session(engine) as session:
        _recover(session)
        if settings.seed_library:
            added = await asyncio.to_thread(seed_library, session)
            if added:
                logger.info("Sample library ready (%d documents ingested)", added)
            seed_gold(session, settings.samples_dir / "eval" / "gold.jsonl")
    background = [asyncio.create_task(cleanup_loop()), asyncio.create_task(_prune_loop(app))]
    if settings.seed_library:
        background.append(asyncio.create_task(ensure_baseline()))
    yield
    for task in [*background, *app.state.tasks]:
        task.cancel()
    for task in [*background, *app.state.tasks]:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Hybrid RAG with built-in evaluation: dense + BM25 + RRF + cross-encoder rerank, "
        "page-level citations, and side-by-side retrieval metrics.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Workspace"],
    )

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "version": settings.version}

    app.include_router(chat_router)
    app.include_router(retrieval_router)
    app.include_router(documents_router)
    app.include_router(eval_router)
    return app


app = create_app()
