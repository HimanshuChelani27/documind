"""Private upload workspaces. No accounts: the browser generates a random id, keeps it in local
storage and sends it as `X-Workspace`. Uploads go to collection `ws_<id>` and expire after a TTL.
The shared sample library ("library") is read-only through the API."""

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import Header, HTTPException, Query, status
from sqlmodel import Session, col, delete, select

from app.config import get_settings
from app.db import engine
from app.ingest import indexer
from app.models import (
    LIBRARY,
    Chunk,
    Document,
    EvalDataset,
    EvalItem,
    EvalResult,
    EvalRun,
)

logger = logging.getLogger("documind.workspaces")
WORKSPACE_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
Scope = Literal["library", "workspace"]


@dataclass(frozen=True)
class Workspace:
    id: str | None

    @property
    def collection(self) -> str | None:
        return f"ws_{self.id}" if self.id else None

    def collections(self) -> list[str]:
        return [LIBRARY] + ([self.collection] if self.collection else [])

    def resolve(self, scope: Scope) -> str:
        if scope == "library":
            return LIBRARY
        if not self.collection:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "X-Workspace header required for your uploads")
        return self.collection

    def can_see(self, collection: str) -> bool:
        return collection in self.collections()


def _parse(raw: str | None) -> Workspace:
    if raw is None or raw == "":
        return Workspace(None)
    if not WORKSPACE_RE.fullmatch(raw):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid workspace id")
    return Workspace(raw)


def get_workspace(
    x_workspace: str | None = Header(default=None),
    workspace: str | None = Query(default=None, include_in_schema=False),  # for <a href> downloads
) -> Workspace:
    return _parse(x_workspace or workspace)


def require_workspace(ws: Workspace) -> str:
    if not ws.collection:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "X-Workspace header required")
    return ws.collection


def delete_document(session: Session, doc: Document, rebuild: bool = True) -> None:
    session.exec(delete(Chunk).where(Chunk.doc_id == doc.id))  # type: ignore[arg-type]
    session.delete(doc)
    session.commit()
    indexer.vectors_path(doc.collection, doc.id).unlink(missing_ok=True)
    (get_settings().upload_dir / doc.collection / f"{doc.id}.pdf").unlink(missing_ok=True)
    if rebuild:
        indexer.rebuild(session, doc.collection)


def purge_collection(session: Session, collection: str) -> None:
    for doc in session.exec(select(Document).where(Document.collection == collection)).all():
        delete_document(session, doc, rebuild=False)
    for ds in session.exec(select(EvalDataset).where(EvalDataset.collection == collection)).all():
        session.exec(delete(EvalItem).where(EvalItem.dataset_id == ds.id))  # type: ignore[arg-type]
        session.delete(ds)
    for run in session.exec(select(EvalRun).where(EvalRun.collection == collection)).all():
        session.exec(delete(EvalResult).where(EvalResult.run_id == run.id))  # type: ignore[arg-type]
        session.delete(run)
    session.commit()
    indexer.drop_collection(collection)
    shutil.rmtree(get_settings().upload_dir / collection, ignore_errors=True)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)  # SQLite returns naive UTC


def cleanup_expired(session: Session, now: datetime | None = None) -> int:
    """Deletes uploaded documents older than the TTL; empty workspaces are purged entirely
    (eval datasets and runs too). Returns the number of documents removed."""
    cutoff = (now or datetime.now(UTC)) - timedelta(hours=get_settings().workspace_ttl_hours)
    touched: set[str] = set()
    removed = 0
    for doc in session.exec(select(Document).where(Document.collection != LIBRARY)).all():
        if _aware(doc.created_at) < cutoff:
            touched.add(doc.collection)
            delete_document(session, doc, rebuild=False)
            removed += 1
    # Custom eval datasets and runs belong to a workspace and expire on the same clock.
    for ds in session.exec(select(EvalDataset).where(col(EvalDataset.owner).is_not(None))).all():
        if _aware(ds.created_at) < cutoff:
            session.exec(delete(EvalItem).where(EvalItem.dataset_id == ds.id))  # type: ignore[arg-type]
            session.delete(ds)
    for run in session.exec(select(EvalRun).where(col(EvalRun.owner).is_not(None))).all():
        if _aware(run.created_at) < cutoff:
            session.exec(delete(EvalResult).where(EvalResult.run_id == run.id))  # type: ignore[arg-type]
            session.delete(run)
    session.commit()
    for collection in touched:
        if session.exec(select(Document).where(Document.collection == collection)).first():
            indexer.rebuild(session, collection)
        else:
            purge_collection(session, collection)
    if removed:
        logger.info("Expired %d uploaded document(s)", removed)
    return removed


async def cleanup_loop(interval_seconds: float = 600) -> None:
    while True:
        try:
            with Session(engine) as session:
                await asyncio.to_thread(cleanup_expired, session)
        except Exception:
            logger.exception("Workspace cleanup failed")
        await asyncio.sleep(interval_seconds)
