import asyncio

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.config import get_settings
from app.db import get_session
from app.ingest.indexer import get_index
from app.ratelimit import enforce
from app.retrieval.strategies import STRATEGIES, Strategy, retrieve
from app.workspaces import Scope, Workspace, get_workspace

router = APIRouter(tags=["retrieval"])


class RetrieveIn(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    scope: Scope = "library"
    doc_ids: list[str] | None = None
    strategy: Strategy = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=20)


@router.get("/strategies")
def strategies() -> list[dict]:
    return [{"id": k, "label": v} for k, v in STRATEGIES.items()]


@router.post("/retrieve")
async def retrieve_only(
    body: RetrieveIn,
    request: Request,
    ws: Workspace = Depends(get_workspace),
    session: Session = Depends(get_session),
) -> dict:
    """Retrieval only, no LLM: every stage's ranking with scores and timings."""
    enforce(request, "chat", get_settings().chat_per_minute, 60, "questions")
    index = await asyncio.to_thread(get_index, session, ws.resolve(body.scope))
    r = await asyncio.to_thread(retrieve, index, body.question, body.strategy, body.top_k, body.doc_ids)
    return r.as_dict()
