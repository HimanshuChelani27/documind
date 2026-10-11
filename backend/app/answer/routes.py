import asyncio
import time

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.answer import llm
from app.answer.service import AnswerState, citations, stream_answer
from app.config import get_settings
from app.db import get_session
from app.ingest.indexer import get_index
from app.memory import trim
from app.ratelimit import enforce
from app.retrieval.strategies import Strategy, retrieve
from app.sse import sse, stream_response
from app.workspaces import Scope, Workspace, get_workspace

router = APIRouter(tags=["chat"])


class ChatIn(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    scope: Scope = "library"
    doc_ids: list[str] | None = None
    strategy: Strategy = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=8)


@router.get("/llm/provider")
def provider() -> dict:
    p = llm.resolve_provider()
    return {"provider": p.name, "model": p.model, "mode": "live" if p.live else "extractive"}


@router.post("/chat")
async def chat(
    body: ChatIn,
    request: Request,
    ws: Workspace = Depends(get_workspace),
    session: Session = Depends(get_session),
):
    """Retrieves, then streams the answer as Server-Sent Events:
    `retrieval` (every stage's hits) → `token`* → `citations` → `done`."""
    s = get_settings()
    enforce(request, "chat", s.chat_per_minute, 60, "questions")
    collection = ws.resolve(body.scope)
    index = await asyncio.to_thread(get_index, session, collection)

    async def events():
        t0 = time.perf_counter()
        r = await asyncio.to_thread(retrieve, index, body.question, body.strategy, body.top_k, body.doc_ids)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
        yield sse({"type": "retrieval", **r.as_dict()})

        state = AnswerState(llm.resolve_provider())
        t1 = time.perf_counter()
        first_token_ms = None
        async for delta in stream_answer(body.question, r.hits, state):
            if first_token_ms is None:
                first_token_ms = round((time.perf_counter() - t1) * 1000, 1)
            yield sse({"type": "token", "text": delta})
        text, sources = citations(state, r.hits)
        yield sse({"type": "citations", "answer": text, "sources": sources})
        yield sse(
            {
                "type": "done",
                "mode": state.mode,
                "provider": state.provider.name if state.mode == "live" else "extractive",
                "model": state.provider.model if state.mode == "live" else None,
                "fallback_reason": state.fallback_reason,
                "usage": {
                    "input_tokens": state.usage.input_tokens,
                    "output_tokens": state.usage.output_tokens,
                },
                "timings_ms": {
                    **r.timings_ms,
                    "retrieval_total": retrieval_ms,
                    "first_token": first_token_ms,
                    "answer": round((time.perf_counter() - t1) * 1000, 1),
                    "total": round((time.perf_counter() - t0) * 1000, 1),
                },
            }
        )
        trim()

    return stream_response(events())
