import asyncio
import logging
import re
from collections.abc import AsyncIterator

from app.answer import llm
from app.answer.citations import cited_numbers, strip_invalid
from app.answer.extractive import extractive_answer
from app.answer.prompt import SYSTEM, build_user_prompt
from app.config import get_settings
from app.retrieval.strategies import Hit

logger = logging.getLogger("documind.answer")


class AnswerState:
    """Collects what the stream produced, for the final `citations`/`done` events."""

    def __init__(self, provider: llm.Provider):
        self.provider = provider
        self.text = ""
        self.mode = "live" if provider.live else "extractive"
        self.usage = llm.Usage()
        self.fallback_reason: str | None = None


async def stream_answer(question: str, hits: list[Hit], state: AnswerState) -> AsyncIterator[str]:
    """Yields answer deltas. Falls back to an extractive answer if the provider fails before
    producing any text."""
    if not hits:
        state.mode = "extractive"
        state.text = "There are no documents to search yet. Upload a PDF or pick the sample library."
        yield state.text
        return
    if state.provider.live:
        try:
            async for delta in llm.stream_completion(
                state.provider,
                SYSTEM,
                build_user_prompt(question, hits),
                get_settings().llm_max_tokens,
                state.usage,
            ):
                state.text += delta
                yield delta
            return
        except llm.provider_errors() as e:
            logger.warning("LLM provider failed: %s", e)
            state.fallback_reason = (
                f"{state.provider.name} failed ({type(e).__name__}); used extractive answer"
            )
            if state.text:  # failed mid-answer: keep what we have
                return
            state.mode = "extractive"
    text, _ = await asyncio.to_thread(extractive_answer, question, hits)
    state.text = text
    # Emit sentence by sentence so the UI renders it the same way as a live stream.
    for part in re.findall(r".+?\]\s*|.+$", text):
        yield part


def citations(state: AnswerState, hits: list[Hit]) -> tuple[str, list[dict]]:
    text = strip_invalid(state.text, len(hits))
    used = set(cited_numbers(text, len(hits)))
    sources = [
        {
            "n": n,
            "cited": n in used,
            **h.model_dump(
                include={"chunk_id", "doc_id", "doc_title", "filename", "page", "heading", "text", "score"}
            ),
        }
        for n, h in enumerate(hits, start=1)
    ]
    return text, sources


async def answer_once(question: str, hits: list[Hit], provider: llm.Provider) -> AnswerState:
    state = AnswerState(provider)
    async for _ in stream_answer(question, hits, state):
        pass
    state.text = strip_invalid(state.text, len(hits))
    return state
