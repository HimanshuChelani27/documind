"""A replayable event channel streamed as Server-Sent Events (ingestion jobs, eval runs).

Producers may run in worker threads; `publish_threadsafe` hands events to the event loop."""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator

from fastapi import Request
from fastapi.responses import StreamingResponse

KEEPALIVE_SECONDS = 15


class Channel:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self.events: list[dict] = []
        self.done = False
        self.finished_at: float | None = None
        self.changed = asyncio.Condition()
        self.loop = loop or asyncio.get_running_loop()

    async def publish(self, event: dict, final: bool = False) -> None:
        async with self.changed:
            self.events.append(event)
            self.done = self.done or final
            if final:
                self.finished_at = time.monotonic()
            self.changed.notify_all()

    def publish_threadsafe(self, event: dict, final: bool = False) -> None:
        asyncio.run_coroutine_threadsafe(self.publish(event, final), self.loop)


def sse(event: dict) -> str:
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, default=str)}\n\n"


async def _stream(channel: Channel, request: Request) -> AsyncIterator[str]:
    sent = 0
    while True:
        async with channel.changed:
            if sent >= len(channel.events) and not channel.done:
                with contextlib.suppress(TimeoutError):  # wake up to send a keep-alive
                    await asyncio.wait_for(channel.changed.wait(), KEEPALIVE_SECONDS)
            pending, finished = channel.events[sent:], channel.done
        for event in pending:
            yield sse(event)
        sent += len(pending)
        if finished and sent >= len(channel.events):
            return
        if not pending:
            if await request.is_disconnected():
                return
            yield ": keep-alive\n\n"


def stream_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        gen, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


def channel_response(channel: Channel, request: Request) -> StreamingResponse:
    return stream_response(_stream(channel, request))


def prune(jobs: dict[str, Channel], older_than_seconds: float = 900) -> None:
    now = time.monotonic()
    for key in [k for k, c in jobs.items() if c.finished_at and now - c.finished_at > older_than_seconds]:
        jobs.pop(key, None)
