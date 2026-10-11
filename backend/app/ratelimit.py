import threading
import time
from collections import deque

from fastapi import HTTPException, Request, status


class RateLimiter:
    """In-memory sliding-window limiter, keyed by client IP and bucket. Fine for one instance."""

    def __init__(self):
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: float) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > window_seconds:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def client_ip(request: Request) -> str:
    # Render/Vercel put the real client first in X-Forwarded-For.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce(request: Request, bucket: str, limit: int, window_seconds: float, what: str) -> None:
    limiter: RateLimiter = request.app.state.limiter
    if not limiter.allow(f"{bucket}:{client_ip(request)}", limit, window_seconds):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, f"Too many {what}, please wait a little")
