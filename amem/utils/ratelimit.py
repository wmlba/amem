"""Simple in-memory rate limiter using token bucket algorithm."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request


class RateLimiter:
    """Token bucket rate limiter.

    Each client gets `capacity` tokens, refilled at `rate` tokens/second.
    Each request consumes 1 token. When empty → 429 Too Many Requests.
    """

    def __init__(self, rate: float = 10.0, capacity: int = 50):
        self.rate = rate            # tokens per second
        self.capacity = capacity    # max tokens
        self._buckets: dict[str, tuple[float, float]] = {}  # key → (tokens, last_time)

    def _get_key(self, request: Request) -> str:
        """Rate limit by API key or IP."""
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"key:{api_key}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    def check(self, request: Request):
        """Check rate limit. Raises 429 if exceeded."""
        key = self._get_key(request)
        now = time.monotonic()

        if key in self._buckets:
            tokens, last_time = self._buckets[key]
            # Refill tokens based on elapsed time
            elapsed = now - last_time
            tokens = min(self.capacity, tokens + elapsed * self.rate)
        else:
            tokens = float(self.capacity)

        if tokens < 1.0:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down.",
                headers={"Retry-After": str(int(1.0 / self.rate))},
            )

        self._buckets[key] = (tokens - 1.0, now)
