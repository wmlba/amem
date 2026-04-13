"""Ollama embedding provider with retry, circuit breaker, and graceful degradation."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import httpx

from .base import EmbeddingProvider
from amem.config import OllamaConfig


class CircuitBreaker:
    """Simple circuit breaker: after N consecutive failures, open for cooldown_seconds."""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at > self.cooldown_seconds:
            # Half-open: allow one attempt
            self._opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self):
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class OllamaEmbedding(EmbeddingProvider):
    """Embeddings via local Ollama server with production resilience.

    Features:
    - Retry with exponential backoff (3 attempts)
    - Circuit breaker (5 failures → 60s cooldown)
    - Connection pooling
    - Graceful error reporting
    """

    def __init__(self, config: OllamaConfig | None = None):
        self._config = config or OllamaConfig()
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=60.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._dim = self._config.embedding_dim
        self._circuit = CircuitBreaker()
        self._max_retries = 3
        self._retry_delays = [1.0, 2.0, 4.0]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_available(self) -> bool:
        return not self._circuit.is_open

    async def embed(self, text: str) -> np.ndarray:
        if self._circuit.is_open:
            raise ConnectionError(
                f"Ollama circuit breaker open — "
                f"service unavailable, retry after {self._circuit.cooldown_seconds}s cooldown"
            )

        last_error = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(
                    "/api/embed",
                    json={"model": self._config.model, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()
                vec = np.array(data["embeddings"][0], dtype=np.float32)
                if self._dim != len(vec):
                    self._dim = len(vec)
                self._circuit.record_success()
                return vec
            except (httpx.HTTPError, httpx.ConnectError, KeyError) as e:
                last_error = e
                self._circuit.record_failure()
                if attempt < self._max_retries - 1 and not self._circuit.is_open:
                    await asyncio.sleep(self._retry_delays[attempt])

        raise ConnectionError(
            f"Ollama embedding failed after {self._max_retries} retries: {last_error}"
        )

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if self._circuit.is_open:
            raise ConnectionError("Ollama circuit breaker open")

        last_error = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(
                    "/api/embed",
                    json={"model": self._config.model, "input": texts},
                )
                resp.raise_for_status()
                data = resp.json()
                vecs = [np.array(e, dtype=np.float32) for e in data["embeddings"]]
                if vecs and self._dim != len(vecs[0]):
                    self._dim = len(vecs[0])
                self._circuit.record_success()
                return vecs
            except (httpx.HTTPError, httpx.ConnectError, KeyError) as e:
                last_error = e
                self._circuit.record_failure()
                if attempt < self._max_retries - 1 and not self._circuit.is_open:
                    await asyncio.sleep(self._retry_delays[attempt])

        raise ConnectionError(
            f"Ollama batch embedding failed after {self._max_retries} retries: {last_error}"
        )

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and has the configured model."""
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return self._config.model.split(":")[0] in models
        except (httpx.HTTPError, httpx.ConnectError):
            return False

    async def close(self):
        await self._client.aclose()
