"""Anthropic/Claude embedding provider via Voyager models.

Uses the Anthropic API's embedding endpoint (voyage-3, voyage-code-3, etc.)
Also works with any Anthropic-compatible proxy.
"""

from __future__ import annotations

import os

import numpy as np
import httpx

from .base import EmbeddingProvider


class AnthropicEmbedding(EmbeddingProvider):
    """Embeddings via Anthropic's Voyage embedding models.

    Models:
    - voyage-3: general purpose, 1024 dims
    - voyage-3-lite: faster, 512 dims
    - voyage-code-3: optimized for code, 1024 dims
    """

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        base_url: str = "https://api.voyageai.com/v1",
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._dim: int | None = None
        self._client = httpx.AsyncClient(timeout=60.0)

    @property
    def dimension(self) -> int:
        return self._dim or 1024

    async def embed(self, text: str) -> np.ndarray:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        resp = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": texts, "input_type": "document"},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        vecs = [np.array(e["embedding"], dtype=np.float32) for e in data["data"]]
        if vecs and self._dim is None:
            self._dim = len(vecs[0])
        return vecs

    async def close(self):
        await self._client.aclose()
