"""OpenAI-compatible embedding provider.

Works with:
- OpenAI API (text-embedding-3-small, text-embedding-3-large)
- Azure OpenAI
- Any OpenAI-compatible endpoint (vLLM, llama.cpp server, LiteLLM, Anyscale, Together, etc.)
- OpenRouter
- Local servers exposing /v1/embeddings

Just set base_url and api_key.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import numpy as np
import httpx

from .base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    """Embeddings via any OpenAI-compatible /v1/embeddings endpoint.

    Covers: OpenAI, Azure, vLLM, llama.cpp, LiteLLM, Together, Anyscale,
    OpenRouter, and any server speaking the OpenAI embeddings protocol.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        dimension: int | None = None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._dim = dimension
        self._dim_probed = False
        self._client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    @property
    def dimension(self) -> int:
        return self._dim or 1536  # OpenAI default, updated on first call

    async def embed(self, text: str) -> np.ndarray:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body: dict[str, Any] = {"model": self._model, "input": texts}
        # Some providers support dimension parameter
        if self._dim:
            body["dimensions"] = self._dim

        resp = await self._client.post(
            f"{self._base_url}/embeddings",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        # Sort by index (API may return out of order)
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        vecs = [np.array(e["embedding"], dtype=np.float32) for e in embeddings]

        if vecs and not self._dim_probed:
            self._dim = len(vecs[0])
            self._dim_probed = True

        return vecs

    async def close(self):
        await self._client.aclose()
