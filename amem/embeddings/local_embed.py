"""Local embedding provider using sentence-transformers.

Runs entirely offline. No API calls, no network, no API key.
Uses HuggingFace sentence-transformers models.

Install: pip install sentence-transformers
"""

from __future__ import annotations

import numpy as np

from .base import EmbeddingProvider


class LocalEmbedding(EmbeddingProvider):
    """Embeddings via local sentence-transformers models.

    Completely offline. No network, no API key, no external dependency at runtime.
    Downloads model on first use, then runs from cache.

    Good models:
    - all-MiniLM-L6-v2: 384 dims, fast, decent quality
    - all-mpnet-base-v2: 768 dims, better quality
    - BAAI/bge-small-en-v1.5: 384 dims, very good
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self._model_name)
            # Probe dimension
            test = self._model.encode(["test"])
            self._dim = test.shape[1]

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dim or 384

    async def embed(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        vec = self._model.encode([text], show_progress_bar=False)
        return vec[0].astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self._ensure_loaded()
        vecs = self._model.encode(texts, show_progress_bar=False)
        return [v.astype(np.float32) for v in vecs]

    async def close(self):
        pass  # No cleanup needed for local models
