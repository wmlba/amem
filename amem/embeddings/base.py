"""Abstract base for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Interface for text → vector embedding."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a vector."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts. Default: sequential calls."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...
