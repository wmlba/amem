"""Importance scoring for episodic chunks.

Scores chunks on information density, novelty, and user signal
to determine initial confidence. Supports both single and batch scoring.

Batch mode reuses novelty scores from dedup (no redundant search).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from amem.episodic.vector_index import AssociativeIndex

# Compiled patterns (single compilation, reused across calls)
_FACT_PATTERNS = [
    re.compile(r'\b(?:my name is|I am|I work|I lead|I manage|I use|I prefer|I live)\b', re.IGNORECASE),
    re.compile(r'\b(?:our team|our project|our company|our product)\b', re.IGNORECASE),
    re.compile(r'\b(?:always|never|must|important|critical|key|remember)\b', re.IGNORECASE),
]

_LOW_VALUE_PATTERNS = [
    re.compile(r'\b(?:hmm|uh|ok|okay|sure|thanks|thank you|got it|yes|no|yep|nope)\b', re.IGNORECASE),
    re.compile(r'\b(?:hello|hi|hey|bye|goodbye|see you)\b', re.IGNORECASE),
]

_ENTITY_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')


def _text_signals(text: str) -> tuple[float, float, float]:
    """Extract density, fact_signal, and anti_signal from text in a single pass.

    Returns (density_score, fact_signal, anti_signal) all in [0, 1].
    """
    words = text.split()
    word_count = max(len(words), 1)

    entities = _ENTITY_PATTERN.findall(text)
    entity_density = min(1.0, len(entities) / word_count * 5)

    fact_hits = sum(1 for p in _FACT_PATTERNS if p.search(text))
    fact_signal = min(1.0, fact_hits / 3.0)

    low_hits = sum(1 for p in _LOW_VALUE_PATTERNS if p.search(text))
    anti_signal = min(1.0, low_hits / 2.0)

    length_signal = min(1.0, word_count / 15.0)

    density_score = entity_density * 0.6 + fact_signal * 0.4

    return density_score, length_signal, anti_signal


def score_importance(
    text: str,
    embedding: np.ndarray | None = None,
    existing_index: "AssociativeIndex | None" = None,
) -> float:
    """Score importance of a single chunk. Legacy API."""
    density, length, anti = _text_signals(text)

    novelty = 0.5
    if embedding is not None and existing_index is not None and existing_index.count > 0:
        results = existing_index.search(embedding, top_k=1, temporal_weight=0.0, reinforcement_weight=0.0)
        if results:
            novelty = max(0.0, 1.0 - results[0].raw_similarity)

    raw = density * 0.35 + novelty * 0.30 + length * 0.15 + (1.0 - anti) * 0.20
    return max(0.1, min(1.0, raw))


def score_importance_batch(
    texts: list[str],
    novelty_scores: list[float],
) -> list[float]:
    """Batch importance scoring — reuses novelty scores from dedup pass.

    No redundant index searches. Text signals computed once per chunk.
    """
    results = []
    for text, novelty in zip(texts, novelty_scores):
        density, length, anti = _text_signals(text)
        raw = density * 0.35 + novelty * 0.30 + length * 0.15 + (1.0 - anti) * 0.20
        results.append(max(0.1, min(1.0, raw)))
    return results
