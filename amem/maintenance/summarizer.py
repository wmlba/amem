"""Session Summarizer — compress sessions into high-density knowledge.

When a session ends, instead of dumping raw text into episodic,
compress it into a structured summary that captures:
- Key facts discussed
- Decisions made
- Action items
- New information learned about the user

Uses the SAME embedding model (no LLM needed) via extractive summarization:
pick the most informative sentences based on embedding centrality.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from amem.embeddings.base import EmbeddingProvider


async def summarize_session(
    texts: list[str],
    embedder: "EmbeddingProvider",
    max_sentences: int = 8,
) -> str:
    """Extractive summarization using embedding centrality.

    No LLM needed. Uses the same embedding model as everything else.

    Algorithm:
    1. Split all texts into sentences
    2. Embed each sentence
    3. Compute centroid of all sentence embeddings
    4. Rank sentences by cosine similarity to centroid
    5. Pick top-k most central sentences (they represent the core content)
    6. Return in original order for coherence
    """
    if not texts:
        return ""

    # Split into sentences
    full_text = " ".join(texts)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]  # Skip tiny fragments

    if not sentences:
        return full_text[:500]

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Embed all sentences
    embeddings = await embedder.embed_batch(sentences)
    emb_matrix = np.array(embeddings)

    # Normalize
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    emb_matrix = emb_matrix / norms

    # Compute centroid
    centroid = emb_matrix.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    # Rank by centrality (cosine sim to centroid)
    scores = emb_matrix @ centroid

    # Pick top-k, preserve original order
    top_indices = np.argsort(scores)[-max_sentences:]
    top_indices_sorted = sorted(top_indices)

    summary_sentences = [sentences[i] for i in top_indices_sorted]
    return " ".join(summary_sentences)


def summarize_session_simple(texts: list[str], max_chars: int = 500) -> str:
    """Fallback: simple extractive summary without embeddings.

    Picks sentences with the most named entities and specific details.
    """
    if not texts:
        return ""

    full_text = " ".join(texts)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return full_text[:max_chars]

    # Score each sentence by information density
    scored = []
    for s in sentences:
        # Count: proper nouns, numbers, specific terms
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', s))
        numbers = len(re.findall(r'\d+', s))
        length_score = min(1.0, len(s.split()) / 20)
        score = proper_nouns * 2 + numbers * 3 + length_score
        scored.append((score, s))

    # Sort by score, take top sentences
    scored.sort(key=lambda x: x[0], reverse=True)
    summary = []
    total_chars = 0
    for score, sent in scored:
        if total_chars + len(sent) > max_chars:
            break
        summary.append(sent)
        total_chars += len(sent)

    return " ".join(summary)
