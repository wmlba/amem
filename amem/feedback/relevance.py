"""Relevance feedback loop — the system learns what's useful.

After an LLM generates a response using retrieved memory context,
this module compares which retrieved chunks were actually used vs ignored.
Used chunks get reinforced; ignored chunks get demoted.

Over time, this trains the retrieval system to surface what matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amem.episodic.temporal_index import TemporalAssociativeIndex


@dataclass
class FeedbackSignal:
    """Feedback for a single retrieved chunk."""
    chunk_id: str
    chunk_text: str
    was_used: bool
    overlap_score: float  # 0.0 (not used) to 1.0 (heavily used)


class RelevanceFeedback:
    """Tracks which retrieved chunks get used in LLM responses.

    Two modes:
    1. Token overlap: compare response tokens against chunk tokens
    2. Explicit: caller marks chunks as used/unused

    Applied as reinforcement (used) or demotion (unused) to the index.
    """

    def __init__(
        self,
        reinforce_amount: int = 2,
        demote_factor: float = 0.95,
        min_overlap_to_count_as_used: float = 0.15,
    ):
        self.reinforce_amount = reinforce_amount
        self.demote_factor = demote_factor
        self.min_overlap = min_overlap_to_count_as_used
        # Track feedback history for analytics
        self._history: list[dict] = []

    def compute_overlap(
        self,
        retrieved_chunks: list[dict],
        response_text: str,
    ) -> list[FeedbackSignal]:
        """Compute token overlap between retrieved chunks and LLM response.

        Uses word-level Jaccard overlap as a lightweight proxy for "was this
        chunk useful in generating the response?"
        """
        if not response_text or not retrieved_chunks:
            return []

        response_tokens = self._tokenize(response_text)

        signals = []
        for chunk in retrieved_chunks:
            chunk_text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", "")
            if not chunk_text:
                continue

            chunk_tokens = self._tokenize(chunk_text)
            overlap = self._jaccard(chunk_tokens, response_tokens)

            signals.append(FeedbackSignal(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                was_used=overlap >= self.min_overlap,
                overlap_score=overlap,
            ))

        return signals

    def apply_feedback(
        self,
        signals: list[FeedbackSignal],
        index: "TemporalAssociativeIndex",
    ) -> dict:
        """Apply feedback signals to the index.

        Used chunks: reinforced (access_count += reinforce_amount)
        Unused chunks: confidence slightly reduced (× demote_factor)
        """
        reinforced = 0
        demoted = 0

        for signal in signals:
            if signal.was_used:
                index.reinforce(signal.chunk_id, self.reinforce_amount)
                reinforced += 1
            else:
                # Gentle demotion: reduce confidence slightly
                for shard in (index._hot, index._warm, index._cold):
                    meta = shard.get_meta(signal.chunk_id)
                    if meta:
                        new_conf = meta.confidence * self.demote_factor
                        shard.update_confidence(signal.chunk_id, new_conf)
                        demoted += 1
                        break

        result = {"reinforced": reinforced, "demoted": demoted, "total": len(signals)}
        self._history.append(result)
        return result

    def get_feedback_rate(self) -> dict:
        """Get aggregate feedback statistics."""
        if not self._history:
            return {"total_rounds": 0, "avg_used_ratio": 0.0}

        total_reinforced = sum(h["reinforced"] for h in self._history)
        total_signals = sum(h["total"] for h in self._history)

        return {
            "total_rounds": len(self._history),
            "total_signals": total_signals,
            "total_reinforced": total_reinforced,
            "avg_used_ratio": total_reinforced / max(total_signals, 1),
        }

    def _tokenize(self, text: str) -> set[str]:
        """Lightweight word tokenization. Lowercase, strip punctuation."""
        return set(re.findall(r'\b\w{3,}\b', text.lower()))

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0
