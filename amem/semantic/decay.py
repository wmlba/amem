"""Confidence decay and reinforcement model for knowledge graph edges."""

from __future__ import annotations

import math
from datetime import datetime, timezone


class ConfidenceDecay:
    """Exponential decay with reinforcement for edge confidence."""

    def __init__(self, decay_lambda: float = 0.005, min_confidence: float = 0.05):
        self.decay_lambda = decay_lambda
        self.min_confidence = min_confidence

    def compute(self, base_confidence: float, last_seen: datetime, now: datetime | None = None) -> float:
        """Compute decayed confidence."""
        if now is None:
            now = datetime.now(timezone.utc)
        days = (now - last_seen).total_seconds() / 86400.0
        decayed = base_confidence * math.exp(-self.decay_lambda * days)
        return max(decayed, 0.0)

    def should_prune(self, confidence: float) -> bool:
        """Whether this confidence level should be pruned."""
        return confidence < self.min_confidence

    def reinforce(self, current_confidence: float, boost: float = 0.2) -> float:
        """Reinforce confidence when an entity/relation is re-mentioned."""
        return min(1.0, current_confidence + boost * (1.0 - current_confidence))
