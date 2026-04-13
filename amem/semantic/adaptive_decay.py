"""Adaptive decay rates based on fact type.

Facts about identity decay slowly; ephemeral facts decay quickly.
Uses predicate and entity type to auto-classify decay tier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class DecayTiers:
    """Configurable decay rates per fact tier."""
    identity: float = 0.001     # Name, role, core identity — very slow
    professional: float = 0.005  # Job, team, company — slow
    project: float = 0.01       # Current projects, tools — medium
    ephemeral: float = 0.05     # Preferences, opinions, recent events — fast

    def get_rate(self, tier: str) -> float:
        return getattr(self, tier, self.project)


# Predicate → tier mapping
PREDICATE_TIERS = {
    # Identity
    "is_a": "identity",
    "named": "identity",
    "born_in": "identity",

    # Professional
    "works_at": "professional",
    "employed_by": "professional",
    "leads": "professional",
    "manages": "professional",
    "reports_to": "professional",
    "role_is": "professional",

    # Project
    "works_on": "project",
    "uses": "project",
    "researches": "project",
    "created": "project",
    "part_of": "project",

    # Ephemeral
    "prefers": "ephemeral",
    "likes": "ephemeral",
    "located_in": "ephemeral",
    "mentioned": "ephemeral",
    "discussed": "ephemeral",
}

# Entity type → default tier (if predicate doesn't have specific mapping)
ENTITY_TYPE_TIERS = {
    "person": "professional",
    "org": "professional",
    "tool": "project",
    "project": "project",
    "concept": "project",
    "location": "ephemeral",
}


class AdaptiveDecay:
    """Compute decay rates adaptively based on fact classification."""

    def __init__(self, tiers: DecayTiers | None = None, min_confidence: float = 0.05):
        self.tiers = tiers or DecayTiers()
        self.min_confidence = min_confidence

    def classify_tier(
        self,
        predicate: str,
        subject_type: str = "",
        object_type: str = "",
    ) -> str:
        """Classify a fact into a decay tier."""
        pred_lower = predicate.lower()

        # Check predicate mapping first (most specific)
        if pred_lower in PREDICATE_TIERS:
            return PREDICATE_TIERS[pred_lower]

        # Fall back to entity type
        for etype in (subject_type.lower(), object_type.lower()):
            if etype in ENTITY_TYPE_TIERS:
                return ENTITY_TYPE_TIERS[etype]

        return "project"  # default tier

    def get_rate(self, predicate: str, subject_type: str = "", object_type: str = "") -> float:
        """Get the decay rate for a specific fact."""
        tier = self.classify_tier(predicate, subject_type, object_type)
        return self.tiers.get_rate(tier)

    def compute(
        self,
        base_confidence: float,
        last_seen: datetime,
        predicate: str,
        subject_type: str = "",
        object_type: str = "",
        now: datetime | None = None,
    ) -> float:
        """Compute decayed confidence with adaptive rate."""
        if now is None:
            now = datetime.now(timezone.utc)
        rate = self.get_rate(predicate, subject_type, object_type)
        days = (now - last_seen).total_seconds() / 86400.0
        return max(0.0, base_confidence * math.exp(-rate * days))

    def should_prune(self, confidence: float) -> bool:
        return confidence < self.min_confidence

    def reinforce(self, current: float, boost: float = 0.2) -> float:
        return min(1.0, current + boost * (1.0 - current))
