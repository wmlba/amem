"""Contradiction detection and resolution for the semantic graph.

Detects conflicting facts, applies temporal reasoning, and manages
fact lifecycle (active → superseded → retracted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by a newer fact
    RETRACTED = "retracted"   # explicitly marked as wrong
    CONFLICTED = "conflicted" # unresolved contradiction


@dataclass
class Contradiction:
    """A detected contradiction between two facts."""
    fact_a: dict  # {subject, predicate, object, timestamp, confidence}
    fact_b: dict
    contradiction_type: str  # "direct", "temporal", "negation"
    resolution: str = "unresolved"  # "newer_wins", "higher_confidence", "user_resolved", "unresolved"
    resolved_at: datetime | None = None
    winner: str = ""  # "a" or "b" or ""

    def to_dict(self) -> dict:
        return {
            "fact_a": self.fact_a,
            "fact_b": self.fact_b,
            "contradiction_type": self.contradiction_type,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "winner": self.winner,
        }


# Predicates that define exclusive relationships (can only have one value at a time)
EXCLUSIVE_PREDICATES = {
    "works_at", "employed_by", "lives_in", "located_in",
    "is_a", "role_is", "reports_to", "leads",
    "uses",  # can use multiple tools, but "X uses Y" where Y is a specific role is exclusive
}

# Predicates that imply temporal succession (new one replaces old)
TEMPORAL_PREDICATES = {
    "works_at", "employed_by", "lives_in", "located_in",
    "role_is", "reports_to",
}

# Negation patterns — if predicate B is seen, it negates predicate A
NEGATION_PAIRS = {
    "works_on": "stopped_working_on",
    "works_at": "left",
    "leads": "stepped_down_from",
    "uses": "stopped_using",
    "researches": "abandoned",
}


class ContradictionDetector:
    """Detects and resolves contradictions in the knowledge graph."""

    def __init__(self):
        self._contradictions: list[Contradiction] = []
        self._fact_status: dict[str, FactStatus] = {}  # fact_key → status
        self._superseded_by: dict[str, str] = {}  # old_fact_key → new_fact_key

    @property
    def contradiction_count(self) -> int:
        return len(self._contradictions)

    @property
    def unresolved_count(self) -> int:
        return sum(1 for c in self._contradictions if c.resolution == "unresolved")

    def _fact_key(self, subject: str, predicate: str, obj: str) -> str:
        return f"{subject.lower()}|{predicate.lower()}|{obj.lower()}"

    def _relation_key(self, subject: str, predicate: str) -> str:
        """Key for exclusive relationship checking (subject + predicate)."""
        return f"{subject.lower()}|{predicate.lower()}"

    def check_and_resolve(
        self,
        new_fact: dict,
        existing_facts: list[dict],
    ) -> list[Contradiction]:
        """Check a new fact against existing facts for contradictions.

        Args:
            new_fact: {subject, predicate, object, timestamp, confidence, ...}
            existing_facts: List of existing facts from the graph

        Returns:
            List of detected contradictions (may be auto-resolved)
        """
        contradictions = []

        new_subj = new_fact.get("subject", "").lower()
        new_pred = new_fact.get("predicate", "").lower()
        new_obj = new_fact.get("object", "").lower()
        new_time = self._parse_time(new_fact.get("timestamp") or new_fact.get("last_seen"))

        for existing in existing_facts:
            ex_subj = existing.get("subject", "").lower()
            ex_pred = existing.get("predicate", "").lower()
            ex_obj = existing.get("object", "").lower()
            ex_time = self._parse_time(existing.get("timestamp") or existing.get("last_seen"))

            # Skip if different subject
            if ex_subj != new_subj:
                continue

            # 1. Direct contradiction: same subject + predicate, different object, exclusive predicate
            if ex_pred == new_pred and ex_obj != new_obj and new_pred in EXCLUSIVE_PREDICATES:
                contradiction = Contradiction(
                    fact_a=existing,
                    fact_b=new_fact,
                    contradiction_type="direct",
                )
                self._auto_resolve(contradiction, ex_time, new_time, existing, new_fact)
                contradictions.append(contradiction)
                self._contradictions.append(contradiction)

            # 2. Temporal supersession: same subject, same predicate in TEMPORAL_PREDICATES
            elif ex_pred == new_pred and ex_obj != new_obj and new_pred in TEMPORAL_PREDICATES:
                contradiction = Contradiction(
                    fact_a=existing,
                    fact_b=new_fact,
                    contradiction_type="temporal",
                )
                self._auto_resolve(contradiction, ex_time, new_time, existing, new_fact)
                contradictions.append(contradiction)
                self._contradictions.append(contradiction)

            # 3. Negation: new predicate negates existing predicate
            elif new_pred in NEGATION_PAIRS.values():
                for pos_pred, neg_pred in NEGATION_PAIRS.items():
                    if neg_pred == new_pred and ex_pred == pos_pred:
                        contradiction = Contradiction(
                            fact_a=existing,
                            fact_b=new_fact,
                            contradiction_type="negation",
                        )
                        # Negation always wins — newer statement explicitly negates
                        contradiction.resolution = "newer_wins"
                        contradiction.winner = "b"
                        contradiction.resolved_at = datetime.now(timezone.utc)

                        old_key = self._fact_key(ex_subj, ex_pred, ex_obj)
                        self._fact_status[old_key] = FactStatus.SUPERSEDED
                        contradictions.append(contradiction)
                        self._contradictions.append(contradiction)

        return contradictions

    def _auto_resolve(
        self,
        contradiction: Contradiction,
        time_a: datetime | None,
        time_b: datetime | None,
        fact_a: dict,
        fact_b: dict,
    ):
        """Auto-resolve based on temporal precedence and confidence."""
        conf_a = fact_a.get("confidence", 0.5)
        conf_b = fact_b.get("confidence", 0.5)

        # Strategy 1: If both have timestamps, newer wins
        if time_a is not None and time_b is not None:
            if time_b > time_a:
                contradiction.resolution = "newer_wins"
                contradiction.winner = "b"
                old_key = self._fact_key(
                    fact_a.get("subject", ""),
                    fact_a.get("predicate", ""),
                    fact_a.get("object", ""),
                )
                self._fact_status[old_key] = FactStatus.SUPERSEDED
            elif time_a > time_b:
                contradiction.resolution = "newer_wins"
                contradiction.winner = "a"
                old_key = self._fact_key(
                    fact_b.get("subject", ""),
                    fact_b.get("predicate", ""),
                    fact_b.get("object", ""),
                )
                self._fact_status[old_key] = FactStatus.SUPERSEDED
            else:
                # Same timestamp — use confidence
                self._resolve_by_confidence(contradiction, conf_a, conf_b, fact_a, fact_b)
        else:
            # No timestamps — use confidence
            self._resolve_by_confidence(contradiction, conf_a, conf_b, fact_a, fact_b)

        if contradiction.resolution != "unresolved":
            contradiction.resolved_at = datetime.now(timezone.utc)

    def _resolve_by_confidence(
        self,
        contradiction: Contradiction,
        conf_a: float,
        conf_b: float,
        fact_a: dict,
        fact_b: dict,
    ):
        """Resolve by confidence difference if significant enough."""
        if abs(conf_a - conf_b) > 0.2:
            if conf_a > conf_b:
                contradiction.resolution = "higher_confidence"
                contradiction.winner = "a"
                old_key = self._fact_key(
                    fact_b.get("subject", ""),
                    fact_b.get("predicate", ""),
                    fact_b.get("object", ""),
                )
                self._fact_status[old_key] = FactStatus.CONFLICTED
            else:
                contradiction.resolution = "higher_confidence"
                contradiction.winner = "b"
                old_key = self._fact_key(
                    fact_a.get("subject", ""),
                    fact_a.get("predicate", ""),
                    fact_a.get("object", ""),
                )
                self._fact_status[old_key] = FactStatus.CONFLICTED

    def retract_fact(self, subject: str, predicate: str, obj: str):
        """Explicitly retract a fact (user-driven)."""
        key = self._fact_key(subject, predicate, obj)
        self._fact_status[key] = FactStatus.RETRACTED

    def get_status(self, subject: str, predicate: str, obj: str) -> FactStatus:
        """Get the status of a fact."""
        key = self._fact_key(subject, predicate, obj)
        return self._fact_status.get(key, FactStatus.ACTIVE)

    def is_active(self, subject: str, predicate: str, obj: str) -> bool:
        return self.get_status(subject, predicate, obj) == FactStatus.ACTIVE

    def get_contradictions(self, subject: str | None = None) -> list[Contradiction]:
        """Get all contradictions, optionally filtered by subject."""
        if subject is None:
            return list(self._contradictions)
        subject_lower = subject.lower()
        return [
            c for c in self._contradictions
            if c.fact_a.get("subject", "").lower() == subject_lower
            or c.fact_b.get("subject", "").lower() == subject_lower
        ]

    def get_unresolved(self) -> list[Contradiction]:
        return [c for c in self._contradictions if c.resolution == "unresolved"]

    def _parse_time(self, val) -> datetime | None:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except (ValueError, TypeError):
                return None
        return None

    def to_dict(self) -> dict:
        return {
            "contradictions": [c.to_dict() for c in self._contradictions],
            "fact_status": {k: v.value for k, v in self._fact_status.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContradictionDetector":
        detector = cls()
        for cs in data.get("contradictions", []):
            c = Contradiction(
                fact_a=cs["fact_a"],
                fact_b=cs["fact_b"],
                contradiction_type=cs["contradiction_type"],
                resolution=cs.get("resolution", "unresolved"),
                winner=cs.get("winner", ""),
            )
            if cs.get("resolved_at"):
                c.resolved_at = datetime.fromisoformat(cs["resolved_at"])
            detector._contradictions.append(c)
        for key, status in data.get("fact_status", {}).items():
            detector._fact_status[key] = FactStatus(status)
        return detector
