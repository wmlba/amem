"""Entity resolution — the hardest problem in associative memory.

Resolves entity aliases, fuzzy matches, and vector-based entity linking.
"Will's GB10" and "the Blackwell workstation" → same canonical entity.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Optional

import numpy as np


@dataclass
class CanonicalEntity:
    """A canonical entity with all known aliases and metadata."""
    canonical_name: str
    entity_type: str
    aliases: set = field(default_factory=set)
    embedding: Optional[np.ndarray] = None
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mention_count: int = 1
    attributes: dict[str, Any] = field(default_factory=dict)

    def matches_name(self, name: str) -> bool:
        """Check if a name matches this entity (canonical or any alias)."""
        name_lower = name.lower().strip()
        if name_lower == self.canonical_name.lower():
            return True
        return any(a.lower() == name_lower for a in self.aliases)


class EntityResolver:
    """Resolves entities across mentions using fuzzy string matching,
    alias tracking, and optional vector similarity.

    Resolution strategy (in order):
    1. Exact match on canonical name or known alias
    2. Fuzzy string match above threshold (handles typos, abbreviations)
    3. Vector similarity match (if embeddings available — handles "GB10" ↔ "Blackwell")
    4. Rule-based patterns (possessive removal, acronym expansion, etc.)
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.80,
        vector_threshold: float = 0.85,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.vector_threshold = vector_threshold
        self._entities: dict[str, CanonicalEntity] = {}  # key → CanonicalEntity
        self._alias_index: dict[str, str] = {}  # lowercase alias → canonical key
        self._embedding_cache: dict[str, np.ndarray] = {}  # key → embedding

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    def _normalize(self, name: str) -> str:
        """Normalize a name for matching."""
        name = name.strip()
        # Remove possessives: "Will's" → "Will"
        name = re.sub(r"'s$", "", name)
        # Remove articles: "the Blackwell workstation" → "Blackwell workstation"
        name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.IGNORECASE)
        # Collapse whitespace
        name = re.sub(r"\s+", " ", name)
        return name

    def _canonical_key(self, name: str) -> str:
        """Generate canonical key from name."""
        return self._normalize(name).lower()

    def register(
        self,
        name: str,
        entity_type: str = "concept",
        aliases: list[str] | None = None,
        embedding: np.ndarray | None = None,
        attributes: dict | None = None,
    ) -> CanonicalEntity:
        """Register a new canonical entity or return existing match."""
        normalized = self._normalize(name)
        key = self._canonical_key(name)

        # Check if already exists
        existing = self.resolve(name)
        if existing is not None:
            existing.last_seen = datetime.now(timezone.utc)
            existing.mention_count += 1
            if aliases:
                for alias in aliases:
                    existing.aliases.add(alias)
                    self._alias_index[alias.lower()] = self._canonical_key(existing.canonical_name)
            if embedding is not None:
                existing.embedding = embedding
                self._embedding_cache[self._canonical_key(existing.canonical_name)] = embedding
            if attributes:
                existing.attributes.update(attributes)
            return existing

        # Create new
        entity = CanonicalEntity(
            canonical_name=normalized,
            entity_type=entity_type,
            aliases=set(aliases or []),
            embedding=embedding,
            attributes=attributes or {},
        )
        self._entities[key] = entity
        self._alias_index[key] = key
        for alias in entity.aliases:
            self._alias_index[alias.lower()] = key
        if embedding is not None:
            self._embedding_cache[key] = embedding

        return entity

    def resolve(self, name: str) -> CanonicalEntity | None:
        """Resolve a name to its canonical entity.

        Tries: exact → alias → fuzzy → vector → rule-based patterns.
        """
        normalized = self._normalize(name)
        key = self._canonical_key(name)

        # 1. Exact match
        if key in self._entities:
            return self._entities[key]

        # 2. Alias index
        if key in self._alias_index:
            canonical_key = self._alias_index[key]
            return self._entities.get(canonical_key)

        # 3. Fuzzy string matching
        best_match = None
        best_score = 0.0
        for ek, entity in self._entities.items():
            # Check against canonical name
            score = self._fuzzy_score(normalized, entity.canonical_name)
            if score > best_score:
                best_score = score
                best_match = entity

            # Check against all aliases
            for alias in entity.aliases:
                score = self._fuzzy_score(normalized, alias)
                if score > best_score:
                    best_score = score
                    best_match = entity

        if best_score >= self.fuzzy_threshold and best_match is not None:
            # Auto-register as alias for future lookups
            self._alias_index[key] = self._canonical_key(best_match.canonical_name)
            best_match.aliases.add(name)
            return best_match

        # 4. Vector similarity (if embeddings available)
        if self._embedding_cache:
            vec_match = self._vector_resolve(name)
            if vec_match is not None:
                self._alias_index[key] = self._canonical_key(vec_match.canonical_name)
                vec_match.aliases.add(name)
                return vec_match

        # 5. Rule-based patterns
        rule_match = self._rule_based_resolve(normalized)
        if rule_match is not None:
            self._alias_index[key] = self._canonical_key(rule_match.canonical_name)
            rule_match.aliases.add(name)
            return rule_match

        return None

    def resolve_or_create(
        self,
        name: str,
        entity_type: str = "concept",
        embedding: np.ndarray | None = None,
    ) -> CanonicalEntity:
        """Resolve to existing entity or create new canonical entity."""
        existing = self.resolve(name)
        if existing is not None:
            existing.last_seen = datetime.now(timezone.utc)
            existing.mention_count += 1
            if embedding is not None and existing.embedding is None:
                existing.embedding = embedding
                self._embedding_cache[self._canonical_key(existing.canonical_name)] = embedding
            return existing
        return self.register(name, entity_type, embedding=embedding)

    def add_alias(self, canonical_name: str, alias: str):
        """Explicitly add an alias for a canonical entity."""
        key = self._canonical_key(canonical_name)
        if key in self._entities:
            self._entities[key].aliases.add(alias)
            # Store both raw lowercase and normalized form for lookup
            self._alias_index[alias.lower()] = key
            normalized_alias = self._canonical_key(alias)
            if normalized_alias != alias.lower():
                self._alias_index[normalized_alias] = key

    def merge(self, name_a: str, name_b: str) -> CanonicalEntity | None:
        """Merge two entities — name_a becomes canonical, name_b becomes alias."""
        entity_a = self.resolve(name_a)
        entity_b = self.resolve(name_b)

        if entity_a is None or entity_b is None:
            return None
        if entity_a is entity_b:
            return entity_a

        # Merge B into A
        entity_a.aliases.add(entity_b.canonical_name)
        entity_a.aliases.update(entity_b.aliases)
        entity_a.mention_count += entity_b.mention_count
        entity_a.attributes.update(entity_b.attributes)
        if entity_b.embedding is not None and entity_a.embedding is None:
            entity_a.embedding = entity_b.embedding

        # Update alias index
        key_b = self._canonical_key(entity_b.canonical_name)
        key_a = self._canonical_key(entity_a.canonical_name)
        for alias in entity_b.aliases | {entity_b.canonical_name}:
            self._alias_index[alias.lower()] = key_a

        # Remove B
        self._entities.pop(key_b, None)
        self._embedding_cache.pop(key_b, None)

        return entity_a

    def _fuzzy_score(self, a: str, b: str) -> float:
        """Compute fuzzy string similarity score."""
        a_lower = a.lower()
        b_lower = b.lower()

        # Exact match
        if a_lower == b_lower:
            return 1.0

        # Containment check (one is substring of other)
        if a_lower in b_lower or b_lower in a_lower:
            shorter = min(len(a_lower), len(b_lower))
            longer = max(len(a_lower), len(b_lower))
            return shorter / longer * 0.95

        # SequenceMatcher
        return SequenceMatcher(None, a_lower, b_lower).ratio()

    def _vector_resolve(self, name: str) -> CanonicalEntity | None:
        """Resolve using embedding similarity."""
        # Need an embedding for the query name
        query_key = self._canonical_key(name)
        if query_key not in self._embedding_cache:
            return None

        query_vec = self._embedding_cache[query_key]
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None
        query_vec = query_vec / query_norm

        best_match = None
        best_sim = 0.0

        for key, embedding in self._embedding_cache.items():
            if key == query_key:
                continue
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            sim = float(query_vec @ (embedding / norm))
            if sim > best_sim:
                best_sim = sim
                best_match = self._entities.get(key)

        if best_sim >= self.vector_threshold and best_match is not None:
            return best_match
        return None

    def _rule_based_resolve(self, name: str) -> CanonicalEntity | None:
        """Rule-based resolution for common patterns."""
        name_lower = name.lower()

        for key, entity in self._entities.items():
            canonical_lower = entity.canonical_name.lower()

            # Acronym: "ML" matches "Machine Learning"
            if len(name) <= 5 and name.isupper():
                words = entity.canonical_name.split()
                if len(words) >= 2:
                    acronym = "".join(w[0].upper() for w in words if w[0].isupper() or w[0].isalpha())
                    if name.upper() == acronym:
                        return entity

            # Partial match: "Pipeline" matches "ML Pipeline"
            words = canonical_lower.split()
            if len(words) > 1 and name_lower in words:
                return entity

        return None

    def get_all_entities(self) -> list[CanonicalEntity]:
        return list(self._entities.values())

    def to_dict(self) -> list[dict]:
        """Serialize for persistence."""
        result = []
        for entity in self._entities.values():
            d = {
                "canonical_name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "aliases": list(entity.aliases),
                "first_seen": entity.first_seen.isoformat(),
                "last_seen": entity.last_seen.isoformat(),
                "mention_count": entity.mention_count,
                "attributes": entity.attributes,
            }
            result.append(d)
        return result

    @classmethod
    def from_dict(cls, data: list[dict]) -> "EntityResolver":
        """Deserialize from persistence."""
        resolver = cls()
        for d in data:
            entity = CanonicalEntity(
                canonical_name=d["canonical_name"],
                entity_type=d["entity_type"],
                aliases=set(d.get("aliases", [])),
                first_seen=datetime.fromisoformat(d["first_seen"]),
                last_seen=datetime.fromisoformat(d["last_seen"]),
                mention_count=d.get("mention_count", 1),
                attributes=d.get("attributes", {}),
            )
            key = resolver._canonical_key(entity.canonical_name)
            resolver._entities[key] = entity
            resolver._alias_index[key] = key
            for alias in entity.aliases:
                resolver._alias_index[alias.lower()] = key
        return resolver
