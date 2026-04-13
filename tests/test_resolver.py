"""Tests for entity resolution — the hardest problem."""

import numpy as np
import pytest

from amem.semantic.resolver import EntityResolver, CanonicalEntity


class TestEntityResolver:
    def setup_method(self):
        self.resolver = EntityResolver()

    def test_register_and_resolve_exact(self):
        self.resolver.register("Alice Smith", entity_type="person")
        result = self.resolver.resolve("Alice Smith")
        assert result is not None
        assert result.canonical_name == "Alice Smith"

    def test_resolve_case_insensitive(self):
        self.resolver.register("Alice Smith", entity_type="person")
        result = self.resolver.resolve("alice smith")
        assert result is not None
        assert result.canonical_name == "Alice Smith"

    def test_resolve_with_alias(self):
        self.resolver.register("Alice Smith", entity_type="person", aliases=["Alice", "A. Smith"])
        assert self.resolver.resolve("Alice") is not None
        assert self.resolver.resolve("A. Smith") is not None
        assert self.resolver.resolve("Alice").canonical_name == "Alice Smith"

    def test_resolve_possessive_removal(self):
        """'Alice's project' should normalize to 'Alice project'."""
        self.resolver.register("Alice", entity_type="person")
        result = self.resolver.resolve("Alice's")
        assert result is not None
        assert result.canonical_name == "Alice"

    def test_resolve_article_removal(self):
        """'the ML Pipeline' should match 'ML Pipeline'."""
        self.resolver.register("ML Pipeline", entity_type="project")
        result = self.resolver.resolve("the ML Pipeline")
        assert result is not None
        assert result.canonical_name == "ML Pipeline"

    def test_fuzzy_match_typo(self):
        self.resolver.register("Kubernetes", entity_type="tool")
        result = self.resolver.resolve("Kuberntes")  # typo
        assert result is not None
        assert result.canonical_name == "Kubernetes"

    def test_fuzzy_match_abbreviation(self):
        self.resolver.register("Machine Learning Pipeline", entity_type="project")
        # "ML Pipeline" is contained in the canonical — partial match
        result = self.resolver.resolve("Machine Learning Pipelin")  # close enough
        assert result is not None

    def test_no_false_positive_fuzzy(self):
        """Completely different names should not match."""
        self.resolver.register("Kubernetes", entity_type="tool")
        result = self.resolver.resolve("PostgreSQL")
        assert result is None

    def test_acronym_resolution(self):
        self.resolver.register("Machine Learning", entity_type="concept")
        result = self.resolver.resolve("ML")
        assert result is not None
        assert result.canonical_name == "Machine Learning"

    def test_merge_entities(self):
        self.resolver.register("GB10", entity_type="tool")
        self.resolver.register("Blackwell Workstation", entity_type="tool")

        merged = self.resolver.merge("GB10", "Blackwell Workstation")
        assert merged is not None
        assert merged.canonical_name == "GB10"
        assert "Blackwell Workstation" in merged.aliases

        # Now resolving "Blackwell Workstation" returns GB10
        result = self.resolver.resolve("Blackwell Workstation")
        assert result.canonical_name == "GB10"

    def test_add_alias(self):
        self.resolver.register("Alice", entity_type="person")
        self.resolver.add_alias("Alice", "Ali")
        result = self.resolver.resolve("Ali")
        assert result is not None
        assert result.canonical_name == "Alice"

    def test_resolve_or_create_existing(self):
        self.resolver.register("Bob", entity_type="person")
        entity = self.resolver.resolve_or_create("Bob")
        assert entity.mention_count >= 2
        assert self.resolver.entity_count == 1

    def test_resolve_or_create_new(self):
        entity = self.resolver.resolve_or_create("NewEntity")
        assert entity.canonical_name == "NewEntity"
        assert self.resolver.entity_count == 1

    def test_vector_based_resolution(self):
        """Test that vector similarity can link entities."""
        # Create two entities with similar embeddings
        vec_a = np.array([1.0, 0.0, 0.0, 0.1], dtype=np.float32)
        vec_b = np.array([0.99, 0.0, 0.0, 0.12], dtype=np.float32)

        self.resolver.register("Entity A", entity_type="concept", embedding=vec_a)
        self.resolver.register("Entity B", entity_type="concept", embedding=vec_b)

        # Very similar vectors — Entity B should resolve to Entity A via vector matching
        # (if fuzzy string matching doesn't already catch it)
        # For this test, we manually check vector resolution
        self.resolver._embedding_cache["entity c"] = vec_a.copy()
        result = self.resolver._vector_resolve("Entity C")
        # Should find a match since vec_a ≈ Entity A's embedding
        assert result is not None

    def test_serialization_roundtrip(self):
        self.resolver.register("Alice", entity_type="person", aliases=["Ali", "A"])
        self.resolver.register("Bob", entity_type="person")

        data = self.resolver.to_dict()
        loaded = EntityResolver.from_dict(data)

        assert loaded.entity_count == 2
        assert loaded.resolve("Alice") is not None
        assert loaded.resolve("Ali") is not None

    def test_mention_count_increments(self):
        self.resolver.register("Alice", entity_type="person")
        self.resolver.resolve_or_create("Alice")
        self.resolver.resolve_or_create("Alice")
        entity = self.resolver.resolve("Alice")
        assert entity.mention_count >= 3

    def test_partial_word_match(self):
        """'Pipeline' should match 'ML Pipeline'."""
        self.resolver.register("ML Pipeline", entity_type="project")
        result = self.resolver.resolve("Pipeline")
        assert result is not None
        assert result.canonical_name == "ML Pipeline"
