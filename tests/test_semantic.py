"""Tests for the semantic graph and entity extraction."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from amem.semantic.graph import SemanticGraph, Entity, Relation
from amem.semantic.extractor import EntityExtractor
from amem.semantic.decay import ConfidenceDecay


class TestConfidenceDecay:
    def test_decay_reduces_confidence(self):
        decay = ConfidenceDecay(decay_lambda=0.01)
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=100)
        result = decay.compute(1.0, old, now)
        assert result < 1.0
        assert result > 0.0

    def test_recent_confidence_stays_high(self):
        decay = ConfidenceDecay(decay_lambda=0.01)
        now = datetime.now(timezone.utc)
        result = decay.compute(1.0, now, now)
        assert result == pytest.approx(1.0)

    def test_prune_threshold(self):
        decay = ConfidenceDecay(min_confidence=0.1)
        assert decay.should_prune(0.05)
        assert not decay.should_prune(0.2)

    def test_reinforce(self):
        decay = ConfidenceDecay()
        result = decay.reinforce(0.5)
        assert result > 0.5
        assert result <= 1.0


class TestEntityExtractor:
    def setup_method(self):
        self.extractor = EntityExtractor()

    def test_extract_proper_nouns(self):
        result = self.extractor.extract("Alice works with Bob on the project.")
        names = [e.name for e in result.entities]
        assert "Alice" in names
        assert "Bob" in names

    def test_extract_tools(self):
        result = self.extractor.extract("We use Python and Docker for deployment.")
        names = [e.name.lower() for e in result.entities]
        assert "python" in names
        assert "docker" in names

    def test_extract_relations(self):
        result = self.extractor.extract("Alice works on the ML pipeline.")
        assert len(result.relations) > 0
        rel = result.relations[0]
        assert rel.subject == "Alice"
        assert rel.predicate == "works_on"

    def test_skip_stopwords(self):
        result = self.extractor.extract("The quick brown fox.")
        names = [e.name for e in result.entities]
        assert "The" not in names


class TestSemanticGraph:
    def setup_method(self):
        self.graph = SemanticGraph()

    def test_add_entity(self):
        self.graph.add_entity(Entity(name="Alice", entity_type="person"))
        assert self.graph.entity_count == 1

    def test_add_relation(self):
        self.graph.add_relation(Relation(
            subject="Alice", predicate="works_on", object="ML Pipeline",
        ))
        assert self.graph.relation_count == 1
        # Entities auto-created
        assert self.graph.entity_count == 2

    def test_reinforce_relation(self):
        self.graph.add_relation(Relation(
            subject="Alice", predicate="works_on", object="ML Pipeline",
            confidence=0.5,
        ))
        # Add same relation again → should reinforce
        self.graph.add_relation(Relation(
            subject="Alice", predicate="works_on", object="ML Pipeline",
            confidence=0.5,
        ))
        assert self.graph.relation_count == 1  # not duplicated

    def test_query_returns_facts(self):
        self.graph.add_relation(Relation(
            subject="Alice", predicate="works_on", object="ML Pipeline",
        ))
        self.graph.add_relation(Relation(
            subject="ML Pipeline", predicate="uses", object="Python",
        ))
        facts = self.graph.query(["Alice"], max_depth=2)
        assert len(facts) >= 1
        subjects = [f["subject"] for f in facts]
        assert "Alice" in subjects

    def test_ingest_text(self):
        result = self.graph.ingest_text("Bob leads the infrastructure team using Kubernetes.")
        assert len(result.entities) > 0
        assert self.graph.entity_count > 0

    def test_save_load(self):
        self.graph.add_relation(Relation(
            subject="Alice", predicate="works_on", object="Project",
        ))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph"
            self.graph.save(path)

            new_graph = SemanticGraph()
            new_graph.load(path)
            assert new_graph.entity_count == 2
            assert new_graph.relation_count == 1

    def test_decay_pass_prunes(self):
        decay = ConfidenceDecay(decay_lambda=0.1, min_confidence=0.1)
        graph = SemanticGraph(decay=decay)

        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        graph.add_relation(Relation(
            subject="Old", predicate="knows", object="Fact",
            confidence=0.5,
            last_seen=old_time,
        ))
        graph.decay_pass()
        # Should have been pruned
        assert graph.relation_count == 0
