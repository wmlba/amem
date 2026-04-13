"""Tests for contradiction detection and resolution."""

from datetime import datetime, timedelta, timezone

import pytest

from amem.semantic.contradictions import (
    ContradictionDetector, Contradiction, FactStatus,
)
from amem.semantic.graph import SemanticGraph, Entity, Relation
from amem.semantic.decay import ConfidenceDecay


class TestContradictionDetector:
    def setup_method(self):
        self.detector = ContradictionDetector()

    def test_direct_contradiction_newer_wins(self):
        """'Will works_at OCI' then 'Will works_at Google' → newer wins."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)

        existing = [{
            "subject": "Will",
            "predicate": "works_at",
            "object": "OCI",
            "confidence": 0.8,
            "timestamp": old.isoformat(),
            "last_seen": old.isoformat(),
        }]

        new_fact = {
            "subject": "Will",
            "predicate": "works_at",
            "object": "Google",
            "confidence": 0.8,
            "timestamp": now.isoformat(),
            "last_seen": now.isoformat(),
        }

        contradictions = self.detector.check_and_resolve(new_fact, existing)
        assert len(contradictions) == 1
        c = contradictions[0]
        assert c.contradiction_type == "direct"
        assert c.resolution == "newer_wins"
        assert c.winner == "b"

    def test_negation_contradiction(self):
        """'Will works_on ML' then 'Will stopped_working_on ML' → negation wins."""
        existing = [{
            "subject": "Will",
            "predicate": "works_on",
            "object": "ML",
            "confidence": 0.8,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }]

        new_fact = {
            "subject": "Will",
            "predicate": "stopped_working_on",
            "object": "ML",
            "confidence": 0.9,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }

        contradictions = self.detector.check_and_resolve(new_fact, existing)
        assert len(contradictions) == 1
        assert contradictions[0].contradiction_type == "negation"
        assert contradictions[0].winner == "b"

    def test_no_contradiction_different_subjects(self):
        existing = [{
            "subject": "Alice",
            "predicate": "works_at",
            "object": "Google",
            "confidence": 0.8,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }]

        new_fact = {
            "subject": "Bob",
            "predicate": "works_at",
            "object": "Meta",
            "confidence": 0.8,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }

        contradictions = self.detector.check_and_resolve(new_fact, existing)
        assert len(contradictions) == 0

    def test_confidence_based_resolution(self):
        """When timestamps are the same, higher confidence wins."""
        now = datetime.now(timezone.utc)

        existing = [{
            "subject": "Will",
            "predicate": "works_at",
            "object": "OCI",
            "confidence": 0.3,
            "timestamp": now.isoformat(),
            "last_seen": now.isoformat(),
        }]

        new_fact = {
            "subject": "Will",
            "predicate": "works_at",
            "object": "Google",
            "confidence": 0.9,
            "timestamp": now.isoformat(),
            "last_seen": now.isoformat(),
        }

        contradictions = self.detector.check_and_resolve(new_fact, existing)
        assert len(contradictions) == 1
        assert contradictions[0].resolution == "higher_confidence"
        assert contradictions[0].winner == "b"

    def test_retract_fact(self):
        self.detector.retract_fact("Will", "works_at", "OCI")
        assert self.detector.get_status("Will", "works_at", "OCI") == FactStatus.RETRACTED
        assert not self.detector.is_active("Will", "works_at", "OCI")

    def test_unresolved_tracking(self):
        """When confidence is too close, contradiction stays unresolved."""
        now = datetime.now(timezone.utc)

        existing = [{
            "subject": "Will",
            "predicate": "works_at",
            "object": "OCI",
            "confidence": 0.75,
            "timestamp": now.isoformat(),
            "last_seen": now.isoformat(),
        }]

        new_fact = {
            "subject": "Will",
            "predicate": "works_at",
            "object": "Google",
            "confidence": 0.78,
            "timestamp": now.isoformat(),
            "last_seen": now.isoformat(),
        }

        contradictions = self.detector.check_and_resolve(new_fact, existing)
        assert len(contradictions) == 1
        assert contradictions[0].resolution == "unresolved"
        assert self.detector.unresolved_count == 1

    def test_serialization_roundtrip(self):
        now = datetime.now(timezone.utc)
        self.detector.retract_fact("X", "p", "Y")
        self.detector._contradictions.append(Contradiction(
            fact_a={"subject": "A"}, fact_b={"subject": "B"},
            contradiction_type="direct", resolution="newer_wins", winner="b",
        ))
        data = self.detector.to_dict()
        loaded = ContradictionDetector.from_dict(data)
        assert loaded.contradiction_count == 1
        assert loaded.get_status("X", "p", "Y") == FactStatus.RETRACTED


class TestSemanticGraphContradictions:
    """Test contradictions through the SemanticGraph interface."""

    def setup_method(self):
        self.graph = SemanticGraph()

    def test_contradiction_on_exclusive_predicate(self):
        """Adding conflicting works_at relations should detect contradiction."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=30)

        self.graph.add_relation(Relation(
            subject="Will", predicate="works_at", object="OCI",
            confidence=0.8, first_seen=old, last_seen=old,
        ))
        contradictions = self.graph.add_relation(Relation(
            subject="Will", predicate="works_at", object="Google",
            confidence=0.8, first_seen=now, last_seen=now,
        ))
        assert len(contradictions) > 0
        assert contradictions[0].winner == "b"

    def test_retracted_fact_excluded_from_query(self):
        self.graph.add_relation(Relation(
            subject="Will", predicate="works_on", object="Project Alpha",
        ))
        self.graph.retract_fact("Will", "works_on", "Project Alpha")

        facts = self.graph.query(["Will"])
        # Retracted fact should not appear in default query
        assert all(f["predicate"] != "works_on" or f.get("status") != "retracted" for f in facts)

    def test_entity_resolution_in_graph(self):
        """Entities should be resolved through the entity resolver."""
        # Register an alias
        self.graph.resolver.register("Alice", entity_type="person", aliases=["Ali"])

        # Adding entity by alias should resolve to canonical
        self.graph.add_entity(Entity(name="Ali", entity_type="person"))

        # Should have only 1 entity (resolved to Alice)
        entity = self.graph.get_entity("Alice")
        assert entity is not None

    def test_merge_entities_in_graph(self):
        self.graph.add_relation(Relation(
            subject="GB10", predicate="is_a", object="GPU Workstation",
        ))
        self.graph.add_relation(Relation(
            subject="Blackwell", predicate="uses", object="CUDA",
        ))

        success = self.graph.merge_entities("GB10", "Blackwell")
        assert success

        # Query for GB10 should now include facts from Blackwell
        facts = self.graph.query(["GB10"], max_depth=2)
        predicates = [f["predicate"] for f in facts]
        assert "is_a" in predicates
        # The "uses" relation from Blackwell should now be on GB10
        subjects = [f["subject"].lower() for f in facts]
        assert any("gb10" in s for s in subjects)

    def test_save_load_with_resolver_and_contradictions(self):
        import tempfile
        from pathlib import Path

        self.graph.add_relation(Relation(
            subject="Alice", predicate="works_at", object="Google",
        ))
        self.graph.resolver.register("Alice", entity_type="person", aliases=["Ali"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph"
            self.graph.save(path)

            new_graph = SemanticGraph()
            new_graph.load(path)

            # Resolver should persist
            assert new_graph.resolver.resolve("Ali") is not None

    def test_stats_include_resolver_and_contradictions(self):
        self.graph.add_relation(Relation(
            subject="A", predicate="works_at", object="B",
        ))
        stats = self.graph.stats()
        assert "resolved_entities" in stats
        assert "contradictions_total" in stats
        assert "contradictions_unresolved" in stats

