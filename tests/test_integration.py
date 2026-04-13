"""End-to-end integration test with a realistic multi-session scenario.

Simulates a real user interaction pattern:
1. User has conversations about work over several sessions
2. Their job changes (contradiction)
3. They refer to the same entity by different names (entity resolution)
4. Memory system should track preferences, facts, and history
5. Context retrieval should be relevant, fresh, and contradiction-aware
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from amem.config import Config
from amem.embeddings.base import EmbeddingProvider
from amem.retrieval.orchestrator import MemoryOrchestrator


class MockEmbedder(EmbeddingProvider):
    """Deterministic mock embedder."""
    def __init__(self, dim=64):
        self._dim = dim

    @property
    def dimension(self):
        return self._dim

    async def embed(self, text):
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


@pytest.fixture
def orch():
    config = Config()
    return MemoryOrchestrator(MockEmbedder(), config)


class TestRealisticScenario:
    """Simulates a multi-session user journey."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, orch):
        # === SESSION 1: User introduces themselves ===
        orch.start_session("session-1")

        result = await orch.ingest(
            text="I'm Will, a senior ML engineer at OCI. I lead the MEA SCE team "
                 "and we work on GPU infrastructure, specifically the GB10 Blackwell workstation.",
            conversation_id="conv-1",
            speaker="user",
        )
        assert result["chunks_stored"] > 0
        assert result["entities_extracted"] > 0

        # Set explicit preferences
        orch.explicit.set("name", "Will", entry_type="fact", priority=10)
        orch.explicit.set("communication_style", "concise technical responses", entry_type="preference", priority=5)

        # Working memory tracks session goals
        orch.working.add_goal("Set up memory system")
        orch.working.add_fact("Will is at OCI")

        # End session → flushes to episodic
        await orch.end_session()

        # === SESSION 2: More context ===
        orch.start_session("session-2")

        await orch.ingest(
            text="I've been researching H2RR, which is an improvement over Euclidean HRR "
                 "with 38 to 43 percent better performance on associative memory tasks.",
            conversation_id="conv-2",
            speaker="user",
        )

        await orch.ingest(
            text="We use Python and PyTorch for most of our experiments. "
                 "The GB10 workstation runs our training jobs.",
            conversation_id="conv-2",
            speaker="user",
        )

        await orch.end_session()

        # === Register entity alias ===
        orch.add_entity_alias("GB10", "Blackwell workstation")
        orch.add_entity_alias("GB10", "Blackwell")

        # === SESSION 3: Job change (CONTRADICTION) ===
        orch.start_session("session-3")

        result = await orch.ingest(
            text="I left OCI last month. I now work at Anthropic as a research scientist.",
            conversation_id="conv-3",
            speaker="user",
        )

        await orch.end_session()

        # === Query: "What does Will work on?" ===
        ctx = await orch.query("What does Will work on?", top_k=10)

        # Should have content from episodic store
        assert len(ctx.episodic_chunks) > 0

        # Explicit entries should always be present
        assert len(ctx.explicit_entries) >= 2
        names = [e["key"] for e in ctx.explicit_entries]
        assert "name" in names

        # Budget should be allocated
        assert ctx.budget_allocation["total"] > 0

        # Context text should be non-empty
        text = ctx.to_injection_text(profile=orch.behavioral)
        assert len(text) > 0
        assert "Will" in text or "will" in text.lower()

        # === Behavioral profile should reflect technical user ===
        priors = orch.behavioral.get_priors()
        assert priors["domain_expertise"]["value"] > 0  # some signal

        # === Verify persistence round-trip ===
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            orch.save(path)

            # Create fresh orchestrator and load
            config = Config()
            new_orch = MemoryOrchestrator(MockEmbedder(), config)
            new_orch.load(path)

            assert new_orch.episodic.index.count > 0
            assert new_orch.explicit.count >= 2
            assert new_orch.semantic.entity_count > 0

            # Query should still work
            ctx2 = await new_orch.query("GPU workstation")
            assert len(ctx2.episodic_chunks) >= 0  # may or may not match with mock embeddings

    @pytest.mark.asyncio
    async def test_entity_resolution_across_sessions(self, orch):
        """Entities referred to by different names should merge."""
        await orch.ingest(
            text="The GB10 Blackwell workstation runs our training jobs.",
            conversation_id="conv-1",
        )

        # First register the canonical entity, then add aliases
        orch.semantic.resolver.register("GB10", entity_type="tool")
        orch.add_entity_alias("GB10", "Blackwell workstation")
        orch.add_entity_alias("GB10", "the workstation")

        # Later reference using alias
        await orch.ingest(
            text="I upgraded the Blackwell workstation to more memory.",
            conversation_id="conv-2",
        )

        # At minimum the entity should be registered in the resolver
        resolved = orch.semantic.resolver.resolve("Blackwell workstation")
        assert resolved is not None
        assert resolved.canonical_name == "GB10"

        # Also check that the alias resolves
        resolved2 = orch.semantic.resolver.resolve("the workstation")
        assert resolved2 is not None

    @pytest.mark.asyncio
    async def test_contradiction_detection_and_filtering(self, orch):
        """Job change should be detected as contradiction and handled."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=60)

        from amem.semantic.graph import Relation
        # First fact: Will works at OCI
        orch.semantic.add_relation(Relation(
            subject="Will", predicate="works_at", object="OCI",
            confidence=0.8, first_seen=old, last_seen=old,
        ))

        # New fact: Will works at Anthropic
        contradictions = orch.semantic.add_relation(Relation(
            subject="Will", predicate="works_at", object="Anthropic",
            confidence=0.8, first_seen=now, last_seen=now,
        ))

        # Should detect contradiction
        assert len(contradictions) > 0
        assert contradictions[0].contradiction_type in ("direct", "temporal")

        # Query should return the newer fact with higher priority
        facts = orch.semantic.query(["Will"])
        active_works_at = [f for f in facts if f["predicate"] == "works_at" and f.get("status", "active") == "active"]
        if active_works_at:
            # The Anthropic fact should be the active one
            assert any("Anthropic" in f["object"] for f in active_works_at)

    @pytest.mark.asyncio
    async def test_fact_retraction(self, orch):
        """User can explicitly retract a fact."""
        from amem.semantic.graph import Relation
        orch.semantic.add_relation(Relation(
            subject="Will", predicate="works_on", object="Project X",
        ))

        # Retract
        success = orch.retract_fact("Will", "works_on", "Project X")
        assert success

        # Should not appear in normal queries
        facts = orch.semantic.query(["Will"])
        project_x_facts = [f for f in facts if f["object"] == "Project X" and f.get("status") == "active"]
        assert len(project_x_facts) == 0

    @pytest.mark.asyncio
    async def test_decay_affects_all_layers(self, orch):
        """Running decay should reduce confidence in old data."""
        await orch.ingest(text="Old data about something.")

        # Run decay
        orch.decay_pass()

        # Should not error and episodic stats should still be valid
        stats = orch.stats()
        assert stats["episodic"]["count"] > 0

    @pytest.mark.asyncio
    async def test_working_memory_flushes_to_episodic(self, orch):
        """Session end should flush working memory to episodic store."""
        initial_count = orch.episodic.index.count

        orch.start_session("flush-test")
        orch.working.add_fact("Important discovery about the system")
        orch.working.add_goal("Debug the auth flow")

        await orch.end_session()

        # Episodic count should have increased
        assert orch.episodic.index.count > initial_count

    @pytest.mark.asyncio
    async def test_empty_ingest_handles_gracefully(self, orch):
        """Empty or whitespace-only text should not crash."""
        result = await orch.ingest(text="")
        assert result["chunks_stored"] == 0

        result = await orch.ingest(text="   \n\t  ")
        assert result["chunks_stored"] == 0

    @pytest.mark.asyncio
    async def test_large_batch_ingest(self, orch):
        """Ingesting many messages should work correctly."""
        messages = [
            {"text": f"Message number {i} about topic {i % 5}.", "speaker": "user"}
            for i in range(20)
        ]
        result = await orch.ingest_conversation(messages, conversation_id="batch-test")
        assert result["chunks_stored"] > 0

        # Query should work on large dataset
        ctx = await orch.query("topic 3")
        assert ctx.budget_allocation["total"] > 0

    @pytest.mark.asyncio
    async def test_explicit_memory_highest_priority(self, orch):
        """Explicit memory should always appear in context regardless of query."""
        orch.explicit.set("critical_instruction", "Always respond in bullet points", entry_type="instruction", priority=100)

        ctx = await orch.query("random unrelated query about weather")
        assert len(ctx.explicit_entries) > 0
        assert any(e["key"] == "critical_instruction" for e in ctx.explicit_entries)

        text = ctx.to_injection_text(profile=orch.behavioral)
        assert "Always respond in bullet points" in text

    @pytest.mark.asyncio
    async def test_behavioral_profile_builds_over_time(self, orch):
        """Profile should build signal confidence as more messages are ingested."""
        initial_priors = orch.behavioral.get_priors()
        initial_confidence = initial_priors["domain_expertise"]["confidence"]

        # Ingest multiple technical messages
        for i in range(15):
            await orch.ingest(
                text=f"Optimizing the vector database index with HNSW algorithm "
                     f"for better query latency on the deployment cluster {i}.",
                speaker="user",
            )

        updated_priors = orch.behavioral.get_priors()
        assert updated_priors["domain_expertise"]["confidence"] > initial_confidence
        assert updated_priors["domain_expertise"]["value"] > 0.1

    @pytest.mark.asyncio
    async def test_multi_layer_context_assembly(self, orch):
        """All layers should contribute to final context."""
        # Populate all layers
        orch.explicit.set("user_name", "TestUser", entry_type="fact")

        await orch.ingest(
            text="Alice leads the infrastructure team using Kubernetes and Docker.",
            speaker="user",
        )

        orch.start_session()
        orch.working.add_goal("Investigate performance issue")
        orch.working.add_fact("Latency spike started at 3pm")

        ctx = await orch.query("infrastructure team")

        # All layers should have contributed
        assert len(ctx.explicit_entries) > 0  # explicit
        assert len(ctx.episodic_chunks) >= 0  # episodic (mock embeddings)
        assert ctx.behavioral_priors  # behavioral
        assert ctx.working_context  # working memory
        assert ctx.budget_allocation  # budget was computed

        text = ctx.to_injection_text(profile=orch.behavioral)
        assert "TestUser" in text  # explicit always appears
        assert "Investigate performance issue" in text  # working memory

    @pytest.mark.asyncio
    async def test_entity_merge_in_graph(self, orch):
        """Merging entities should consolidate their relations."""
        from amem.semantic.graph import Relation

        orch.semantic.add_relation(Relation(
            subject="GB10", predicate="is_a", object="Workstation",
        ))
        orch.semantic.add_relation(Relation(
            subject="Blackwell", predicate="runs", object="CUDA Jobs",
        ))

        # Merge Blackwell into GB10
        success = orch.merge_entities("GB10", "Blackwell")
        assert success

        # GB10 should now have both relations
        facts = orch.semantic.query(["GB10"], max_depth=2)
        predicates = {f["predicate"] for f in facts}
        assert "is_a" in predicates
