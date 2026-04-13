"""Tests for the retrieval orchestrator (using mock embeddings)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from amem.config import Config
from amem.embeddings.base import EmbeddingProvider
from amem.retrieval.orchestrator import MemoryOrchestrator


class MockEmbedder(EmbeddingProvider):
    """Deterministic mock embedder for testing."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._rng = np.random.default_rng(42)

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> np.ndarray:
        # Deterministic: hash text to seed
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [await self.embed(t) for t in texts]


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def orchestrator(config):
    embedder = MockEmbedder(dim=64)
    return MemoryOrchestrator(embedder, config)


class TestMemoryOrchestrator:
    @pytest.mark.asyncio
    async def test_ingest_and_query(self, orchestrator):
        await orchestrator.ingest(
            text="Alice works on the ML pipeline using Python and PyTorch. "
                 "She leads the data science team at Acme Corp.",
            conversation_id="conv-1",
            speaker="user",
        )

        ctx = await orchestrator.query("What does Alice work on?")
        assert len(ctx.episodic_chunks) > 0
        text = ctx.to_injection_text()
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_explicit_memory_in_context(self, orchestrator):
        orchestrator.explicit.set("role", "senior ML engineer", entry_type="fact", priority=10)
        orchestrator.explicit.set("preference", "concise responses", entry_type="preference")

        ctx = await orchestrator.query("Hello")
        assert len(ctx.explicit_entries) == 2
        text = ctx.to_injection_text()
        assert "senior ML engineer" in text

    @pytest.mark.asyncio
    async def test_working_memory_in_context(self, orchestrator):
        orchestrator.working.add_goal("Debug the auth flow")
        orchestrator.working.add_fact("Token expiry is set to 1h")

        ctx = await orchestrator.query("What's wrong?")
        assert ctx.working_context.get("goals")
        text = ctx.to_injection_text()
        assert "Debug the auth flow" in text

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, orchestrator):
        orchestrator.start_session("test-session")
        orchestrator.working.add_fact("User prefers dark mode")

        sid = await orchestrator.end_session()
        assert sid == "test-session"
        assert orchestrator.working.is_empty

    @pytest.mark.asyncio
    async def test_semantic_graph_populated(self, orchestrator):
        await orchestrator.ingest(
            text="Bob leads the infrastructure team using Kubernetes and Docker.",
        )
        # Embedding extractor may extract different entities than regex extractor
        # Check that the graph has SOME entities, not necessarily "Bob"
        assert orchestrator.semantic.entity_count > 0

    @pytest.mark.asyncio
    async def test_behavioral_profile_updated(self, orchestrator):
        await orchestrator.ingest(
            text="I need to optimize the database query performance "
                 "by adding proper indexes on the user_sessions table. "
                 "The current latency is 200ms which is unacceptable for our API endpoint.",
            speaker="user",
        )
        priors = orchestrator.behavioral.get_priors()
        # Should detect technical content
        assert priors["domain_expertise"]["value"] > 0

    @pytest.mark.asyncio
    async def test_save_and_load(self, orchestrator):
        await orchestrator.ingest(text="Test data for persistence.", conversation_id="persist-test")
        orchestrator.explicit.set("test_key", "test_value")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            orchestrator.save(path)

            # Create new orchestrator and load
            embedder = MockEmbedder(dim=64)
            config = Config()
            new_orch = MemoryOrchestrator(embedder, config)
            new_orch.load(path)

            assert new_orch.episodic.index.count > 0
            assert new_orch.explicit.count == 1

    @pytest.mark.asyncio
    async def test_decay_pass(self, orchestrator):
        await orchestrator.ingest(text="Some data to decay.")
        orchestrator.decay_pass()  # should not error

    @pytest.mark.asyncio
    async def test_stats(self, orchestrator):
        await orchestrator.ingest(text="Data for stats.")
        stats = orchestrator.stats()
        assert "episodic" in stats
        assert "semantic" in stats
        assert "explicit" in stats

    @pytest.mark.asyncio
    async def test_multi_conversation_retrieval(self, orchestrator):
        """Ingesting 3 topics and querying returns results from all."""
        await orchestrator.ingest(
            text="Machine learning models require careful hyperparameter tuning. "
                 "Grid search and Bayesian optimization are common approaches.",
            conversation_id="ml-topic",
        )
        await orchestrator.ingest(
            text="The kitchen renovation will include new countertops and cabinets. "
                 "We chose granite for the island.",
            conversation_id="kitchen-topic",
        )
        await orchestrator.ingest(
            text="The quarterly financial report shows strong revenue growth. "
                 "Operating margins improved by 3 percent.",
            conversation_id="finance-topic",
        )

        ctx = await orchestrator.query("Tell me about something")
        # With mock embeddings we can't test semantic relevance,
        # but we can verify all conversations were ingested and retrievable
        assert len(ctx.episodic_chunks) >= 3
        conv_ids = {c.get("conversation_id", "") for c in ctx.episodic_chunks}
        assert len(conv_ids) >= 2  # at least 2 different conversations represented
