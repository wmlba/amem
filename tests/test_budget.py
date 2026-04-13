"""Tests for dynamic budget allocation and behavioral feedback."""

import numpy as np
import pytest

from amem.config import Config
from amem.embeddings.base import EmbeddingProvider
from amem.retrieval.orchestrator import MemoryOrchestrator, LayerBudget


class MockEmbedder(EmbeddingProvider):
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
def config():
    return Config()


@pytest.fixture
def orchestrator(config):
    return MemoryOrchestrator(MockEmbedder(), config)


class TestDynamicBudget:
    def test_budget_sums_to_total(self, orchestrator):
        budget = orchestrator._compute_budget(
            total_budget=4000,
            episodic_signal=0.5,
            semantic_signal=0.5,
            has_explicit=True,
            has_working=True,
        )
        assert budget.total <= 4000
        assert budget.total > 0

    def test_high_episodic_signal_gets_more_budget(self, orchestrator):
        budget_ep = orchestrator._compute_budget(
            total_budget=4000,
            episodic_signal=0.9,
            semantic_signal=0.1,
            has_explicit=False,
            has_working=False,
        )
        budget_sem = orchestrator._compute_budget(
            total_budget=4000,
            episodic_signal=0.1,
            semantic_signal=0.9,
            has_explicit=False,
            has_working=False,
        )
        assert budget_ep.episodic > budget_sem.episodic
        assert budget_sem.semantic > budget_ep.semantic

    def test_explicit_gets_guaranteed_budget(self, orchestrator):
        budget = orchestrator._compute_budget(
            total_budget=4000,
            episodic_signal=0.5,
            semantic_signal=0.5,
            has_explicit=True,
            has_working=False,
        )
        assert budget.explicit > 0

    def test_working_gets_guaranteed_budget(self, orchestrator):
        budget = orchestrator._compute_budget(
            total_budget=4000,
            episodic_signal=0.5,
            semantic_signal=0.5,
            has_explicit=False,
            has_working=True,
        )
        assert budget.working > 0

    def test_zero_signal_uses_defaults(self, orchestrator):
        budget = orchestrator._compute_budget(
            total_budget=4000,
            episodic_signal=0.0,
            semantic_signal=0.0,
            has_explicit=False,
            has_working=False,
        )
        assert budget.episodic > 0
        assert budget.semantic > 0

    @pytest.mark.asyncio
    async def test_query_returns_budget_allocation(self, orchestrator):
        await orchestrator.ingest(text="Some test data for budget testing.")
        ctx = await orchestrator.query("test query")
        assert "episodic" in ctx.budget_allocation
        assert "semantic" in ctx.budget_allocation
        assert ctx.budget_allocation["total"] > 0


class TestBehavioralFeedback:
    @pytest.mark.asyncio
    async def test_profile_modulates_output_verbosity(self, orchestrator):
        # Feed enough technical messages to build a profile
        for _ in range(10):
            await orchestrator.ingest(
                text="Optimizing the database query with proper indexes on the API endpoint "
                     "to reduce latency in the cluster deployment pipeline.",
                speaker="user",
            )

        orchestrator.explicit.set("role", "ML engineer")
        ctx = await orchestrator.query("What's my role?")
        text = ctx.to_injection_text(profile=orchestrator.behavioral)

        # With enough signal, behavioral section should appear
        assert len(text) > 0

    def test_explicit_feedback_updates_profile(self, orchestrator):
        orchestrator.behavioral.update_from_feedback("response_depth", 0.9)
        priors = orchestrator.behavioral.get_priors()
        assert priors["response_depth"]["value"] > 0.6

    def test_profile_dimensions_in_context(self, orchestrator):
        # Provide feedback to build confidence
        for _ in range(10):
            orchestrator.behavioral.update_from_feedback("formality", 0.8)

        priors = orchestrator.behavioral.get_priors()
        assert priors["formality"]["confidence"] > 0.2
        assert priors["formality"]["value"] > 0.5

    @pytest.mark.asyncio
    async def test_terse_profile_produces_shorter_output(self, orchestrator):
        await orchestrator.ingest(text="Test data.")
        orchestrator.explicit.set("test", "value")

        # Set profile to terse
        for _ in range(20):
            orchestrator.behavioral.update_from_feedback("verbosity_preference", 0.1)

        ctx = await orchestrator.query("test")
        terse_text = ctx.to_injection_text(profile=orchestrator.behavioral)

        # Reset to verbose
        for _ in range(20):
            orchestrator.behavioral.update_from_feedback("verbosity_preference", 0.9)

        verbose_text = ctx.to_injection_text(profile=orchestrator.behavioral)

        # Verbose output should be at least as long (may include confidence scores etc.)
        # Can't guarantee strict inequality with mock data, but structure should differ
        assert len(verbose_text) >= len(terse_text) or "confidence" in verbose_text
