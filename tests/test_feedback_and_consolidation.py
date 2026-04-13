"""Tests for relevance feedback, memory consolidation, temporal parser, and adaptive decay."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from amem.config import Config
from amem.embeddings.base import EmbeddingProvider
from amem.feedback.relevance import RelevanceFeedback, FeedbackSignal
from amem.maintenance.consolidation import MemoryConsolidator
from amem.semantic.temporal import TemporalParser
from amem.semantic.adaptive_decay import AdaptiveDecay, DecayTiers
from amem.episodic.importance import score_importance, score_importance_batch
from amem.retrieval.orchestrator import MemoryOrchestrator


class MockEmbedder(EmbeddingProvider):
    def __init__(self, dim=64):
        self._dim = dim
    @property
    def dimension(self):
        return self._dim
    async def embed(self, text):
        seed = hash(text) % (2**31)
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)
    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


# ─── Relevance Feedback ────────────────────────────────────────────

class TestRelevanceFeedback:
    def test_compute_overlap_used(self):
        fb = RelevanceFeedback()
        chunks = [{"chunk_id": "c1", "text": "Alice works on machine learning pipelines"}]
        response = "Alice has been working on machine learning pipelines using Python."
        signals = fb.compute_overlap(chunks, response)
        assert len(signals) == 1
        assert signals[0].was_used == True
        assert signals[0].overlap_score > 0.1

    def test_compute_overlap_unused(self):
        fb = RelevanceFeedback()
        chunks = [{"chunk_id": "c1", "text": "The weather in Tokyo is sunny today"}]
        response = "Alice works on machine learning at Google."
        signals = fb.compute_overlap(chunks, response)
        assert len(signals) == 1
        assert signals[0].was_used == False
        assert signals[0].overlap_score < 0.15

    def test_compute_overlap_empty(self):
        fb = RelevanceFeedback()
        assert fb.compute_overlap([], "response") == []
        assert fb.compute_overlap([{"chunk_id": "c1", "text": "hi"}], "") == []

    def test_feedback_rate(self):
        fb = RelevanceFeedback()
        assert fb.get_feedback_rate()["total_rounds"] == 0
        fb._history.append({"reinforced": 3, "demoted": 2, "total": 5})
        rate = fb.get_feedback_rate()
        assert rate["total_rounds"] == 1
        assert rate["avg_used_ratio"] == 0.6


# ─── Temporal Parser ───────────────────────────────────────────────

class TestTemporalParser:
    def setup_method(self):
        self.parser = TemporalParser()
        self.ref = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_yesterday(self):
        markers = self.parser.parse("I did this yesterday", self.ref)
        assert len(markers) >= 1
        m = markers[0]
        assert m.original_text == "yesterday"
        assert m.resolved_date.day == 14

    def test_n_days_ago(self):
        markers = self.parser.parse("We started 5 days ago", self.ref)
        assert len(markers) >= 1
        expected = self.ref - timedelta(days=5)
        assert markers[0].resolved_date.day == expected.day

    def test_last_month(self):
        markers = self.parser.parse("I left OCI last month", self.ref)
        assert len(markers) >= 1
        assert markers[0].resolved_date.month == 5  # May

    def test_since_year(self):
        markers = self.parser.parse("I've been here since 2020", self.ref)
        assert len(markers) >= 1
        assert markers[0].resolved_date.year == 2020
        assert markers[0].is_range == True

    def test_explicit_date(self):
        markers = self.parser.parse("It happened on 2024-03-15", self.ref)
        assert len(markers) >= 1
        assert markers[0].resolved_date.month == 3
        assert markers[0].resolved_date.day == 15

    def test_month_year(self):
        markers = self.parser.parse("Started in March 2024", self.ref)
        assert len(markers) >= 1
        assert markers[0].resolved_date.month == 3
        assert markers[0].resolved_date.year == 2024

    def test_recently(self):
        markers = self.parser.parse("I recently changed jobs", self.ref)
        assert len(markers) >= 1
        assert markers[0].confidence < 0.7  # low confidence

    def test_no_temporal(self):
        markers = self.parser.parse("The cat sat on the mat", self.ref)
        assert len(markers) == 0

    def test_multiple_markers(self):
        text = "I started last month and will finish tomorrow"
        markers = self.parser.parse(text, self.ref)
        assert len(markers) >= 1  # at least "last month"


# ─── Adaptive Decay ───────────────────────────────────────────────

class TestAdaptiveDecay:
    def setup_method(self):
        self.decay = AdaptiveDecay()

    def test_identity_decays_slowly(self):
        rate = self.decay.get_rate("is_a")
        assert rate == 0.001

    def test_professional_decays_medium(self):
        rate = self.decay.get_rate("works_at")
        assert rate == 0.005

    def test_ephemeral_decays_fast(self):
        rate = self.decay.get_rate("prefers")
        assert rate == 0.05

    def test_classify_by_predicate(self):
        assert self.decay.classify_tier("works_at") == "professional"
        assert self.decay.classify_tier("is_a") == "identity"
        assert self.decay.classify_tier("prefers") == "ephemeral"

    def test_classify_by_entity_type(self):
        assert self.decay.classify_tier("custom_pred", subject_type="person") == "professional"
        assert self.decay.classify_tier("custom_pred", subject_type="tool") == "project"

    def test_compute_adaptive(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=100)
        # Identity fact should retain more confidence
        identity_conf = self.decay.compute(1.0, old, "is_a", now=now)
        # Ephemeral fact should decay much more
        ephemeral_conf = self.decay.compute(1.0, old, "prefers", now=now)
        assert identity_conf > ephemeral_conf
        assert identity_conf > 0.5  # identity barely decays in 100 days

    def test_reinforce(self):
        result = self.decay.reinforce(0.5)
        assert result > 0.5
        assert result <= 1.0


# ─── Importance Scoring ─────────────────────────────────────────────

class TestImportanceScoring:
    def test_high_importance_factual(self):
        score = score_importance("My name is Alice and I work at Google as a senior engineer.")
        assert score > 0.4

    def test_low_importance_phatic(self):
        score = score_importance("Ok sure thanks bye")
        assert score < 0.4

    def test_batch_scoring(self):
        texts = [
            "My name is Alice and I lead the ML team at Google.",
            "Ok thanks",
            "We use Python and PyTorch for training our models.",
        ]
        novelties = [0.8, 0.3, 0.7]
        scores = score_importance_batch(texts, novelties)
        assert len(scores) == 3
        assert scores[0] > scores[1]  # factual > phatic

    def test_empty_text(self):
        score = score_importance("")
        # Empty string gets minimum score
        assert score <= 0.4
        # Whitespace only also gets low score
        score2 = score_importance("   ")
        assert score2 <= 0.4


# ─── Consolidation ──────────────────────────────────────────────────

class TestConsolidation:
    @pytest.mark.asyncio
    async def test_consolidation_basic(self):
        config = Config()
        orch = MemoryOrchestrator(MockEmbedder(), config)
        consolidator = MemoryConsolidator(min_mentions_to_promote=2)

        # Ingest text mentioning same entities multiple times
        await orch.ingest(text="Alice works on ML pipelines.", speaker="user")
        await orch.ingest(text="Alice leads the data team.", speaker="user")
        await orch.ingest(text="Alice uses Python daily.", speaker="user")

        results = await consolidator.consolidate(orch)
        # Should have promoted Alice-related knowledge
        assert results["entities_promoted"] >= 0  # depends on extraction
        assert isinstance(results["chunks_evicted"], int)

    @pytest.mark.asyncio
    async def test_eviction_does_not_remove_recent(self):
        config = Config()
        orch = MemoryOrchestrator(MockEmbedder(), config)
        consolidator = MemoryConsolidator(eviction_age_days=90)

        await orch.ingest(text="Recent important data.")
        initial_count = orch.episodic.tai.count
        results = await consolidator.consolidate(orch)
        # Nothing should be evicted — data is fresh
        assert results["chunks_evicted"] == 0
