"""Tests for the custom AssociativeIndex vector search engine."""

import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from amem.episodic.vector_index import AssociativeIndex, ChunkMetadata


def _random_vec(dim: int = 64, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestAssociativeIndex:
    def setup_method(self):
        self.dim = 64
        self.index = AssociativeIndex(dimension=self.dim)
        self.rng = np.random.default_rng(42)

    def test_add_and_count(self):
        vec = _random_vec(self.dim, self.rng)
        meta = ChunkMetadata(text="hello world")
        cid = self.index.add(vec, meta)
        assert self.index.count == 1
        assert cid == meta.chunk_id
        assert len(cid) > 0

    def test_search_returns_similar(self):
        # Add a known vector and search with same direction
        target = _random_vec(self.dim, self.rng)
        meta = ChunkMetadata(text="target chunk")
        self.index.add(target, meta)

        # Add noise vectors
        for i in range(20):
            self.index.add(
                _random_vec(self.dim, self.rng),
                ChunkMetadata(text=f"noise {i}"),
            )

        results = self.index.search(target, top_k=5)
        assert len(results) > 0
        # The target itself should be the top result (highest similarity)
        assert results[0].metadata.text == "target chunk"
        assert results[0].raw_similarity > 0.9

    def test_temporal_decay_affects_ranking(self):
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=365)

        vec = _random_vec(self.dim, self.rng)

        # Add old chunk
        self.index.add(vec.copy(), ChunkMetadata(text="old chunk", timestamp=old_time))
        # Add recent chunk with same vector
        self.index.add(vec.copy(), ChunkMetadata(text="new chunk", timestamp=now))

        results = self.index.search(vec, top_k=2, temporal_weight=0.5, now=now)
        assert len(results) == 2
        # Recent chunk should rank higher due to temporal decay
        assert results[0].metadata.text == "new chunk"
        assert results[0].temporal_factor > results[1].temporal_factor

    def test_reinforcement_affects_ranking(self):
        vec = _random_vec(self.dim, self.rng)

        id1 = self.index.add(vec.copy(), ChunkMetadata(text="unreinforced"))
        id2 = self.index.add(vec.copy(), ChunkMetadata(text="reinforced"))

        # Reinforce the second one many times
        for _ in range(50):
            self.index.reinforce(id2)

        results = self.index.search(
            vec, top_k=2,
            temporal_weight=0.0,
            reinforcement_weight=0.5,
        )
        assert results[0].metadata.text == "reinforced"
        assert results[0].reinforcement_factor > results[1].reinforcement_factor

    def test_metadata_filter_conversation_id(self):
        vec = _random_vec(self.dim, self.rng)

        self.index.add(vec.copy(), ChunkMetadata(text="conv A", conversation_id="aaa"))
        self.index.add(vec.copy(), ChunkMetadata(text="conv B", conversation_id="bbb"))

        results = self.index.search(vec, filters={"conversation_id": "aaa"})
        assert len(results) == 1
        assert results[0].metadata.conversation_id == "aaa"

    def test_metadata_filter_entity_mentions(self):
        vec = _random_vec(self.dim, self.rng)

        self.index.add(vec.copy(), ChunkMetadata(text="about Alice", entity_mentions=["Alice"]))
        self.index.add(vec.copy(), ChunkMetadata(text="about Bob", entity_mentions=["Bob"]))

        results = self.index.search(vec, filters={"entity_mentions": ["Alice"]})
        assert len(results) == 1
        assert "Alice" in results[0].metadata.entity_mentions

    def test_metadata_filter_time_range(self):
        now = datetime.now(timezone.utc)
        vec = _random_vec(self.dim, self.rng)

        self.index.add(vec.copy(), ChunkMetadata(text="old", timestamp=now - timedelta(days=30)))
        self.index.add(vec.copy(), ChunkMetadata(text="new", timestamp=now))

        results = self.index.search(vec, filters={"after": now - timedelta(days=1)})
        assert len(results) == 1
        assert results[0].metadata.text == "new"

    def test_confidence_affects_score(self):
        vec = _random_vec(self.dim, self.rng)

        self.index.add(vec.copy(), ChunkMetadata(text="low conf", confidence=0.1))
        self.index.add(vec.copy(), ChunkMetadata(text="high conf", confidence=1.0))

        results = self.index.search(vec, top_k=2)
        assert results[0].metadata.text == "high conf"

    def test_remove(self):
        vec = _random_vec(self.dim, self.rng)
        cid = self.index.add(vec, ChunkMetadata(text="to remove"))
        assert self.index.count == 1

        assert self.index.remove(cid)
        assert self.index.count == 0
        assert not self.index.remove("nonexistent")

    def test_save_and_load(self):
        for i in range(10):
            self.index.add(
                _random_vec(self.dim, self.rng),
                ChunkMetadata(text=f"chunk {i}", conversation_id=f"conv-{i}"),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index"
            self.index.save(path)

            loaded = AssociativeIndex.load(path)
            assert loaded.count == 10
            assert loaded.dimension == self.dim

            # Verify search still works
            query = _random_vec(self.dim, self.rng)
            results = loaded.search(query, top_k=5)
            assert len(results) == 5

    def test_decay_pass(self):
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=500)

        vec = _random_vec(self.dim, self.rng)
        self.index.add(vec, ChunkMetadata(text="very old", timestamp=old_time, confidence=1.0))

        self.index.decay_pass(now=now)
        meta = self.index._metadata[0]
        assert meta.confidence < 0.1  # heavily decayed

    def test_empty_search(self):
        results = self.index.search(_random_vec(self.dim, self.rng))
        assert results == []

    def test_stats(self):
        for i in range(5):
            self.index.add(
                _random_vec(self.dim, self.rng),
                ChunkMetadata(text=f"chunk {i}"),
            )
        stats = self.index.stats()
        assert stats["count"] == 5
        assert stats["dimension"] == self.dim
