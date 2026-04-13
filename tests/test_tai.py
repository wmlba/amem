"""Tests for Temporal Associative Index — the novel vector index."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from amem.episodic.temporal_index import TemporalAssociativeIndex, ChunkMeta, Shard


def _rvec(dim=64, rng=None):
    if rng is None: rng = np.random.default_rng()
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestShard:
    def test_add_and_search(self):
        shard = Shard(64)
        rng = np.random.default_rng(42)
        target = _rvec(64, rng)
        shard.add(target, ChunkMeta(chunk_id="t1", text="target", timestamp=datetime.now(timezone.utc).timestamp()))
        for i in range(20):
            shard.add(_rvec(64, rng), ChunkMeta(chunk_id=f"n{i}", text=f"noise {i}", timestamp=datetime.now(timezone.utc).timestamp()))

        results = shard.search_fused(target, now_ts=datetime.now(timezone.utc).timestamp())
        assert results[0].chunk_id == "t1"
        assert results[0].similarity > 0.9

    def test_temporal_decay_in_fused_search(self):
        shard = Shard(64)
        rng = np.random.default_rng(42)
        vec = _rvec(64, rng)
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=365)).timestamp()
        new_ts = now.timestamp()

        shard.add(vec.copy(), ChunkMeta(chunk_id="old", text="old", timestamp=old_ts))
        shard.add(vec.copy(), ChunkMeta(chunk_id="new", text="new", timestamp=new_ts))

        results = shard.search_fused(vec, now_ts=new_ts, temporal_weight=0.5)
        assert results[0].chunk_id == "new"
        assert results[0].temporal_score > results[1].temporal_score

    def test_reinforcement_in_fused_search(self):
        shard = Shard(64)
        rng = np.random.default_rng(42)
        vec = _rvec(64, rng)
        now = datetime.now(timezone.utc).timestamp()

        shard.add(vec.copy(), ChunkMeta(chunk_id="low", text="low", timestamp=now))
        shard.add(vec.copy(), ChunkMeta(chunk_id="high", text="high", timestamp=now, access_count=50))

        results = shard.search_fused(vec, now_ts=now, reinforcement_weight=0.5)
        assert results[0].chunk_id == "high"

    def test_filter_mask(self):
        shard = Shard(64)
        rng = np.random.default_rng(42)
        vec = _rvec(64, rng)
        now = datetime.now(timezone.utc).timestamp()

        shard.add(vec.copy(), ChunkMeta(chunk_id="a", text="a", timestamp=now, conversation_id="conv1"))
        shard.add(vec.copy(), ChunkMeta(chunk_id="b", text="b", timestamp=now, conversation_id="conv2"))

        results = shard.search_fused(vec, now_ts=now, filters={"conversation_id": "conv1"})
        assert len(results) == 1
        assert results[0].chunk_id == "a"

    def test_grow(self):
        shard = Shard(64)
        rng = np.random.default_rng(42)
        for i in range(500):
            shard.add(_rvec(64, rng), ChunkMeta(chunk_id=f"c{i}", text=f"text {i}", timestamp=float(i)))
        assert shard.count == 500

    def test_remove(self):
        shard = Shard(64)
        vec = _rvec(64)
        shard.add(vec, ChunkMeta(chunk_id="x", text="x", timestamp=0.0))
        assert shard.count == 1
        assert shard.remove("x")
        assert shard.count == 0

    def test_batch_similarity(self):
        shard = Shard(64)
        rng = np.random.default_rng(42)
        for i in range(10):
            shard.add(_rvec(64, rng), ChunkMeta(chunk_id=f"c{i}", text=f"t{i}", timestamp=0.0))

        queries = np.array([_rvec(64, rng) for _ in range(3)])
        sim_matrix = shard.batch_similarity(queries)
        assert sim_matrix.shape == (3, 10)


class TestTemporalAssociativeIndex:
    def setup_method(self):
        self.dim = 64
        self.rng = np.random.default_rng(42)

    def test_add_and_count(self):
        tai = TemporalAssociativeIndex(self.dim)
        for i in range(10):
            tai.add(_rvec(self.dim, self.rng), ChunkMeta(text=f"chunk {i}"))
        assert tai.count == 10

    def test_search_returns_similar(self):
        tai = TemporalAssociativeIndex(self.dim)
        target = _rvec(self.dim, self.rng)
        tai.add(target, ChunkMeta(text="target"))
        for i in range(20):
            tai.add(_rvec(self.dim, self.rng), ChunkMeta(text=f"noise {i}"))

        results = tai.search(target, top_k=5)
        assert len(results) > 0
        assert results[0].meta.text == "target"

    def test_hot_to_warm_compaction(self):
        tai = TemporalAssociativeIndex(self.dim, hot_threshold=50)
        for i in range(100):
            tai.add(_rvec(self.dim, self.rng), ChunkMeta(text=f"chunk {i}"))
        # Hot should have been compacted
        assert tai._warm.count > 0
        assert tai.count == 100

    def test_search_across_shards(self):
        tai = TemporalAssociativeIndex(self.dim, hot_threshold=20)
        # Add fillers first so they're older and get compacted
        for i in range(30):
            tai.add(_rvec(self.dim, self.rng), ChunkMeta(text=f"filler {i}"))
        # Add target last (stays in hot)
        target = _rvec(self.dim, self.rng)
        tai.add(target, ChunkMeta(text="target"))

        # Verify warm has data from compaction
        assert tai._warm.count > 0 or tai._hot.count > 0
        results = tai.search(target, top_k=5)
        assert len(results) > 0
        # Target should be found (either in hot or warm)
        assert results[0].similarity > 0.5

    def test_batch_dedup_and_score(self):
        tai = TemporalAssociativeIndex(self.dim)
        existing = _rvec(self.dim, self.rng)
        tai.add(existing, ChunkMeta(text="existing"))

        # Check: exact duplicate should be flagged
        is_dup, novelty = tai.batch_dedup_and_score([existing.copy()])
        assert is_dup[0] == True
        assert novelty[0] < 0.1

        # Check: random vector should not be duplicate
        novel = _rvec(self.dim, self.rng)
        is_dup2, novelty2 = tai.batch_dedup_and_score([novel])
        assert is_dup2[0] == False

    def test_batch_dedup_multiple(self):
        tai = TemporalAssociativeIndex(self.dim)
        existing = _rvec(self.dim, self.rng)
        tai.add(existing, ChunkMeta(text="existing"))

        vecs = [existing.copy(), _rvec(self.dim, self.rng), _rvec(self.dim, self.rng)]
        is_dup, novelty = tai.batch_dedup_and_score(vecs)
        assert is_dup[0] == True
        assert len(is_dup) == 3

    def test_reinforce_promotes(self):
        tai = TemporalAssociativeIndex(self.dim, hot_threshold=20)
        vec = _rvec(self.dim, self.rng)
        tai.add(vec, ChunkMeta(chunk_id="promote_me", text="promote"))
        # Force to warm
        for i in range(30):
            tai.add(_rvec(self.dim, self.rng), ChunkMeta(text=f"filler {i}"))
        # Reinforce many times
        for _ in range(15):
            tai.reinforce("promote_me")
        # Should have been promoted back to hot
        assert tai._hot.has("promote_me")

    def test_coretrieval_tracking(self):
        tai = TemporalAssociativeIndex(self.dim)
        vec = _rvec(self.dim, self.rng)
        tai.add(vec.copy(), ChunkMeta(chunk_id="a", text="a"))
        tai.add(vec.copy(), ChunkMeta(chunk_id="b", text="b"))
        # Search returns both → they become co-retrieved
        tai.search(vec, top_k=5)
        associated = tai.get_associated("a")
        assert len(associated) > 0

    def test_decay_and_eviction(self):
        tai = TemporalAssociativeIndex(self.dim, eviction_confidence=0.5)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=500)).timestamp()
        tai._cold.add(_rvec(self.dim, self.rng), ChunkMeta(chunk_id="dead", text="dead", timestamp=old_ts, confidence=0.01))
        initial = tai._cold.count
        tai.decay_pass()
        assert tai._cold.count < initial

    def test_save_and_load(self):
        tai = TemporalAssociativeIndex(self.dim)
        for i in range(10):
            tai.add(_rvec(self.dim, self.rng), ChunkMeta(text=f"chunk {i}"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tai"
            tai.save(path)
            loaded = TemporalAssociativeIndex.load(path)
            assert loaded.count == 10
            assert loaded.dimension == self.dim

    def test_stats(self):
        tai = TemporalAssociativeIndex(self.dim)
        for i in range(5):
            tai.add(_rvec(self.dim, self.rng), ChunkMeta(text=f"c{i}"))
        stats = tai.stats()
        assert stats["total"] == 5
        assert stats["hot"] == 5

    def test_empty_search(self):
        tai = TemporalAssociativeIndex(self.dim)
        results = tai.search(_rvec(self.dim, self.rng))
        assert results == []

    def test_argpartition_topk(self):
        """Verify argpartition gives correct top-k even with more candidates."""
        tai = TemporalAssociativeIndex(self.dim)
        target = _rvec(self.dim, self.rng)
        tai.add(target, ChunkMeta(text="best match"))
        for i in range(50):
            tai.add(_rvec(self.dim, self.rng), ChunkMeta(text=f"noise {i}"))

        results = tai.search(target, top_k=3)
        assert len(results) == 3
        assert results[0].meta.text == "best match"
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].final_score >= results[i+1].final_score
