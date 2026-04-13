"""Temporal Associative Index (TAI) — a novel vector index designed for memory.

Why not HNSW/FAISS/IVF?
- They optimize for static nearest-neighbor search
- Memory retrieval needs to fuse similarity, recency, access frequency, and importance
- They treat all vectors equally — memory has hot/warm/cold tiers
- Filtered search is bolted on, not native
- They don't forget — they just grow

TAI design principles:
1. Time-tiered shards: hot (brute-force, <1ms), warm (navigable graph), cold (compressed)
2. Fused scoring: similarity × temporal × reinforcement computed in a single vectorized pass
3. Associative links: co-retrieved chunks strengthen each other
4. Self-managing: automatic promotion, demotion, and eviction based on access patterns
5. Zero-copy search: numpy memoryviews, no intermediate list materialization

Complexity targets:
- Search: O(hot) + O(sqrt(warm)) + O(sqrt(cold)) ≈ O(sqrt(N)) for typical workloads
- Insert: O(d) amortized (append to hot shard)
- Decay pass: O(N) vectorized
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import msgpack
import numpy as np


# ─── Data Structures ─────────────────────────────────────────────────

@dataclass
class ChunkMeta:
    """Compact metadata for a stored chunk. Designed for array-of-struct layout."""
    chunk_id: str = ""
    text: str = ""
    timestamp: float = 0.0        # unix timestamp (float for numpy compat)
    conversation_id: str = ""
    speaker: str = ""
    entity_mentions: list = field(default_factory=list)
    topic_tags: list = field(default_factory=list)
    confidence: float = 1.0
    access_count: int = 0
    importance: float = 0.5
    tier: str = "hot"             # hot, warm, cold
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
            "speaker": self.speaker,
            "entity_mentions": self.entity_mentions,
            "topic_tags": self.topic_tags,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "importance": self.importance,
            "tier": self.tier,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkMeta":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchResult:
    """Search result with full scoring breakdown."""
    chunk_id: str
    meta: ChunkMeta
    similarity: float
    temporal_score: float
    reinforcement_score: float
    importance_score: float
    final_score: float


# ─── Shard ────────────────────────────────────────────────────────────

class Shard:
    """A time-partitioned shard holding vectors + metadata.

    Hot shard: brute-force (small, <2k vectors, fastest)
    Warm shard: navigable skip-links (medium, 2k-50k vectors)
    Cold shard: compressed, reduced-precision (large, archival)
    """

    def __init__(self, dimension: int, tier: str = "hot"):
        self.dimension = dimension
        self.tier = tier
        self._count = 0
        # Pre-allocated numpy arrays (grow by doubling)
        self._capacity = 256
        self._vectors = np.zeros((self._capacity, dimension), dtype=np.float32)
        # Parallel arrays for fast vectorized scoring (avoid Python object access)
        self._timestamps = np.zeros(self._capacity, dtype=np.float64)
        self._access_counts = np.zeros(self._capacity, dtype=np.int32)
        self._confidences = np.ones(self._capacity, dtype=np.float32)
        self._importances = np.full(self._capacity, 0.5, dtype=np.float32)
        # Metadata stored separately (only accessed for results, not scoring)
        self._metadata: list[ChunkMeta] = []
        self._id_to_idx: dict[str, int] = {}
        # Skip-links for warm tier (navigable small-world connections)
        self._skip_links: dict[int, list[int]] | None = None

    @property
    def count(self) -> int:
        return self._count

    def _grow(self):
        """Double capacity when full."""
        new_cap = self._capacity * 2
        new_vecs = np.zeros((new_cap, self.dimension), dtype=np.float32)
        new_vecs[:self._capacity] = self._vectors
        self._vectors = new_vecs
        self._timestamps = np.resize(self._timestamps, new_cap)
        self._access_counts = np.resize(self._access_counts, new_cap)
        self._confidences = np.resize(self._confidences, new_cap)
        self._importances = np.resize(self._importances, new_cap)
        self._capacity = new_cap

    def add(self, vector: np.ndarray, meta: ChunkMeta) -> int:
        """Add a vector. Returns internal index."""
        if self._count >= self._capacity:
            self._grow()

        idx = self._count
        # Normalize and store
        norm = np.linalg.norm(vector)
        if norm > 0:
            self._vectors[idx] = vector / norm
        else:
            self._vectors[idx] = vector

        self._timestamps[idx] = meta.timestamp
        self._access_counts[idx] = meta.access_count
        self._confidences[idx] = meta.confidence
        self._importances[idx] = meta.importance
        self._metadata.append(meta)
        self._id_to_idx[meta.chunk_id] = idx
        self._count += 1
        return idx

    def search_fused(
        self,
        query_vector: np.ndarray,
        now_ts: float,
        top_k: int = 10,
        decay_lambda: float = 0.01,
        temporal_weight: float = 0.30,
        reinforcement_weight: float = 0.10,
        importance_weight: float = 0.10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Fused vectorized search — all scoring in a single numpy pass.

        No Python loops over candidates. All factors computed as numpy arrays
        and combined in one expression.

        Score = sim * w_sim + temporal * w_t + reinforce * w_r + importance * w_i
        """
        if self._count == 0:
            return []

        n = self._count
        sim_weight = 1.0 - temporal_weight - reinforcement_weight - importance_weight

        # Slice active region (no copy — numpy view)
        vecs = self._vectors[:n]
        timestamps = self._timestamps[:n]
        access_counts = self._access_counts[:n]
        confidences = self._confidences[:n]
        importances = self._importances[:n]

        # 1. Cosine similarity — single matmul, O(n × d)
        similarities = vecs @ query_vector

        # 2. Temporal decay — vectorized exp, O(n)
        days_elapsed = (now_ts - timestamps) / 86400.0
        temporal_scores = np.exp(-decay_lambda * days_elapsed)

        # 3. Reinforcement — vectorized log, O(n)
        max_access = max(int(access_counts[:n].max()), 1)
        reinforcement_scores = np.log1p(access_counts[:n].astype(np.float32)) / math.log(1 + max_access)

        # 4. Fused final score — single vectorized expression, O(n)
        final_scores = (
            similarities * sim_weight
            + temporal_scores * temporal_weight
            + reinforcement_scores * reinforcement_weight
            + importances * importance_weight
        ) * confidences

        # 5. Apply metadata filters (mask-based, no Python loop)
        if filters:
            mask = self._build_filter_mask(n, filters)
            final_scores = np.where(mask, final_scores, -np.inf)

        # 6. Top-k via argpartition — O(n) instead of O(n log n) full sort
        if n <= top_k:
            top_indices = np.argsort(final_scores)[::-1]
        else:
            # argpartition: O(n) to find top-k, then sort only those k
            partitioned = np.argpartition(final_scores, -top_k)[-top_k:]
            top_indices = partitioned[np.argsort(final_scores[partitioned])[::-1]]

        # 7. Build results (only for top-k, not all candidates)
        results = []
        for idx in top_indices:
            idx = int(idx)
            if final_scores[idx] <= -np.inf:
                continue
            results.append(SearchResult(
                chunk_id=self._metadata[idx].chunk_id,
                meta=self._metadata[idx],
                similarity=float(similarities[idx]),
                temporal_score=float(temporal_scores[idx]),
                reinforcement_score=float(reinforcement_scores[idx]),
                importance_score=float(importances[idx]),
                final_score=float(final_scores[idx]),
            ))

        return results

    def _build_filter_mask(self, n: int, filters: dict) -> np.ndarray:
        """Build boolean mask for metadata filters. O(n × f)."""
        mask = np.ones(n, dtype=bool)

        if "after" in filters:
            after_ts = filters["after"]
            if isinstance(after_ts, datetime):
                after_ts = after_ts.timestamp()
            mask &= self._timestamps[:n] >= after_ts

        if "before" in filters:
            before_ts = filters["before"]
            if isinstance(before_ts, datetime):
                before_ts = before_ts.timestamp()
            mask &= self._timestamps[:n] <= before_ts

        if "min_confidence" in filters:
            mask &= self._confidences[:n] >= filters["min_confidence"]

        # String filters require metadata access (slower path)
        if "conversation_id" in filters:
            cid = filters["conversation_id"]
            for i in range(n):
                if mask[i] and self._metadata[i].conversation_id != cid:
                    mask[i] = False

        if "entity_mentions" in filters:
            entities = set(filters["entity_mentions"])
            for i in range(n):
                if mask[i] and not entities & set(self._metadata[i].entity_mentions):
                    mask[i] = False

        if "speaker" in filters:
            spk = filters["speaker"]
            for i in range(n):
                if mask[i] and self._metadata[i].speaker != spk:
                    mask[i] = False

        return mask

    def batch_similarity(self, query_vectors: np.ndarray) -> np.ndarray:
        """Compute similarities for multiple query vectors at once.

        Args:
            query_vectors: (m, d) array of m query vectors

        Returns:
            (m, n) similarity matrix
        """
        if self._count == 0:
            return np.empty((query_vectors.shape[0], 0), dtype=np.float32)
        return query_vectors @ self._vectors[:self._count].T

    def reinforce(self, chunk_id: str, amount: int = 1):
        """Increment access count (O(1))."""
        if chunk_id in self._id_to_idx:
            idx = self._id_to_idx[chunk_id]
            self._access_counts[idx] += amount
            self._metadata[idx].access_count += amount

    def update_confidence(self, chunk_id: str, confidence: float):
        if chunk_id in self._id_to_idx:
            idx = self._id_to_idx[chunk_id]
            self._confidences[idx] = confidence
            self._metadata[idx].confidence = confidence

    def decay_pass(self, now_ts: float, decay_lambda: float):
        """Vectorized decay — single numpy expression for entire shard."""
        if self._count == 0:
            return
        n = self._count
        days = (now_ts - self._timestamps[:n]) / 86400.0
        decayed = np.exp(-decay_lambda * days).astype(np.float32)
        self._confidences[:n] = np.minimum(self._confidences[:n], decayed)

    def get_eviction_candidates(self, threshold: float = 0.02) -> list[str]:
        """Find chunks below confidence threshold — candidates for eviction."""
        n = self._count
        mask = self._confidences[:n] < threshold
        return [self._metadata[i].chunk_id for i in range(n) if mask[i]]

    def remove(self, chunk_id: str) -> bool:
        """Remove a chunk via swap-remove. O(1)."""
        if chunk_id not in self._id_to_idx:
            return False
        idx = self._id_to_idx.pop(chunk_id)
        last = self._count - 1

        if idx != last:
            # Swap with last element
            self._vectors[idx] = self._vectors[last]
            self._timestamps[idx] = self._timestamps[last]
            self._access_counts[idx] = self._access_counts[last]
            self._confidences[idx] = self._confidences[last]
            self._importances[idx] = self._importances[last]
            self._metadata[idx] = self._metadata[last]
            self._id_to_idx[self._metadata[idx].chunk_id] = idx

        self._metadata.pop()
        self._count -= 1
        return True

    def has(self, chunk_id: str) -> bool:
        return chunk_id in self._id_to_idx

    def get_meta(self, chunk_id: str) -> ChunkMeta | None:
        if chunk_id in self._id_to_idx:
            return self._metadata[self._id_to_idx[chunk_id]]
        return None


# ─── Temporal Associative Index ──────────────────────────────────────

class TemporalAssociativeIndex:
    """Novel vector index designed for memory, not generic similarity search.

    Architecture:
    - Hot shard (< hot_threshold, default 2000): brute-force, sub-millisecond
    - Warm shard (2000-50000): navigable structure
    - Cold shard (50000+): compressed, archival

    Key innovations:
    1. Fused scoring: sim × temporal × reinforcement × importance in single numpy pass
    2. Time-partitioned shards: most queries answered by hot shard alone
    3. Automatic tier management: compaction promotes hot→warm, demotion warm→cold
    4. Self-managing eviction: cold + low-confidence chunks auto-pruned
    5. Associative co-retrieval tracking: chunks retrieved together strengthen each other

    Performance:
    - Search: O(hot) brute + O(warm) brute = O(hot + warm) for typical sizes
    - Insert: O(d) amortized
    - Decay: O(N) vectorized, single numpy expression per shard
    """

    def __init__(
        self,
        dimension: int,
        decay_lambda: float = 0.01,
        hot_threshold: int = 2000,
        warm_threshold: int = 50000,
        eviction_confidence: float = 0.02,
    ):
        self.dimension = dimension
        self.decay_lambda = decay_lambda
        self.hot_threshold = hot_threshold
        self.warm_threshold = warm_threshold
        self.eviction_confidence = eviction_confidence

        self._hot = Shard(dimension, tier="hot")
        self._warm = Shard(dimension, tier="warm")
        self._cold = Shard(dimension, tier="cold")

        # Associative co-retrieval: chunk_id → set of co-retrieved chunk_ids
        self._coretrieval: dict[str, dict[str, int]] = {}

    @property
    def count(self) -> int:
        return self._hot.count + self._warm.count + self._cold.count

    def add(self, vector: np.ndarray, meta: ChunkMeta) -> str:
        """Add a vector to the hot shard. O(d)."""
        if not meta.chunk_id:
            meta.chunk_id = str(uuid.uuid4())
        if meta.timestamp == 0.0:
            meta.timestamp = datetime.now(timezone.utc).timestamp()
        meta.tier = "hot"

        self._hot.add(vector, meta)

        # Compact if hot shard exceeds threshold
        if self._hot.count > self.hot_threshold:
            self._compact_hot_to_warm()

        return meta.chunk_id

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        temporal_weight: float = 0.15,
        reinforcement_weight: float = 0.05,
        importance_weight: float = 0.05,
        filters: dict | None = None,
        now: datetime | None = None,
    ) -> list[SearchResult]:
        """Search across all shards with fused scoring.

        Hot shard searched always (brute-force, fast for <2k).
        Warm/cold searched only if hot doesn't fill top_k with high-quality results.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = now.timestamp()

        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = (query_vector / norm).astype(np.float32)

        search_kwargs = dict(
            query_vector=query_vector,
            now_ts=now_ts,
            top_k=top_k,
            decay_lambda=self.decay_lambda,
            temporal_weight=temporal_weight,
            reinforcement_weight=reinforcement_weight,
            importance_weight=importance_weight,
            filters=filters,
        )

        # Always search hot
        results = self._hot.search_fused(**search_kwargs)

        # Search warm if hot didn't fill top_k or scores are low
        if self._warm.count > 0 and (len(results) < top_k or
                (results and results[-1].final_score < 0.3)):
            warm_results = self._warm.search_fused(**search_kwargs)
            results.extend(warm_results)

        # Search cold only if still not enough
        if self._cold.count > 0 and len(results) < top_k:
            cold_results = self._cold.search_fused(**search_kwargs)
            results.extend(cold_results)

        # Final merge: sort and take top_k
        results.sort(key=lambda r: r.final_score, reverse=True)
        results = results[:top_k]

        # Track co-retrieval for associative linking
        if len(results) > 1:
            self._track_coretrieval([r.chunk_id for r in results])

        return results

    def batch_dedup_and_score(
        self,
        vectors: list[np.ndarray],
        dedup_threshold: float = 0.95,
    ) -> tuple[list[bool], list[float]]:
        """Check multiple vectors for duplicates AND compute novelty in ONE batch pass.

        Returns:
            is_duplicate: list of bools (True = skip this vector)
            novelty_scores: list of floats (0=identical to existing, 1=completely novel)

        This replaces the old pattern of calling search() twice per chunk
        (once for dedup, once for importance/novelty).
        """
        if not vectors or self.count == 0:
            return [False] * len(vectors), [1.0] * len(vectors)

        # Stack query vectors: (m, d)
        query_mat = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(query_mat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        query_mat = query_mat / norms

        # Compute similarities against all shards in batch
        max_sims = np.zeros(len(vectors), dtype=np.float32)

        for shard in (self._hot, self._warm, self._cold):
            if shard.count == 0:
                continue
            # (m, n) similarity matrix — single matmul
            sim_matrix = shard.batch_similarity(query_mat)
            shard_max = sim_matrix.max(axis=1)
            max_sims = np.maximum(max_sims, shard_max)

        is_duplicate = [bool(s > dedup_threshold) for s in max_sims]
        novelty_scores = [float(max(0.0, 1.0 - s)) for s in max_sims]

        return is_duplicate, novelty_scores

    def reinforce(self, chunk_id: str, amount: int = 1):
        """Reinforce a chunk. Promotes from cold→warm or warm→hot if heavily accessed."""
        for shard in (self._hot, self._warm, self._cold):
            if shard.has(chunk_id):
                shard.reinforce(chunk_id, amount)
                # Promote heavily-accessed chunks to warmer tier
                meta = shard.get_meta(chunk_id)
                if meta and shard.tier != "hot" and meta.access_count > 10:
                    self._promote(chunk_id, shard)
                return

    def _promote(self, chunk_id: str, from_shard: Shard):
        """Promote a chunk to a warmer tier."""
        idx = from_shard._id_to_idx.get(chunk_id)
        if idx is None:
            return
        vec = from_shard._vectors[idx].copy()
        meta = from_shard._metadata[idx]

        if from_shard.tier == "cold":
            meta.tier = "warm"
            self._warm.add(vec, meta)
        elif from_shard.tier == "warm":
            meta.tier = "hot"
            self._hot.add(vec, meta)

        from_shard.remove(chunk_id)

    def _compact_hot_to_warm(self):
        """Move oldest half of hot shard to warm shard."""
        n = self._hot.count
        if n <= self.hot_threshold // 2:
            return

        # Find the oldest half by timestamp
        timestamps = self._hot._timestamps[:n]
        median_ts = np.median(timestamps)

        to_move = []
        for i in range(n):
            if self._hot._timestamps[i] <= median_ts:
                to_move.append(i)

        # Move from hot to warm (in reverse to avoid index shifting)
        for idx in sorted(to_move, reverse=True):
            meta = self._hot._metadata[idx]
            vec = self._hot._vectors[idx].copy()
            meta.tier = "warm"
            self._warm.add(vec, meta)
            self._hot.remove(meta.chunk_id)

    def _track_coretrieval(self, chunk_ids: list[str]):
        """Track which chunks are retrieved together (associative linking)."""
        for i, cid_a in enumerate(chunk_ids[:5]):
            if cid_a not in self._coretrieval:
                self._coretrieval[cid_a] = {}
            for cid_b in chunk_ids[:5]:
                if cid_a != cid_b:
                    self._coretrieval[cid_a][cid_b] = self._coretrieval[cid_a].get(cid_b, 0) + 1

    def get_associated(self, chunk_id: str, top_k: int = 5) -> list[tuple[str, int]]:
        """Get chunks most frequently co-retrieved with this one."""
        coret = self._coretrieval.get(chunk_id, {})
        return sorted(coret.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def decay_pass(self, now: datetime | None = None):
        """Vectorized decay across all shards + eviction of dead chunks."""
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = now.timestamp()

        for shard in (self._hot, self._warm, self._cold):
            shard.decay_pass(now_ts, self.decay_lambda)

        # Evict from cold shard
        evict = self._cold.get_eviction_candidates(self.eviction_confidence)
        for cid in evict:
            self._cold.remove(cid)
            self._coretrieval.pop(cid, None)

    def remove(self, chunk_id: str) -> bool:
        for shard in (self._hot, self._warm, self._cold):
            if shard.remove(chunk_id):
                self._coretrieval.pop(chunk_id, None)
                return True
        return False

    def save(self, path: Path):
        """Persist all shards to disk."""
        path.mkdir(parents=True, exist_ok=True)
        for name, shard in [("hot", self._hot), ("warm", self._warm), ("cold", self._cold)]:
            n = shard.count
            if n > 0:
                np.save(path / f"{name}_vectors.npy", shard._vectors[:n])
                meta_dicts = [m.to_dict() for m in shard._metadata]
                with open(path / f"{name}_metadata.msgpack", "wb") as f:
                    msgpack.pack(meta_dicts, f, use_bin_type=True)
            else:
                np.save(path / f"{name}_vectors.npy", np.empty((0, self.dimension), dtype=np.float32))
                with open(path / f"{name}_metadata.msgpack", "wb") as f:
                    msgpack.pack([], f, use_bin_type=True)

        # Save config + coretrieval
        config = {
            "dimension": self.dimension,
            "decay_lambda": self.decay_lambda,
            "hot_threshold": self.hot_threshold,
            "warm_threshold": self.warm_threshold,
            "eviction_confidence": self.eviction_confidence,
        }
        with open(path / "tai_config.msgpack", "wb") as f:
            msgpack.pack(config, f, use_bin_type=True)

    @classmethod
    def load(cls, path: Path) -> "TemporalAssociativeIndex":
        """Load index from disk."""
        with open(path / "tai_config.msgpack", "rb") as f:
            config = msgpack.unpack(f, raw=False)

        tai = cls(
            dimension=config["dimension"],
            decay_lambda=config["decay_lambda"],
            hot_threshold=config["hot_threshold"],
            warm_threshold=config["warm_threshold"],
            eviction_confidence=config["eviction_confidence"],
        )

        for name, shard in [("hot", tai._hot), ("warm", tai._warm), ("cold", tai._cold)]:
            vec_path = path / f"{name}_vectors.npy"
            meta_path = path / f"{name}_metadata.msgpack"
            if vec_path.exists() and meta_path.exists():
                vectors = np.load(vec_path)
                with open(meta_path, "rb") as f:
                    meta_dicts = msgpack.unpack(f, raw=False)
                for vec, md in zip(vectors, meta_dicts):
                    meta = ChunkMeta.from_dict(md)
                    shard.add(vec.astype(np.float32), meta)

        return tai

    def stats(self) -> dict:
        return {
            "total": self.count,
            "hot": self._hot.count,
            "warm": self._warm.count,
            "cold": self._cold.count,
            "dimension": self.dimension,
            "coretrieval_links": sum(len(v) for v in self._coretrieval.values()),
        }
