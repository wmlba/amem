"""Custom vector search engine with temporal decay, reinforcement, and metadata filtering.

Purpose-built for associative memory retrieval where recency, access patterns,
and structured metadata matter as much as raw vector similarity.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgpack
import numpy as np


@dataclass
class ChunkMetadata:
    """Metadata attached to each stored chunk."""
    chunk_id: str = ""
    text: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_id: str = ""
    speaker: str = ""
    entity_mentions: list[str] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    access_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "conversation_id": self.conversation_id,
            "speaker": self.speaker,
            "entity_mentions": self.entity_mentions,
            "topic_tags": self.topic_tags,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ChunkMetadata:
        d = dict(d)
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


@dataclass
class SearchResult:
    """A single search result with scoring breakdown."""
    chunk_id: str
    metadata: ChunkMetadata
    raw_similarity: float
    temporal_factor: float
    reinforcement_factor: float
    final_score: float


class AssociativeIndex:
    """Custom vector index with integrated temporal decay and reinforcement scoring.

    For collections < ivf_threshold: brute-force cosine similarity.
    For larger collections: IVF-style partitioned search with centroids.
    """

    def __init__(
        self,
        dimension: int,
        temporal_decay_lambda: float = 0.01,
        reinforcement_weight: float = 0.1,
        ivf_threshold: int = 10000,
        n_partitions: int = 64,
    ):
        self.dimension = dimension
        self.temporal_decay_lambda = temporal_decay_lambda
        self.reinforcement_weight = reinforcement_weight
        self.ivf_threshold = ivf_threshold
        self.n_partitions = n_partitions

        # Core storage
        self._vectors: list[np.ndarray] = []
        self._metadata: list[ChunkMetadata] = []
        self._id_to_idx: dict[str, int] = {}

        # IVF structures (built lazily)
        self._centroids: np.ndarray | None = None
        self._partition_assignments: list[list[int]] | None = None
        self._ivf_dirty = True

    @property
    def count(self) -> int:
        return len(self._vectors)

    def add(self, vector: np.ndarray, metadata: ChunkMetadata) -> str:
        """Add a vector with metadata. Returns chunk_id."""
        if not metadata.chunk_id:
            metadata.chunk_id = str(uuid.uuid4())

        # Normalize vector for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        idx = len(self._vectors)
        self._vectors.append(vector.astype(np.float32))
        self._metadata.append(metadata)
        self._id_to_idx[metadata.chunk_id] = idx
        self._ivf_dirty = True

        return metadata.chunk_id

    def add_batch(self, vectors: list[np.ndarray], metadatas: list[ChunkMetadata]) -> list[str]:
        """Add multiple vectors at once."""
        ids = []
        for v, m in zip(vectors, metadatas):
            ids.append(self.add(v, m))
        return ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        temporal_weight: float = 0.3,
        reinforcement_weight: float | None = None,
        now: datetime | None = None,
    ) -> list[SearchResult]:
        """Search for top-k similar vectors with temporal decay and reinforcement.

        Scoring formula:
            final = sim * (1 - tw - rw) + temporal_factor * tw + reinforcement_factor * rw

        Where:
            sim = cosine similarity (0 to 1, vectors pre-normalized)
            temporal_factor = exp(-lambda * days_elapsed)
            reinforcement_factor = log(1 + access_count) / log(1 + max_access_count)
        """
        if self.count == 0:
            return []

        if now is None:
            now = datetime.now(timezone.utc)
        if reinforcement_weight is None:
            reinforcement_weight = self.reinforcement_weight

        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Determine candidate indices (with optional metadata pre-filtering)
        candidate_indices = self._get_candidates(query_vector, filters)

        if not candidate_indices:
            return []

        # Compute similarities for candidates
        candidate_vecs = np.array([self._vectors[i] for i in candidate_indices])
        similarities = candidate_vecs @ query_vector  # dot product of normalized vectors = cosine sim

        # Compute temporal and reinforcement factors
        max_access = max((self._metadata[i].access_count for i in candidate_indices), default=1)
        max_access = max(max_access, 1)

        results: list[SearchResult] = []
        sim_weight = 1.0 - temporal_weight - reinforcement_weight

        for rank, (cand_idx, sim) in enumerate(zip(candidate_indices, similarities)):
            meta = self._metadata[cand_idx]

            # Temporal decay
            days_elapsed = (now - meta.timestamp).total_seconds() / 86400.0
            temporal_factor = math.exp(-self.temporal_decay_lambda * days_elapsed)

            # Reinforcement
            rf = math.log(1 + meta.access_count) / math.log(1 + max_access) if max_access > 0 else 0.0

            # Confidence modulation
            conf = meta.confidence

            # Combined score
            raw_sim = float(sim)
            final = (raw_sim * sim_weight + temporal_factor * temporal_weight + rf * reinforcement_weight) * conf

            results.append(SearchResult(
                chunk_id=meta.chunk_id,
                metadata=meta,
                raw_similarity=raw_sim,
                temporal_factor=temporal_factor,
                reinforcement_factor=rf,
                final_score=final,
            ))

        # Sort by final score descending, take top_k
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:top_k]

    def _get_candidates(self, query_vector: np.ndarray, filters: dict[str, Any] | None) -> list[int]:
        """Get candidate indices, applying metadata filters and optional IVF narrowing."""
        # Start with all indices
        if self.count > self.ivf_threshold and self._use_ivf():
            candidates = self._ivf_candidates(query_vector)
        else:
            candidates = list(range(self.count))

        # Apply metadata filters
        if filters:
            candidates = self._apply_filters(candidates, filters)

        return candidates

    def _apply_filters(self, indices: list[int], filters: dict[str, Any]) -> list[int]:
        """Filter indices by metadata fields.

        Supported filter keys:
            - conversation_id: exact match
            - speaker: exact match
            - entity_mentions: any overlap (list intersection)
            - topic_tags: any overlap
            - min_confidence: minimum confidence threshold
            - after: datetime — only chunks after this time
            - before: datetime — only chunks before this time
        """
        result = []
        for idx in indices:
            meta = self._metadata[idx]
            keep = True

            if "conversation_id" in filters and meta.conversation_id != filters["conversation_id"]:
                keep = False
            if "speaker" in filters and meta.speaker != filters["speaker"]:
                keep = False
            if "entity_mentions" in filters:
                if not set(filters["entity_mentions"]) & set(meta.entity_mentions):
                    keep = False
            if "topic_tags" in filters:
                if not set(filters["topic_tags"]) & set(meta.topic_tags):
                    keep = False
            if "min_confidence" in filters and meta.confidence < filters["min_confidence"]:
                keep = False
            if "after" in filters and meta.timestamp < filters["after"]:
                keep = False
            if "before" in filters and meta.timestamp > filters["before"]:
                keep = False

            if keep:
                result.append(idx)
        return result

    def _use_ivf(self) -> bool:
        """Build or reuse IVF index.

        Only marks as clean if build actually succeeds (centroids != None).
        This fixes the bug where a failed build would prevent future rebuilds.
        """
        if self._ivf_dirty or self._centroids is None:
            self._build_ivf()
            # Only mark clean if we successfully built the index
            if self._centroids is not None:
                self._ivf_dirty = False
        return self._centroids is not None

    def _build_ivf(self):
        """Build IVF partitions using k-means on current vectors."""
        if self.count < self.n_partitions:
            self._centroids = None
            return

        vectors = np.array(self._vectors)
        n_parts = min(self.n_partitions, self.count // 10)

        # Simple k-means (good enough for prototype)
        rng = np.random.default_rng(42)
        centroid_indices = rng.choice(self.count, size=n_parts, replace=False)
        centroids = vectors[centroid_indices].copy()

        for _ in range(20):  # 20 iterations
            # Assign each vector to nearest centroid
            sims = vectors @ centroids.T
            assignments = np.argmax(sims, axis=1)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_parts):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = vectors[mask].mean(axis=0)
                    n = np.linalg.norm(new_centroids[k])
                    if n > 0:
                        new_centroids[k] /= n
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self._centroids = centroids

        # Build partition lists
        sims = vectors @ centroids.T
        assignments = np.argmax(sims, axis=1)
        self._partition_assignments = [[] for _ in range(n_parts)]
        for idx, part in enumerate(assignments):
            self._partition_assignments[part].append(idx)

    def _ivf_candidates(self, query_vector: np.ndarray, n_probes: int = 4) -> list[int]:
        """Return candidates from the closest IVF partitions."""
        if self._centroids is None:
            return list(range(self.count))

        sims = self._centroids @ query_vector
        top_partitions = np.argsort(sims)[-n_probes:]

        candidates = []
        for p in top_partitions:
            candidates.extend(self._partition_assignments[p])
        return candidates

    def reinforce(self, chunk_id: str):
        """Increment access count for a chunk (reinforcement signal)."""
        if chunk_id in self._id_to_idx:
            idx = self._id_to_idx[chunk_id]
            self._metadata[idx].access_count += 1

    def decay_pass(self, now: datetime | None = None):
        """Apply explicit confidence decay to all entries.

        Uses vectorized computation for performance on large indexes.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        if not self._metadata:
            return

        # Vectorized: compute all days-elapsed at once
        days_array = np.array([
            (now - meta.timestamp).total_seconds() / 86400.0
            for meta in self._metadata
        ], dtype=np.float64)

        decayed = np.exp(-self.temporal_decay_lambda * days_array)

        for i, meta in enumerate(self._metadata):
            meta.confidence = min(meta.confidence, float(decayed[i]))

    def remove(self, chunk_id: str) -> bool:
        """Remove a chunk by ID. Returns True if found."""
        if chunk_id not in self._id_to_idx:
            return False
        idx = self._id_to_idx.pop(chunk_id)

        # Swap-remove for O(1)
        last_idx = len(self._vectors) - 1
        if idx != last_idx:
            self._vectors[idx] = self._vectors[last_idx]
            self._metadata[idx] = self._metadata[last_idx]
            self._id_to_idx[self._metadata[idx].chunk_id] = idx

        self._vectors.pop()
        self._metadata.pop()
        self._ivf_dirty = True
        return True

    def save(self, path: Path):
        """Persist index to disk: vectors as .npy, metadata as msgpack."""
        path.mkdir(parents=True, exist_ok=True)

        if self._vectors:
            vectors = np.array(self._vectors)
            np.save(path / "vectors.npy", vectors)
        else:
            # Save empty marker
            np.save(path / "vectors.npy", np.empty((0, self.dimension), dtype=np.float32))

        meta_dicts = [m.to_dict() for m in self._metadata]
        with open(path / "metadata.msgpack", "wb") as f:
            msgpack.pack(meta_dicts, f, use_bin_type=True)

        # Save config
        config = {
            "dimension": self.dimension,
            "temporal_decay_lambda": self.temporal_decay_lambda,
            "reinforcement_weight": self.reinforcement_weight,
            "ivf_threshold": self.ivf_threshold,
            "n_partitions": self.n_partitions,
        }
        with open(path / "index_config.msgpack", "wb") as f:
            msgpack.pack(config, f, use_bin_type=True)

    @classmethod
    def load(cls, path: Path) -> AssociativeIndex:
        """Load index from disk."""
        with open(path / "index_config.msgpack", "rb") as f:
            config = msgpack.unpack(f, raw=False)

        index = cls(
            dimension=config["dimension"],
            temporal_decay_lambda=config["temporal_decay_lambda"],
            reinforcement_weight=config["reinforcement_weight"],
            ivf_threshold=config["ivf_threshold"],
            n_partitions=config["n_partitions"],
        )

        vectors = np.load(path / "vectors.npy")
        with open(path / "metadata.msgpack", "rb") as f:
            meta_dicts = msgpack.unpack(f, raw=False)

        for vec, md in zip(vectors, meta_dicts):
            meta = ChunkMetadata.from_dict(md)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            index._vectors.append(vec.astype(np.float32))
            index._metadata.append(meta)
            index._id_to_idx[meta.chunk_id] = len(index._vectors) - 1

        index._ivf_dirty = True
        return index

    def stats(self) -> dict:
        """Return index statistics."""
        if not self._metadata:
            return {"count": 0}
        ages = [(datetime.now(timezone.utc) - m.timestamp).days for m in self._metadata]
        accesses = [m.access_count for m in self._metadata]
        return {
            "count": self.count,
            "dimension": self.dimension,
            "avg_age_days": sum(ages) / len(ages),
            "max_age_days": max(ages),
            "avg_access_count": sum(accesses) / len(accesses),
            "ivf_active": self._centroids is not None,
        }
