"""Episodic memory store — Layer 1.

Ingests conversations, chunks, embeds, deduplicates, scores importance,
and stores in the Temporal Associative Index (TAI).

Performance: fused batch dedup+importance eliminates redundant searches.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from amem.config import Config
from amem.embeddings.base import EmbeddingProvider
from amem.episodic.chunker import SentenceChunker
from amem.episodic.importance import score_importance_batch
from amem.episodic.smart_dedup import is_true_duplicate
from amem.episodic.temporal_index import TemporalAssociativeIndex, ChunkMeta, SearchResult

# Keep old index for backward compatibility
from amem.episodic.vector_index import AssociativeIndex, ChunkMetadata

if TYPE_CHECKING:
    from amem.persistence.sqlite import SQLiteStore


class EpisodicStore:
    """Manages episodic memory with the Temporal Associative Index."""

    def __init__(self, embedder: EmbeddingProvider, config: Config):
        self._embedder = embedder
        self._config = config
        self._chunker = SentenceChunker(
            sentences_per_chunk=config.episodic.chunk_sentences,
            overlap=config.episodic.chunk_overlap,
        )
        # New: Temporal Associative Index
        self._tai = TemporalAssociativeIndex(
            dimension=embedder.dimension,
            decay_lambda=config.vector_index.temporal_decay_lambda,
        )
        # Legacy: keep old index for backward compat
        self._index = AssociativeIndex(
            dimension=embedder.dimension,
            temporal_decay_lambda=config.vector_index.temporal_decay_lambda,
            reinforcement_weight=config.vector_index.reinforcement_weight,
            ivf_threshold=config.vector_index.ivf_threshold,
            n_partitions=config.vector_index.n_partitions,
        )
        self._db: SQLiteStore | None = None
        self._use_tai = True  # Feature flag for new index

    def set_db(self, db: SQLiteStore):
        self._db = db

    @property
    def index(self) -> AssociativeIndex:
        """Legacy index access for backward compat."""
        return self._index

    @property
    def tai(self) -> TemporalAssociativeIndex:
        return self._tai

    async def ingest(
        self,
        text: str,
        conversation_id: str | None = None,
        speaker: str = "",
        entity_mentions: list[str] | None = None,
        topic_tags: list[str] | None = None,
        timestamp: datetime | None = None,
    ) -> list[str]:
        """Ingest text with fused batch dedup + importance scoring.

        Pipeline:
        1. Chunk text into overlapping sentence windows
        2. Embed all chunks in a single batch call
        3. Batch dedup + novelty scoring (ONE matmul, not N searches)
        4. Compute importance (reuses novelty from step 3)
        5. Store non-duplicate chunks with importance-weighted confidence
        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        chunks = self._chunker.chunk(text)
        if not chunks:
            return []

        # Step 2: Batch embed all chunks
        texts = [c.text for c in chunks]
        vectors = await self._embedder.embed_batch(texts)

        ts = timestamp.timestamp() if isinstance(timestamp, datetime) else float(timestamp)

        if self._use_tai:
            return self._ingest_tai(chunks, vectors, ts, conversation_id, speaker,
                                     entity_mentions, topic_tags)
        else:
            return self._ingest_legacy(chunks, vectors, timestamp, conversation_id, speaker,
                                        entity_mentions, topic_tags)

    def _ingest_tai(self, chunks, vectors, ts, conversation_id, speaker,
                    entity_mentions, topic_tags) -> list[str]:
        """Ingest using Temporal Associative Index with fused batch operations."""

        # Step 3: Batch dedup + novelty — ONE operation for ALL chunks
        cosine_is_dup, novelty_scores = self._tai.batch_dedup_and_score(vectors)

        # Step 3b: NOVEL — Smart dedup: check distinctive tokens, not just cosine
        # "Working on Kubernetes..." and "Working on React..." may have cosine > 0.95
        # but contain DIFFERENT information. Smart dedup catches this.
        is_duplicate = []
        for i, (chunk, dup) in enumerate(zip(chunks, cosine_is_dup)):
            if dup:
                # Cosine says duplicate — verify with smart dedup
                # Find what it matched against
                nearest = self._tai._hot.search_fused(
                    vectors[i] / max(float(np.linalg.norm(vectors[i])), 1e-8),
                    now_ts=ts, top_k=1,
                    temporal_weight=0.0, reinforcement_weight=0.0, importance_weight=0.0,
                )
                if nearest:
                    true_dup = is_true_duplicate(
                        chunk.text, nearest[0].meta.text, nearest[0].similarity,
                    )
                    is_duplicate.append(true_dup)
                else:
                    is_duplicate.append(False)
            else:
                is_duplicate.append(False)

        # Step 4: Batch importance scoring (reuses novelty from step 3)
        importance_scores = score_importance_batch(
            texts=[c.text for c in chunks],
            novelty_scores=novelty_scores,
        )

        # Step 5: Store non-duplicates
        chunk_ids = []
        db_batch = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            if is_duplicate[i]:
                continue

            meta = ChunkMeta(
                chunk_id=str(uuid.uuid4()),
                text=chunk.text,
                timestamp=ts,
                conversation_id=conversation_id,
                speaker=speaker,
                entity_mentions=entity_mentions or [],
                topic_tags=topic_tags or [],
                confidence=importance_scores[i],
                importance=importance_scores[i],
            )
            cid = self._tai.add(vec, meta)
            chunk_ids.append(cid)

            # Also add to legacy index for backward compat
            legacy_meta = ChunkMetadata(
                chunk_id=cid,
                text=chunk.text,
                timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                conversation_id=conversation_id,
                speaker=speaker,
                entity_mentions=entity_mentions or [],
                topic_tags=topic_tags or [],
                confidence=importance_scores[i],
            )
            self._index.add(vec, legacy_meta)

            if self._db is not None:
                db_batch.append((cid, vec, legacy_meta.to_dict()))

        if self._db and db_batch:
            self._db.save_chunks_batch(db_batch)

        return chunk_ids

    def _ingest_legacy(self, chunks, vectors, timestamp, conversation_id, speaker,
                       entity_mentions, topic_tags) -> list[str]:
        """Legacy ingest path using old AssociativeIndex."""
        chunk_ids = []
        db_batch = []
        for chunk, vec in zip(chunks, vectors):
            meta = ChunkMetadata(
                text=chunk.text,
                timestamp=timestamp,
                conversation_id=conversation_id,
                speaker=speaker,
                entity_mentions=entity_mentions or [],
                topic_tags=topic_tags or [],
                confidence=self._config.vector_index.confidence_default,
            )
            cid = self._index.add(vec, meta)
            chunk_ids.append(cid)
            if self._db is not None:
                db_batch.append((cid, vec, meta.to_dict()))

        if self._db and db_batch:
            self._db.save_chunks_batch(db_batch)
        return chunk_ids

    async def ingest_conversation(
        self,
        messages: list[dict],
        conversation_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> list[str]:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        all_ids = []
        for msg in messages:
            ids = await self.ingest(
                text=msg.get("text", ""),
                conversation_id=conversation_id,
                speaker=msg.get("speaker", ""),
                entity_mentions=msg.get("entity_mentions", []),
                topic_tags=msg.get("topic_tags", []),
                timestamp=timestamp,
            )
            all_ids.extend(ids)
        return all_ids

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
        temporal_weight: float = 0.3,
    ) -> list:
        """Retrieve relevant chunks."""
        if top_k is None:
            top_k = self._config.episodic.default_top_k

        query_vec = await self._embedder.embed(query)

        if self._use_tai:
            results = self._tai.search(
                query_vec, top_k=top_k,
                temporal_weight=temporal_weight,
                filters=filters,
            )
            for r in results:
                self._tai.reinforce(r.chunk_id)
        else:
            results = self._index.search(
                query_vec, top_k=top_k,
                filters=filters,
                temporal_weight=temporal_weight,
            )
            for r in results:
                self._index.reinforce(r.chunk_id)

        return results

    def save(self, path: Path | None = None):
        if path is None:
            path = Path(self._config.storage.data_dir) / "episodic"
        self._index.save(path)
        tai_path = path / "tai"
        self._tai.save(tai_path)

    def load(self, path: Path | None = None):
        if path is None:
            path = Path(self._config.storage.data_dir) / "episodic"
        if (path / "index_config.msgpack").exists():
            self._index = AssociativeIndex.load(path)
        tai_path = path / "tai"
        if (tai_path / "tai_config.msgpack").exists():
            self._tai = TemporalAssociativeIndex.load(tai_path)

    def load_from_db(self):
        if self._db is None:
            return
        vectors, metadatas = self._db.load_all_chunks(self._embedder.dimension)
        if not vectors:
            return
        self._index = AssociativeIndex(
            dimension=self._embedder.dimension,
            temporal_decay_lambda=self._config.vector_index.temporal_decay_lambda,
            reinforcement_weight=self._config.vector_index.reinforcement_weight,
            ivf_threshold=self._config.vector_index.ivf_threshold,
            n_partitions=self._config.vector_index.n_partitions,
        )
        self._tai = TemporalAssociativeIndex(
            dimension=self._embedder.dimension,
            decay_lambda=self._config.vector_index.temporal_decay_lambda,
        )
        for vec, meta_dict in zip(vectors, metadatas):
            legacy_meta = ChunkMetadata.from_dict(meta_dict)
            self._index.add(vec, legacy_meta)
            tai_meta = ChunkMeta(
                chunk_id=meta_dict.get("chunk_id", ""),
                text=meta_dict.get("text", ""),
                timestamp=datetime.fromisoformat(meta_dict["timestamp"]).timestamp()
                    if isinstance(meta_dict.get("timestamp"), str) else float(meta_dict.get("timestamp", 0)),
                conversation_id=meta_dict.get("conversation_id", ""),
                speaker=meta_dict.get("speaker", ""),
                entity_mentions=meta_dict.get("entity_mentions", []),
                topic_tags=meta_dict.get("topic_tags", []),
                confidence=meta_dict.get("confidence", 1.0),
                access_count=meta_dict.get("access_count", 0),
            )
            self._tai.add(vec, tai_meta)

    def stats(self) -> dict:
        s = self._index.stats()
        s["tai"] = self._tai.stats()
        if self._db:
            s["db_chunks"] = self._db.get_chunk_count()
        return s
