"""Tests for SQLite persistence backend."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from amem.persistence.sqlite import SQLiteStore
from amem.config import Config
from amem.embeddings.base import EmbeddingProvider
from amem.retrieval.orchestrator import MemoryOrchestrator


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


class TestSQLiteStore:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.db = SQLiteStore(self.db_path)

    def teardown_method(self):
        self.db.close()

    def test_init_creates_db(self):
        assert self.db_path.exists()

    def test_save_and_load_chunk(self):
        vec = np.random.randn(64).astype(np.float32)
        self.db.save_chunk("chunk-1", vec, {
            "text": "Hello world",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversation_id": "conv-1",
            "speaker": "user",
        })

        vectors, metadatas = self.db.load_all_chunks(64)
        assert len(vectors) == 1
        assert metadatas[0]["text"] == "Hello world"
        assert metadatas[0]["chunk_id"] == "chunk-1"
        np.testing.assert_array_almost_equal(vectors[0], vec, decimal=5)

    def test_batch_save_chunks(self):
        chunks = []
        for i in range(10):
            vec = np.random.randn(64).astype(np.float32)
            chunks.append((f"chunk-{i}", vec, {
                "text": f"Chunk {i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }))
        self.db.save_chunks_batch(chunks)

        assert self.db.get_chunk_count() == 10

    def test_update_chunk_access(self):
        vec = np.random.randn(64).astype(np.float32)
        self.db.save_chunk("chunk-1", vec, {
            "text": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.db.update_chunk_access("chunk-1", 5)

        _, metadatas = self.db.load_all_chunks(64)
        assert metadatas[0]["access_count"] == 5

    def test_delete_chunk(self):
        vec = np.random.randn(64).astype(np.float32)
        self.db.save_chunk("chunk-1", vec, {
            "text": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.db.delete_chunk("chunk-1")
        assert self.db.get_chunk_count() == 0

    def test_save_and_load_explicit(self):
        self.db.save_explicit("role", {
            "value": "ML engineer",
            "entry_type": "fact",
            "priority": 10,
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        })

        entries = self.db.load_all_explicit()
        assert len(entries) == 1
        assert entries[0]["key"] == "role"
        assert entries[0]["value"] == "ML engineer"

    def test_delete_explicit(self):
        self.db.save_explicit("key1", {
            "value": "val",
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        })
        self.db.delete_explicit("key1")
        assert len(self.db.load_all_explicit()) == 0

    def test_save_and_load_profile(self):
        self.db.save_profile_dimension("response_depth", {
            "value": 0.75,
            "confidence": 0.9,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "signals": [0.6, 0.7, 0.8],
        })

        dims = self.db.load_all_profile_dimensions()
        assert "response_depth" in dims
        assert dims["response_depth"]["value"] == 0.75
        assert dims["response_depth"]["signals"] == [0.6, 0.7, 0.8]

    def test_save_and_load_entity(self):
        self.db.save_entity("alice", {
            "name": "Alice",
            "entity_type": "person",
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
        })

        entities = self.db.load_all_entities()
        assert len(entities) == 1
        assert entities[0]["name"] == "Alice"

    def test_save_and_load_relation(self):
        self.db.save_relation("alice", "ml pipeline", {
            "predicate": "works_on",
            "confidence": 0.8,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
        })

        relations = self.db.load_all_relations()
        assert len(relations) == 1
        assert relations[0]["predicate"] == "works_on"

    def test_save_and_load_contradiction(self):
        self.db.save_contradiction({
            "fact_a": {"subject": "Will", "predicate": "works_at", "object": "OCI"},
            "fact_b": {"subject": "Will", "predicate": "works_at", "object": "Google"},
            "contradiction_type": "direct",
            "resolution": "newer_wins",
            "winner": "b",
        })

        contradictions = self.db.load_all_contradictions()
        assert len(contradictions) == 1
        assert contradictions[0]["winner"] == "b"

    def test_fact_status(self):
        self.db.save_fact_status("will|works_at|oci", "superseded")
        statuses = self.db.load_all_fact_statuses()
        assert statuses["will|works_at|oci"] == "superseded"

    def test_user_isolation(self):
        """Two users should have isolated data."""
        db_user1 = SQLiteStore(self.db_path, user_id="user1")
        db_user2 = SQLiteStore(self.db_path, user_id="user2")

        db_user1.save_explicit("key1", {
            "value": "user1_value",
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        })
        db_user2.save_explicit("key1", {
            "value": "user2_value",
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        })

        u1_entries = db_user1.load_all_explicit()
        u2_entries = db_user2.load_all_explicit()

        assert len(u1_entries) == 1
        assert u1_entries[0]["value"] == "user1_value"
        assert len(u2_entries) == 1
        assert u2_entries[0]["value"] == "user2_value"

        db_user1.close()
        db_user2.close()

    def test_delete_user_data(self):
        self.db.save_explicit("key1", {
            "value": "val",
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        })
        vec = np.random.randn(64).astype(np.float32)
        self.db.save_chunk("chunk-1", vec, {
            "text": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        self.db.delete_user_data("default")

        assert self.db.get_chunk_count() == 0
        assert len(self.db.load_all_explicit()) == 0

    def test_transaction_rollback(self):
        """Failed transaction should not leave partial state."""
        try:
            with self.db.transaction() as conn:
                conn.execute("""
                    INSERT INTO explicit_entries (key, value, entry_type, priority, created, updated, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("key1", '"val1"', "fact", 0,
                      datetime.now(timezone.utc).isoformat(),
                      datetime.now(timezone.utc).isoformat(),
                      "default"))
                # Force an error
                raise ValueError("simulated error")
        except ValueError:
            pass

        assert len(self.db.load_all_explicit()) == 0

    def test_stats(self):
        vec = np.random.randn(64).astype(np.float32)
        self.db.save_chunk("chunk-1", vec, {
            "text": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.db.save_explicit("key1", {
            "value": "val",
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat(),
        })

        stats = self.db.stats()
        assert stats["chunks"] == 1
        assert stats["explicit_entries"] == 1

    def test_backup(self):
        vec = np.random.randn(64).astype(np.float32)
        self.db.save_chunk("chunk-1", vec, {
            "text": "backup test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        backup_path = Path(self.tmpdir) / "backup.db"
        self.db.backup(backup_path)
        assert backup_path.exists()

        # Load backup and verify
        backup_db = SQLiteStore(backup_path)
        assert backup_db.get_chunk_count() == 1
        backup_db.close()


class TestSQLiteIntegration:
    """Test SQLite through the orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config()
            config.storage.data_dir = tmpdir
            embedder = MockEmbedder()
            orch = MemoryOrchestrator(embedder, config)
            orch.init_db(Path(tmpdir) / "amem.db")

            # Ingest
            result = await orch.ingest(
                text="Alice works on the ML pipeline using Python.",
                conversation_id="conv-1",
                speaker="user",
            )
            assert result["chunks_stored"] > 0

            # Explicit memory
            orch.explicit.set("role", "engineer")

            # Query
            ctx = await orch.query("What does Alice work on?")
            assert len(ctx.explicit_entries) > 0

            # Verify stats show SQLite
            stats = orch.stats()
            assert stats["persistence"] == "sqlite"
            assert stats["db_stats"]["chunks"] > 0

            orch.close()

    @pytest.mark.asyncio
    async def test_sqlite_persistence_roundtrip(self):
        """Data survives orchestrator restart via SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "amem.db"
            config = Config()
            config.storage.data_dir = tmpdir

            # Session 1: ingest data
            embedder = MockEmbedder()
            orch1 = MemoryOrchestrator(embedder, config)
            orch1.init_db(db_path)

            await orch1.ingest(text="Bob leads infrastructure.", speaker="user")
            orch1.explicit.set("name", "Bob")
            orch1.save()
            orch1.close()

            # Session 2: new orchestrator, same DB
            orch2 = MemoryOrchestrator(MockEmbedder(), config)
            orch2.init_db(db_path)
            orch2.load()

            # Verify data survived
            assert orch2.episodic.index.count > 0
            assert orch2.explicit.count == 1
            assert orch2.explicit.get("name").value == "Bob"

            orch2.close()

    @pytest.mark.asyncio
    async def test_sqlite_incremental_writes(self):
        """Each ingest writes to SQLite immediately, no manual save needed for core data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "amem.db"
            config = Config()
            config.storage.data_dir = tmpdir

            embedder = MockEmbedder()
            orch = MemoryOrchestrator(embedder, config)
            orch.init_db(db_path)

            await orch.ingest(text="Data point one.")
            await orch.ingest(text="Data point two.")

            # Check DB directly — should have chunks without manual save()
            db = SQLiteStore(db_path)
            assert db.get_chunk_count() >= 2
            db.close()

            orch.close()
