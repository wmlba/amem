"""SQLite-based persistence backend with WAL mode.

Replaces the naive full-dump .npy/.json persistence with:
- Single amem.db file with WAL mode for concurrent reads
- Incremental writes (INSERT/UPDATE individual records)
- Transaction support (ingest is atomic)
- Schema migrations via version table
- Crash-safe: WAL guarantees durability
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Episodic chunks: vectors + metadata
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    text TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    conversation_id TEXT NOT NULL DEFAULT '',
    speaker TEXT NOT NULL DEFAULT '',
    entity_mentions TEXT NOT NULL DEFAULT '[]',
    topic_tags TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    extra TEXT NOT NULL DEFAULT '{}',
    user_id TEXT NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_chunks_conversation ON chunks(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);

-- Semantic graph: entities
CREATE TABLE IF NOT EXISTS entities (
    entity_key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    attributes TEXT NOT NULL DEFAULT '{}',
    aliases TEXT NOT NULL DEFAULT '[]',
    user_id TEXT NOT NULL DEFAULT 'default'
);
CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);

-- Semantic graph: relations
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_key TEXT NOT NULL,
    object_key TEXT NOT NULL,
    predicate TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.8,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    source_chunks TEXT NOT NULL DEFAULT '[]',
    mention_count INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    user_id TEXT NOT NULL DEFAULT 'default',
    UNIQUE(subject_key, object_key, predicate, user_id)
);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_key);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_key);
CREATE INDEX IF NOT EXISTS idx_relations_user ON relations(user_id);

-- Entity resolver: canonical entities
CREATE TABLE IF NOT EXISTS canonical_entities (
    canonical_key TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    aliases TEXT NOT NULL DEFAULT '[]',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    attributes TEXT NOT NULL DEFAULT '{}',
    user_id TEXT NOT NULL DEFAULT 'default'
);

-- Contradictions
CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_a TEXT NOT NULL,
    fact_b TEXT NOT NULL,
    contradiction_type TEXT NOT NULL,
    resolution TEXT NOT NULL DEFAULT 'unresolved',
    winner TEXT NOT NULL DEFAULT '',
    resolved_at TEXT,
    user_id TEXT NOT NULL DEFAULT 'default'
);

-- Fact status tracking
CREATE TABLE IF NOT EXISTS fact_status (
    fact_key TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    user_id TEXT NOT NULL DEFAULT 'default'
);

-- Explicit memory entries
CREATE TABLE IF NOT EXISTS explicit_entries (
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    entry_type TEXT NOT NULL DEFAULT 'fact',
    priority INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (key, user_id)
);

-- Behavioral profile dimensions
CREATE TABLE IF NOT EXISTS profile_dimensions (
    dimension TEXT NOT NULL,
    value REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.1,
    last_updated TEXT NOT NULL,
    signals TEXT NOT NULL DEFAULT '[]',
    user_id TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (dimension, user_id)
);

-- Index configuration (for vector index rebuild)
CREATE TABLE IF NOT EXISTS index_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteStore:
    """SQLite persistence backend for all memory layers.

    Uses WAL mode for concurrent read access and atomic transactions
    for write operations.
    """

    def __init__(self, db_path: str | Path, user_id: str = "default"):
        self.db_path = Path(db_path)
        self.user_id = user_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self):
        """Initialize database with schema."""
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        # Set WAL mode for concurrent reads
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Check/set schema version
        cur = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def transaction(self):
        """Context manager for atomic transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ==================== Episodic (Chunks) ====================

    def save_chunk(self, chunk_id: str, vector: np.ndarray, metadata: dict):
        """Insert or update a single chunk."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO chunks
            (chunk_id, vector, text, timestamp, conversation_id, speaker,
             entity_mentions, topic_tags, confidence, access_count, extra, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            vector.astype(np.float32).tobytes(),
            metadata.get("text", ""),
            metadata.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metadata.get("conversation_id", ""),
            metadata.get("speaker", ""),
            json.dumps(metadata.get("entity_mentions", [])),
            json.dumps(metadata.get("topic_tags", [])),
            metadata.get("confidence", 1.0),
            metadata.get("access_count", 0),
            json.dumps(metadata.get("extra", {})),
            self.user_id,
        ))
        conn.commit()

    def save_chunks_batch(self, chunks: list[tuple[str, np.ndarray, dict]]):
        """Insert multiple chunks in a single transaction."""
        with self.transaction() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, vector, text, timestamp, conversation_id, speaker,
                 entity_mentions, topic_tags, confidence, access_count, extra, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    cid,
                    vec.astype(np.float32).tobytes(),
                    meta.get("text", ""),
                    meta.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    meta.get("conversation_id", ""),
                    meta.get("speaker", ""),
                    json.dumps(meta.get("entity_mentions", [])),
                    json.dumps(meta.get("topic_tags", [])),
                    meta.get("confidence", 1.0),
                    meta.get("access_count", 0),
                    json.dumps(meta.get("extra", {})),
                    self.user_id,
                )
                for cid, vec, meta in chunks
            ])

    def update_chunk_access(self, chunk_id: str, access_count: int):
        """Increment access count (reinforcement)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE chunks SET access_count = ? WHERE chunk_id = ? AND user_id = ?",
            (access_count, chunk_id, self.user_id),
        )
        conn.commit()

    def update_chunk_confidence(self, chunk_id: str, confidence: float):
        """Update confidence after decay."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE chunks SET confidence = ? WHERE chunk_id = ? AND user_id = ?",
            (confidence, chunk_id, self.user_id),
        )
        conn.commit()

    def delete_chunk(self, chunk_id: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks WHERE chunk_id = ? AND user_id = ?", (chunk_id, self.user_id))
        conn.commit()

    def load_all_chunks(self, dimension: int) -> tuple[list[np.ndarray], list[dict]]:
        """Load all chunks for building the in-memory vector index."""
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM chunks WHERE user_id = ? ORDER BY timestamp",
            (self.user_id,),
        )
        vectors = []
        metadatas = []
        for row in cur:
            vec = np.frombuffer(row["vector"], dtype=np.float32).copy()
            if len(vec) != dimension:
                continue  # skip corrupted vectors
            vectors.append(vec)
            metadatas.append({
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "timestamp": row["timestamp"],
                "conversation_id": row["conversation_id"],
                "speaker": row["speaker"],
                "entity_mentions": json.loads(row["entity_mentions"]),
                "topic_tags": json.loads(row["topic_tags"]),
                "confidence": row["confidence"],
                "access_count": row["access_count"],
                "extra": json.loads(row["extra"]),
            })
        return vectors, metadatas

    def get_chunk_count(self) -> int:
        conn = self._get_conn()
        cur = conn.execute("SELECT COUNT(*) FROM chunks WHERE user_id = ?", (self.user_id,))
        return cur.fetchone()[0]

    # ==================== Semantic Graph ====================

    def save_entity(self, key: str, data: dict):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO entities
            (entity_key, name, entity_type, first_seen, last_seen, mention_count, attributes, aliases, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            data.get("name", key),
            data.get("entity_type", "concept"),
            data.get("first_seen", datetime.now(timezone.utc).isoformat()),
            data.get("last_seen", datetime.now(timezone.utc).isoformat()),
            data.get("mention_count", 1),
            json.dumps(data.get("attributes", {})),
            json.dumps(data.get("aliases", [])),
            self.user_id,
        ))
        conn.commit()

    def save_relation(self, subject_key: str, object_key: str, data: dict):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO relations
            (subject_key, object_key, predicate, confidence, first_seen, last_seen,
             source_chunks, mention_count, status, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subject_key,
            object_key,
            data.get("predicate", ""),
            data.get("confidence", 0.8),
            data.get("first_seen", datetime.now(timezone.utc).isoformat()),
            data.get("last_seen", datetime.now(timezone.utc).isoformat()),
            json.dumps(data.get("source_chunks", [])),
            data.get("mention_count", 1),
            data.get("status", "active"),
            self.user_id,
        ))
        conn.commit()

    def load_all_entities(self) -> list[dict]:
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM entities WHERE user_id = ?", (self.user_id,))
        return [
            {
                "key": row["entity_key"],
                "name": row["name"],
                "entity_type": row["entity_type"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "mention_count": row["mention_count"],
                "attributes": json.loads(row["attributes"]),
                "aliases": json.loads(row["aliases"]),
            }
            for row in cur
        ]

    def load_all_relations(self) -> list[dict]:
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM relations WHERE user_id = ?", (self.user_id,))
        return [
            {
                "subject_key": row["subject_key"],
                "object_key": row["object_key"],
                "predicate": row["predicate"],
                "confidence": row["confidence"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "source_chunks": json.loads(row["source_chunks"]),
                "mention_count": row["mention_count"],
                "status": row["status"],
            }
            for row in cur
        ]

    def delete_entity(self, key: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM entities WHERE entity_key = ? AND user_id = ?", (key, self.user_id))
        conn.commit()

    def delete_relation(self, subject_key: str, object_key: str, predicate: str):
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM relations WHERE subject_key = ? AND object_key = ? AND predicate = ? AND user_id = ?",
            (subject_key, object_key, predicate, self.user_id),
        )
        conn.commit()

    # ==================== Entity Resolver ====================

    def save_canonical_entity(self, key: str, data: dict):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO canonical_entities
            (canonical_key, canonical_name, entity_type, aliases, first_seen, last_seen,
             mention_count, attributes, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            data.get("canonical_name", key),
            data.get("entity_type", "concept"),
            json.dumps(data.get("aliases", [])),
            data.get("first_seen", datetime.now(timezone.utc).isoformat()),
            data.get("last_seen", datetime.now(timezone.utc).isoformat()),
            data.get("mention_count", 1),
            json.dumps(data.get("attributes", {})),
            self.user_id,
        ))
        conn.commit()

    def load_all_canonical_entities(self) -> list[dict]:
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM canonical_entities WHERE user_id = ?", (self.user_id,))
        return [
            {
                "canonical_name": row["canonical_name"],
                "entity_type": row["entity_type"],
                "aliases": json.loads(row["aliases"]),
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "mention_count": row["mention_count"],
                "attributes": json.loads(row["attributes"]),
            }
            for row in cur
        ]

    # ==================== Contradictions ====================

    def save_contradiction(self, data: dict):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO contradictions
            (fact_a, fact_b, contradiction_type, resolution, winner, resolved_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            json.dumps(data.get("fact_a", {})),
            json.dumps(data.get("fact_b", {})),
            data.get("contradiction_type", ""),
            data.get("resolution", "unresolved"),
            data.get("winner", ""),
            data.get("resolved_at"),
            self.user_id,
        ))
        conn.commit()

    def load_all_contradictions(self) -> list[dict]:
        conn = self._get_conn()
        cur = conn.execute("SELECT * FROM contradictions WHERE user_id = ?", (self.user_id,))
        return [
            {
                "fact_a": json.loads(row["fact_a"]),
                "fact_b": json.loads(row["fact_b"]),
                "contradiction_type": row["contradiction_type"],
                "resolution": row["resolution"],
                "winner": row["winner"],
                "resolved_at": row["resolved_at"],
            }
            for row in cur
        ]

    def save_fact_status(self, fact_key: str, status: str):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO fact_status (fact_key, status, user_id)
            VALUES (?, ?, ?)
        """, (fact_key, status, self.user_id))
        conn.commit()

    def load_all_fact_statuses(self) -> dict[str, str]:
        conn = self._get_conn()
        cur = conn.execute("SELECT fact_key, status FROM fact_status WHERE user_id = ?", (self.user_id,))
        return {row["fact_key"]: row["status"] for row in cur}

    # ==================== Explicit Memory ====================

    def save_explicit(self, key: str, data: dict):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO explicit_entries
            (key, value, entry_type, priority, created, updated, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            key,
            json.dumps(data.get("value")),
            data.get("entry_type", "fact"),
            data.get("priority", 0),
            data.get("created", datetime.now(timezone.utc).isoformat()),
            data.get("updated", datetime.now(timezone.utc).isoformat()),
            self.user_id,
        ))
        conn.commit()

    def delete_explicit(self, key: str):
        conn = self._get_conn()
        conn.execute("DELETE FROM explicit_entries WHERE key = ? AND user_id = ?", (key, self.user_id))
        conn.commit()

    def load_all_explicit(self) -> list[dict]:
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM explicit_entries WHERE user_id = ? ORDER BY priority DESC, key",
            (self.user_id,),
        )
        return [
            {
                "key": row["key"],
                "value": json.loads(row["value"]),
                "entry_type": row["entry_type"],
                "priority": row["priority"],
                "created": row["created"],
                "updated": row["updated"],
            }
            for row in cur
        ]

    # ==================== Behavioral Profile ====================

    def save_profile_dimension(self, dimension: str, data: dict):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO profile_dimensions
            (dimension, value, confidence, last_updated, signals, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            dimension,
            data.get("value", 0.5),
            data.get("confidence", 0.1),
            data.get("last_updated", datetime.now(timezone.utc).isoformat()),
            json.dumps(data.get("signals", [])),
            self.user_id,
        ))
        conn.commit()

    def load_all_profile_dimensions(self) -> dict[str, dict]:
        conn = self._get_conn()
        cur = conn.execute(
            "SELECT * FROM profile_dimensions WHERE user_id = ?",
            (self.user_id,),
        )
        return {
            row["dimension"]: {
                "value": row["value"],
                "confidence": row["confidence"],
                "last_updated": row["last_updated"],
                "signals": json.loads(row["signals"]),
            }
            for row in cur
        }

    # ==================== Config ====================

    def save_config(self, key: str, value: Any):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO index_config (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()

    def load_config(self, key: str, default: Any = None) -> Any:
        conn = self._get_conn()
        cur = conn.execute("SELECT value FROM index_config WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    # ==================== Utility ====================

    def delete_user_data(self, user_id: str):
        """Delete ALL data for a user (GDPR compliance)."""
        with self.transaction() as conn:
            for table in [
                "chunks", "entities", "relations", "canonical_entities",
                "contradictions", "fact_status", "explicit_entries", "profile_dimensions",
            ]:
                conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))

    def backup(self, backup_path: str | Path):
        """Create a safe backup using VACUUM INTO."""
        conn = self._get_conn()
        conn.execute(f"VACUUM INTO '{backup_path}'")

    def stats(self) -> dict:
        conn = self._get_conn()
        result = {}
        for table in ["chunks", "entities", "relations", "explicit_entries", "contradictions"]:
            cur = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (self.user_id,))
            result[table] = cur.fetchone()[0]
        return result
