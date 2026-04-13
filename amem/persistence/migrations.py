"""SQLite schema migrations.

Simple, forward-only migrations. Each migration is a SQL string
with a version number. Migrations run on startup if the current
schema version is behind.
"""

from __future__ import annotations

import sqlite3
from typing import List, Tuple

# (version, description, sql)
MIGRATIONS: List[Tuple[int, str, str]] = [
    # Version 1 is the initial schema (created in sqlite.py SCHEMA_SQL)
    # Future migrations go here:
    #
    # (2, "Add embedding column to entities", """
    #     ALTER TABLE entities ADD COLUMN embedding BLOB;
    # """),
    #
    # (3, "Add topic_clusters table", """
    #     CREATE TABLE IF NOT EXISTS topic_clusters (
    #         cluster_id TEXT PRIMARY KEY,
    #         chunk_ids TEXT NOT NULL DEFAULT '[]',
    #         topic_label TEXT NOT NULL DEFAULT '',
    #         created TEXT NOT NULL,
    #         user_id TEXT NOT NULL DEFAULT 'default'
    #     );
    # """),
]


def get_current_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version."""
    try:
        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        row = cur.fetchone()
        return row[0] if row and row[0] else 0
    except sqlite3.OperationalError:
        return 0


def run_migrations(conn: sqlite3.Connection) -> list[int]:
    """Run pending migrations. Returns list of applied version numbers."""
    current = get_current_version(conn)
    applied = []

    for version, description, sql in MIGRATIONS:
        if version > current:
            try:
                conn.executescript(sql)
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (version,),
                )
                conn.commit()
                applied.append(version)
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(
                    f"Migration v{version} ({description}) failed: {e}"
                ) from e

    return applied


def needs_migration(conn: sqlite3.Connection) -> bool:
    """Check if there are pending migrations."""
    current = get_current_version(conn)
    return any(v > current for v, _, _ in MIGRATIONS)
