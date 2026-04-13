"""Explicit memory store — Layer 5.

User-controlled typed key-value store. Highest priority, never decays.
Dual-writes: in-memory for fast access, SQLite for durable persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from amem.persistence.sqlite import SQLiteStore


@dataclass
class ExplicitEntry:
    key: str
    value: Any
    entry_type: str = "fact"  # fact, preference, instruction, context
    priority: int = 0  # higher = more important
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "entry_type": self.entry_type,
            "priority": self.priority,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExplicitEntry:
        d = dict(d)
        for field_name in ("created", "updated"):
            if isinstance(d.get(field_name), str):
                d[field_name] = datetime.fromisoformat(d[field_name])
        return cls(**d)


class ExplicitStore:
    """User-controlled explicit memory with typed entries."""

    def __init__(self):
        self._entries: dict[str, ExplicitEntry] = {}
        self._db: SQLiteStore | None = None

    def set_db(self, db: SQLiteStore):
        self._db = db

    def set(
        self,
        key: str,
        value: Any,
        entry_type: str = "fact",
        priority: int = 0,
    ) -> ExplicitEntry:
        """Set or update an explicit memory entry."""
        now = datetime.now(timezone.utc)
        if key in self._entries:
            entry = self._entries[key]
            entry.value = value
            entry.entry_type = entry_type
            entry.priority = priority
            entry.updated = now
        else:
            entry = ExplicitEntry(
                key=key,
                value=value,
                entry_type=entry_type,
                priority=priority,
                created=now,
                updated=now,
            )
            self._entries[key] = entry

        if self._db:
            self._db.save_explicit(key, entry.to_dict())

        return entry

    def get(self, key: str) -> ExplicitEntry | None:
        return self._entries.get(key)

    def delete(self, key: str) -> bool:
        removed = self._entries.pop(key, None) is not None
        if removed and self._db:
            self._db.delete_explicit(key)
        return removed

    def list_all(self) -> list[ExplicitEntry]:
        return sorted(
            self._entries.values(),
            key=lambda e: (-e.priority, e.key),
        )

    def search(self, query: str) -> list[ExplicitEntry]:
        query_lower = query.lower()
        return [
            e for e in self._entries.values()
            if query_lower in e.key.lower() or query_lower in str(e.value).lower()
        ]

    def get_all_for_context(self) -> list[dict]:
        return [e.to_dict() for e in self.list_all()]

    @property
    def count(self) -> int:
        return len(self._entries)

    def save(self, path: Path):
        """Legacy file-based save."""
        path.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries.values()]
        with open(path / "explicit.json", "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        """Legacy file-based load."""
        filepath = path / "explicit.json"
        if not filepath.exists():
            return
        with open(filepath) as f:
            data = json.load(f)
        self._entries.clear()
        for d in data:
            entry = ExplicitEntry.from_dict(d)
            self._entries[entry.key] = entry

    def load_from_db(self):
        """Load from SQLite."""
        if self._db is None:
            return
        self._entries.clear()
        for d in self._db.load_all_explicit():
            entry = ExplicitEntry.from_dict(d)
            self._entries[entry.key] = entry
