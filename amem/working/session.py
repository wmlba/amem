"""Working memory — Layer 4.

In-session scratchpad tracking current goals, established facts, and open threads.
Discarded at session end but feeds into episodic store.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class WorkingEntry:
    """A single entry in working memory."""
    entry_id: str
    entry_type: str  # "goal", "fact", "thread", "entity", "note"
    content: str
    created: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """In-session scratchpad for tracking active context."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc)
        self._entries: dict[str, WorkingEntry] = {}
        self._entity_mentions: dict[str, int] = {}

    def add(self, entry_type: str, content: str, metadata: dict | None = None) -> str:
        """Add an entry to working memory."""
        entry_id = str(uuid.uuid4())
        self._entries[entry_id] = WorkingEntry(
            entry_id=entry_id,
            entry_type=entry_type,
            content=content,
            created=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        return entry_id

    def add_goal(self, goal: str) -> str:
        return self.add("goal", goal)

    def add_fact(self, fact: str) -> str:
        return self.add("fact", fact)

    def add_thread(self, thread: str) -> str:
        return self.add("thread", thread)

    def note_entity(self, entity: str):
        """Track entity mention frequency in this session."""
        self._entity_mentions[entity.lower()] = self._entity_mentions.get(entity.lower(), 0) + 1

    def remove(self, entry_id: str) -> bool:
        return self._entries.pop(entry_id, None) is not None

    def get_context(self) -> dict:
        """Get full working memory context for injection."""
        by_type: dict[str, list[str]] = {}
        for entry in self._entries.values():
            by_type.setdefault(entry.entry_type, []).append(entry.content)
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "goals": by_type.get("goal", []),
            "facts": by_type.get("fact", []),
            "open_threads": by_type.get("thread", []),
            "notes": by_type.get("note", []),
            "top_entities": sorted(
                self._entity_mentions.items(),
                key=lambda x: x[1], reverse=True
            )[:10],
        }

    def get_all_text(self) -> str:
        """Get all working memory as text for episodic ingestion at session end."""
        parts = []
        for entry in self._entries.values():
            parts.append(f"[{entry.entry_type}] {entry.content}")
        return "\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def clear(self):
        self._entries.clear()
        self._entity_mentions.clear()
