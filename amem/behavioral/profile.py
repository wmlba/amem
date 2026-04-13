"""User behavioral profile — Layer 3.

Tracks how to interact (tone, depth, formality) based on conversation signals.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from amem.persistence.sqlite import SQLiteStore


@dataclass
class DimensionEstimate:
    """Estimate for a single behavioral dimension."""
    value: float = 0.5  # 0.0 to 1.0
    confidence: float = 0.1  # 0.0 to 1.0, grows with more signals
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def update(self, signal: float, weight: float = 0.2):
        """Exponential moving average update."""
        self.signal_history.append(signal)
        self.value = self.value * (1 - weight) + signal * weight
        n = len(self.signal_history)
        self.confidence = min(1.0, n / 20.0)
        self.last_updated = datetime.now(timezone.utc)
        self._dirty = True

    _dirty: bool = False

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat(),
            "n_signals": len(self.signal_history),
        }


class UserProfile:
    """Behavioral model tracking interaction preferences.

    Dimensions:
        response_depth: 0=brief → 1=deep/technical
        formality: 0=casual → 1=formal
        domain_expertise: 0=beginner → 1=expert
        verbosity_preference: 0=terse → 1=verbose
    """

    DIMENSIONS = ["response_depth", "formality", "domain_expertise", "verbosity_preference"]

    def __init__(self):
        self._dimensions: dict[str, DimensionEstimate] = {
            d: DimensionEstimate() for d in self.DIMENSIONS
        }
        self._db: "SQLiteStore | None" = None

    def set_db(self, db: "SQLiteStore"):
        self._db = db

    def _persist_dirty(self):
        """Write dirty dimensions to SQLite."""
        if self._db is None:
            return
        for name, est in self._dimensions.items():
            if est._dirty:
                self._db.save_profile_dimension(name, {
                    "value": est.value,
                    "confidence": est.confidence,
                    "last_updated": est.last_updated.isoformat(),
                    "signals": list(est.signal_history),
                })
                est._dirty = False

    def load_from_db(self):
        """Load profile from SQLite."""
        if self._db is None:
            return
        data = self._db.load_all_profile_dimensions()
        for name, vals in data.items():
            if name in self._dimensions:
                est = self._dimensions[name]
                est.value = vals["value"]
                est.confidence = vals["confidence"]
                est.last_updated = datetime.fromisoformat(vals["last_updated"])
                est.signal_history = deque(vals.get("signals", []), maxlen=50)
                est._dirty = False

    def update_from_message(self, message: str, role: str = "user"):
        """Extract behavioral signals from a message and update estimates."""
        if role != "user":
            return

        words = message.split()
        word_count = len(words)

        # Response depth: longer messages with technical terms → deeper
        tech_terms = len(re.findall(
            r'\b(function|class|method|API|database|query|model|vector|'
            r'algorithm|architecture|latency|throughput|config|deploy|'
            r'cluster|node|index|schema|pipeline|endpoint)\b',
            message, re.IGNORECASE
        ))
        depth_signal = min(1.0, (word_count / 200.0) * 0.5 + (tech_terms / 5.0) * 0.5)
        self._dimensions["response_depth"].update(depth_signal)

        # Formality: contractions, slang → low; full sentences → high
        contractions = len(re.findall(r"\w+'\w+", message))
        sentences = len(re.findall(r'[.!?]+', message)) or 1
        avg_sentence_len = word_count / sentences
        formality_signal = min(1.0, max(0.0,
            (avg_sentence_len / 25.0) * 0.5 - (contractions / 3.0) * 0.3 + 0.3
        ))
        self._dimensions["formality"].update(formality_signal)

        # Domain expertise: technical vocabulary density
        expertise_signal = min(1.0, tech_terms / max(word_count, 1) * 10)
        self._dimensions["domain_expertise"].update(expertise_signal)

        # Verbosity preference: raw word count
        verbosity_signal = min(1.0, word_count / 300.0)
        self._dimensions["verbosity_preference"].update(verbosity_signal)

        self._persist_dirty()

    def update_from_feedback(self, feedback_type: str, value: float):
        """Direct update from explicit user feedback.

        feedback_type: one of DIMENSIONS
        value: 0.0 to 1.0
        """
        if feedback_type in self._dimensions:
            self._dimensions[feedback_type].update(value, weight=0.5)
            self._persist_dirty()

    def get_priors(self) -> dict[str, dict]:
        """Get current behavioral priors for all dimensions."""
        return {name: est.to_dict() for name, est in self._dimensions.items()}

    def get_summary(self) -> dict[str, str]:
        """Human-readable summary of behavioral profile."""
        labels = {
            "response_depth": {0.0: "brief", 0.3: "moderate", 0.7: "detailed", 0.9: "deep/technical"},
            "formality": {0.0: "very casual", 0.3: "casual", 0.6: "semi-formal", 0.8: "formal"},
            "domain_expertise": {0.0: "beginner", 0.3: "intermediate", 0.6: "advanced", 0.8: "expert"},
            "verbosity_preference": {0.0: "terse", 0.3: "concise", 0.6: "moderate", 0.8: "verbose"},
        }

        summary = {}
        for name, est in self._dimensions.items():
            label_map = labels.get(name, {})
            label = "unknown"
            for threshold in sorted(label_map.keys(), reverse=True):
                if est.value >= threshold:
                    label = label_map[threshold]
                    break
            summary[name] = f"{label} ({est.value:.2f}, confidence: {est.confidence:.2f})"
        return summary

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, est in self._dimensions.items():
            data[name] = {
                "value": est.value,
                "confidence": est.confidence,
                "last_updated": est.last_updated.isoformat(),
                "signals": list(est.signal_history),
            }
        with open(path / "profile.json", "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        profile_file = path / "profile.json"
        if not profile_file.exists():
            return
        with open(profile_file) as f:
            data = json.load(f)
        for name, vals in data.items():
            if name in self._dimensions:
                est = self._dimensions[name]
                est.value = vals["value"]
                est.confidence = vals["confidence"]
                est.last_updated = datetime.fromisoformat(vals["last_updated"])
                est.signal_history = deque(vals.get("signals", []), maxlen=50)
