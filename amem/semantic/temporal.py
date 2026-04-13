"""Temporal expression parsing for associative memory.

Resolves relative time expressions ("last month", "yesterday", "since 2023")
into absolute timestamps anchored to a reference time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class TemporalMarker:
    """A resolved temporal expression."""
    original_text: str
    resolved_date: datetime
    is_range: bool = False
    range_end: Optional[datetime] = None
    confidence: float = 0.8
    context: str = ""  # surrounding text


# Patterns: (regex, resolver_function_name)
# Each resolver takes (match, reference_time) → TemporalMarker

_PATTERNS = [
    # Explicit dates: "March 2024", "2024-03-15", "Jan 5, 2023"
    (re.compile(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'), "_resolve_iso_date"),
    (re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE), "_resolve_month_year"),
    (re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE), "_resolve_abbrev_date"),

    # Relative days
    (re.compile(r'\byesterday\b', re.IGNORECASE), "_resolve_yesterday"),
    (re.compile(r'\btoday\b', re.IGNORECASE), "_resolve_today"),
    (re.compile(r'\b(\d+)\s+days?\s+ago\b', re.IGNORECASE), "_resolve_n_days_ago"),

    # Relative weeks
    (re.compile(r'\blast\s+week\b', re.IGNORECASE), "_resolve_last_week"),
    (re.compile(r'\b(\d+)\s+weeks?\s+ago\b', re.IGNORECASE), "_resolve_n_weeks_ago"),
    (re.compile(r'\bthis\s+week\b', re.IGNORECASE), "_resolve_this_week"),

    # Relative months
    (re.compile(r'\blast\s+month\b', re.IGNORECASE), "_resolve_last_month"),
    (re.compile(r'\b(\d+)\s+months?\s+ago\b', re.IGNORECASE), "_resolve_n_months_ago"),
    (re.compile(r'\bthis\s+month\b', re.IGNORECASE), "_resolve_this_month"),

    # Relative years
    (re.compile(r'\blast\s+year\b', re.IGNORECASE), "_resolve_last_year"),
    (re.compile(r'\b(\d+)\s+years?\s+ago\b', re.IGNORECASE), "_resolve_n_years_ago"),

    # "Since" expressions
    (re.compile(r'\bsince\s+(\d{4})\b', re.IGNORECASE), "_resolve_since_year"),
    (re.compile(r'\bsince\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+(\d{4}))?\b', re.IGNORECASE), "_resolve_since_month"),

    # "Recently", "lately"
    (re.compile(r'\brecently\b', re.IGNORECASE), "_resolve_recently"),
    (re.compile(r'\blately\b', re.IGNORECASE), "_resolve_recently"),
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


class TemporalParser:
    """Parse temporal expressions in text and resolve to absolute dates."""

    def parse(self, text: str, reference_time: datetime | None = None) -> list[TemporalMarker]:
        """Find and resolve all temporal expressions in text."""
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        markers = []
        for pattern, resolver_name in _PATTERNS:
            resolver = getattr(self, resolver_name)
            for match in pattern.finditer(text):
                try:
                    marker = resolver(match, reference_time)
                    if marker:
                        # Add context (surrounding words)
                        start = max(0, match.start() - 30)
                        end = min(len(text), match.end() + 30)
                        marker.context = text[start:end].strip()
                        markers.append(marker)
                except (ValueError, OverflowError):
                    continue

        return markers

    # ─── Resolvers ───────────────────────────────────────────────────

    def _resolve_iso_date(self, match: re.Match, ref: datetime) -> TemporalMarker:
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        dt = datetime(y, m, d, tzinfo=timezone.utc)
        return TemporalMarker(original_text=match.group(), resolved_date=dt, confidence=0.95)

    def _resolve_month_year(self, match: re.Match, ref: datetime) -> TemporalMarker:
        month = MONTH_MAP[match.group(1).lower()]
        year = int(match.group(2))
        dt = datetime(year, month, 1, tzinfo=timezone.utc)
        return TemporalMarker(
            original_text=match.group(), resolved_date=dt,
            is_range=True, range_end=_end_of_month(dt), confidence=0.9,
        )

    def _resolve_abbrev_date(self, match: re.Match, ref: datetime) -> TemporalMarker:
        month = MONTH_MAP[match.group(1).lower()]
        day = int(match.group(2))
        year = int(match.group(3))
        dt = datetime(year, month, day, tzinfo=timezone.utc)
        return TemporalMarker(original_text=match.group(), resolved_date=dt, confidence=0.95)

    def _resolve_yesterday(self, match: re.Match, ref: datetime) -> TemporalMarker:
        dt = ref - timedelta(days=1)
        return TemporalMarker(original_text="yesterday", resolved_date=dt, confidence=0.9)

    def _resolve_today(self, match: re.Match, ref: datetime) -> TemporalMarker:
        return TemporalMarker(original_text="today", resolved_date=ref, confidence=0.9)

    def _resolve_n_days_ago(self, match: re.Match, ref: datetime) -> TemporalMarker:
        n = int(match.group(1))
        dt = ref - timedelta(days=n)
        return TemporalMarker(original_text=match.group(), resolved_date=dt, confidence=0.85)

    def _resolve_last_week(self, match: re.Match, ref: datetime) -> TemporalMarker:
        dt = ref - timedelta(weeks=1)
        return TemporalMarker(
            original_text="last week", resolved_date=dt,
            is_range=True, range_end=dt + timedelta(days=7), confidence=0.75,
        )

    def _resolve_n_weeks_ago(self, match: re.Match, ref: datetime) -> TemporalMarker:
        n = int(match.group(1))
        dt = ref - timedelta(weeks=n)
        return TemporalMarker(original_text=match.group(), resolved_date=dt, confidence=0.8)

    def _resolve_this_week(self, match: re.Match, ref: datetime) -> TemporalMarker:
        start = ref - timedelta(days=ref.weekday())
        return TemporalMarker(
            original_text="this week", resolved_date=start,
            is_range=True, range_end=ref, confidence=0.8,
        )

    def _resolve_last_month(self, match: re.Match, ref: datetime) -> TemporalMarker:
        if ref.month == 1:
            dt = ref.replace(year=ref.year - 1, month=12, day=1)
        else:
            dt = ref.replace(month=ref.month - 1, day=1)
        return TemporalMarker(
            original_text="last month", resolved_date=dt,
            is_range=True, range_end=_end_of_month(dt), confidence=0.75,
        )

    def _resolve_n_months_ago(self, match: re.Match, ref: datetime) -> TemporalMarker:
        n = int(match.group(1))
        year = ref.year
        month = ref.month - n
        while month <= 0:
            month += 12
            year -= 1
        dt = datetime(year, month, 1, tzinfo=timezone.utc)
        return TemporalMarker(original_text=match.group(), resolved_date=dt, confidence=0.8)

    def _resolve_this_month(self, match: re.Match, ref: datetime) -> TemporalMarker:
        dt = ref.replace(day=1)
        return TemporalMarker(
            original_text="this month", resolved_date=dt,
            is_range=True, range_end=ref, confidence=0.8,
        )

    def _resolve_last_year(self, match: re.Match, ref: datetime) -> TemporalMarker:
        dt = ref.replace(year=ref.year - 1, month=1, day=1)
        return TemporalMarker(
            original_text="last year", resolved_date=dt,
            is_range=True, range_end=datetime(ref.year - 1, 12, 31, tzinfo=timezone.utc),
            confidence=0.7,
        )

    def _resolve_n_years_ago(self, match: re.Match, ref: datetime) -> TemporalMarker:
        n = int(match.group(1))
        dt = ref.replace(year=ref.year - n, month=1, day=1)
        return TemporalMarker(original_text=match.group(), resolved_date=dt, confidence=0.75)

    def _resolve_since_year(self, match: re.Match, ref: datetime) -> TemporalMarker:
        year = int(match.group(1))
        dt = datetime(year, 1, 1, tzinfo=timezone.utc)
        return TemporalMarker(
            original_text=match.group(), resolved_date=dt,
            is_range=True, range_end=ref, confidence=0.85,
        )

    def _resolve_since_month(self, match: re.Match, ref: datetime) -> TemporalMarker:
        month = MONTH_MAP[match.group(1).lower()]
        year = int(match.group(2)) if match.group(2) else ref.year
        dt = datetime(year, month, 1, tzinfo=timezone.utc)
        return TemporalMarker(
            original_text=match.group(), resolved_date=dt,
            is_range=True, range_end=ref, confidence=0.8,
        )

    def _resolve_recently(self, match: re.Match, ref: datetime) -> TemporalMarker:
        dt = ref - timedelta(days=14)
        return TemporalMarker(
            original_text=match.group(), resolved_date=dt,
            is_range=True, range_end=ref, confidence=0.5,
        )


def _end_of_month(dt: datetime) -> datetime:
    """Get the last day of the month."""
    if dt.month == 12:
        return datetime(dt.year, 12, 31, tzinfo=timezone.utc)
    return datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
