"""Turn Enricher — resolve temporal expressions and add attribution in CODE.

No LLM needed. Works with any model. Deterministic.

Before storing a turn, enrich it:
  1. Resolve relative dates → absolute dates using session timestamp
  2. Prepend speaker attribution
  3. Result is a self-contained, searchable fact

Input:  speaker="Caroline", text="I went to a support group yesterday", session_date="8 May, 2023"
Output: "Caroline went to a support group on 7 May 2023"
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from amem.semantic.temporal import TemporalParser, TemporalMarker


_TEMPORAL_PARSER = TemporalParser()

# First-person patterns to rewrite with speaker name
_FIRST_PERSON = re.compile(
    r"\b(I am|I'm|I was|I have|I've|I had|I do|I did|I went|I got|"
    r"I joined|I started|I left|I moved|I quit|I decided|I plan|"
    r"I work|I worked|I lead|I manage|I use|I used|I prefer|"
    r"I live|I lived|I bought|I sold|I made|I found|I lost|"
    r"I think|I feel|I want|I need|I like|I love|I hate|"
    r"I ran|I ran|I attended|I visited|I saw|I met|I called|"
    r"I painted|I cooked|I played|I watched|I read|I wrote|"
    r"my)\b",
    re.IGNORECASE,
)

# Relative time expressions to resolve
_RELATIVE_TIME = re.compile(
    r"\b(yesterday|today|last\s+(?:week|month|year|night|time)|"
    r"this\s+(?:week|month|year|morning|afternoon|evening)|"
    r"the\s+other\s+day|"
    r"(?:a\s+)?(?:few|couple(?:\s+of)?)\s+(?:days?|weeks?|months?)\s+ago|"
    r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago|"
    r"recently|lately|just\s+(?:now|recently))\b",
    re.IGNORECASE,
)


def parse_session_date(date_str: str) -> Optional[datetime]:
    """Parse a session date string into a datetime."""
    if not date_str:
        return None
    # Try common formats
    for fmt in [
        "%I:%M %p on %d %B, %Y",
        "%I:%M %p on %d %B %Y",
        "%d %B, %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
    ]:
        try:
            return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    # Try extracting date portion
    match = re.search(r'(\d{1,2}\s+\w+,?\s+\d{4})', date_str)
    if match:
        for fmt in ["%d %B, %Y", "%d %B %Y"]:
            try:
                return datetime.strptime(match.group(1), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def resolve_relative_date(match_text: str, session_dt: datetime) -> str:
    """Resolve a relative date expression to an absolute date string."""
    lower = match_text.lower().strip()

    if lower == "yesterday":
        dt = session_dt - timedelta(days=1)
    elif lower == "today":
        dt = session_dt
    elif lower == "last night":
        dt = session_dt - timedelta(days=1)
    elif lower == "last week":
        dt = session_dt - timedelta(weeks=1)
    elif lower == "last month":
        if session_dt.month == 1:
            dt = session_dt.replace(year=session_dt.year - 1, month=12, day=15)
        else:
            dt = session_dt.replace(month=session_dt.month - 1, day=15)
    elif lower == "last year":
        dt = session_dt.replace(year=session_dt.year - 1)
    elif lower in ("this week", "this morning", "this afternoon", "this evening"):
        dt = session_dt
    elif lower in ("this month",):
        dt = session_dt.replace(day=1)
    elif lower in ("this year",):
        dt = session_dt
    elif lower in ("recently", "lately", "just now", "just recently", "the other day"):
        dt = session_dt - timedelta(days=3)
    elif "ago" in lower:
        num_match = re.search(r'(\d+)', lower)
        num = int(num_match.group(1)) if num_match else 2
        if "day" in lower:
            dt = session_dt - timedelta(days=num)
        elif "week" in lower:
            dt = session_dt - timedelta(weeks=num)
        elif "month" in lower:
            month = session_dt.month - num
            year = session_dt.year
            while month <= 0:
                month += 12; year -= 1
            dt = session_dt.replace(year=year, month=month, day=min(session_dt.day, 28))
        elif "year" in lower:
            dt = session_dt.replace(year=session_dt.year - num)
        else:
            dt = session_dt - timedelta(days=num)
    elif "few" in lower or "couple" in lower:
        if "day" in lower:
            dt = session_dt - timedelta(days=3)
        elif "week" in lower:
            dt = session_dt - timedelta(weeks=2)
        elif "month" in lower:
            month = session_dt.month - 2
            year = session_dt.year
            if month <= 0: month += 12; year -= 1
            dt = session_dt.replace(year=year, month=month, day=15)
        else:
            dt = session_dt - timedelta(days=3)
    else:
        return match_text  # Can't resolve, return as-is

    return f"on {dt.strftime('%d %B %Y').lstrip('0')}"


def enrich_turn(
    text: str,
    speaker: str = "",
    session_date: str = "",
) -> str:
    """Enrich a conversation turn with attribution and resolved dates.

    This is deterministic, free, works with any LLM or no LLM.

    1. Resolve relative dates → absolute using session date
    2. Replace first-person pronouns with speaker name
    3. Return self-contained, searchable text
    """
    if not text.strip():
        return text

    enriched = text

    # Step 1: Resolve temporal expressions
    session_dt = parse_session_date(session_date)
    if session_dt:
        def replace_temporal(match):
            return resolve_relative_date(match.group(0), session_dt)
        enriched = _RELATIVE_TIME.sub(replace_temporal, enriched)

    # Step 2: Attribution — replace first-person with speaker name
    # Only replace the FIRST "I" occurrence to avoid double-attribution
    if speaker and speaker.lower() not in ("system", "assistant", "fact", ""):
        first_replaced = False
        def replace_first_person(match):
            nonlocal first_replaced
            word = match.group(0)
            lower = word.lower()
            if lower == "my":
                # Only replace "my" if we haven't already attributed
                if first_replaced:
                    return "their"
                return f"{speaker}'s"
            elif not first_replaced:
                first_replaced = True
                if lower in ("i am", "i'm"):
                    return f"{speaker} is"
                elif lower in ("i have", "i've"):
                    return f"{speaker} has"
                elif lower == "i was":
                    return f"{speaker} was"
                elif lower == "i had":
                    return f"{speaker} had"
                elif lower == "i do":
                    return f"{speaker} does"
                elif lower == "i did":
                    return f"{speaker} did"
                else:
                    return speaker + word[1:]
            else:
                return word  # Leave subsequent I/my as-is
        enriched = _FIRST_PERSON.sub(replace_first_person, enriched)

    return enriched
