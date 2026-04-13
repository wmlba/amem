"""Intent-Aware Dynamic Scoring — a novel approach.

No existing vector DB does this. FAISS/Qdrant/Weaviate use fixed scoring.
We detect the INTENT behind a query and dynamically reweight the scoring
factors to match what the user actually wants.

"What's my CURRENT research conclusion?" → boost temporal (recency matters)
"What did Team Beta discuss?" → boost context matching (specificity matters)
"What's the exact training config?" → boost similarity (precision matters)
"Tell me everything about Alice" → balanced (breadth matters)

The weights shift per-query based on signal words, not per-system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ScoringIntent:
    """Dynamic scoring weights derived from query intent analysis."""
    similarity_weight: float = 0.75
    temporal_weight: float = 0.15
    reinforcement_weight: float = 0.05
    importance_weight: float = 0.05
    context_boost_terms: list = None  # terms to boost in conversation_id/entity matching

    def __post_init__(self):
        if self.context_boost_terms is None:
            self.context_boost_terms = []


# ─── Recency Signals ─────────────────────────────────────────────────
# When query contains these, the user wants RECENT information.
_RECENCY_PATTERNS = re.compile(
    r'\b(current|latest|recent|now|today|this week|this month|'
    r'updated|newest|last|most recent|right now|at the moment|'
    r'currently|presently|ongoing|active)\b',
    re.IGNORECASE,
)

# ─── Precision Signals ───────────────────────────────────────────────
# When query contains these, the user wants EXACT matches.
_PRECISION_PATTERNS = re.compile(
    r'\b(exact|specific|precisely|details|configuration|config|'
    r'number|value|parameter|setting|how much|how many|'
    r'what was the|verbatim|literal)\b',
    re.IGNORECASE,
)

# ─── Breadth Signals ─────────────────────────────────────────────────
# When query contains these, the user wants BROAD coverage.
_BREADTH_PATTERNS = re.compile(
    r'\b(everything|all|overview|summary|tell me about|'
    r'what do you know|background|history|context)\b',
    re.IGNORECASE,
)

# ─── Context Anchor Signals ──────────────────────────────────────────
# Named groups, teams, projects — boost matching conversation contexts.
_CONTEXT_ANCHOR_PATTERN = re.compile(
    r'\b(?:team|project|session|meeting|sprint|channel)\s+(\w+)',
    re.IGNORECASE,
)

# Proper nouns that might be context anchors
_PROPER_NOUN_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')


def analyze_intent(query: str) -> ScoringIntent:
    """Analyze a query and produce dynamic scoring weights.

    This is the novel part: instead of fixed weights, we read the
    user's intent from their word choice and adjust scoring to match.
    """
    recency_hits = len(_RECENCY_PATTERNS.findall(query))
    precision_hits = len(_PRECISION_PATTERNS.findall(query))
    breadth_hits = len(_BREADTH_PATTERNS.findall(query))

    # Start with defaults
    sim_w = 0.75
    temp_w = 0.15
    reinf_w = 0.05
    imp_w = 0.05

    # Recency intent: "What's my CURRENT status?"
    if recency_hits > 0:
        boost = min(0.30, recency_hits * 0.15)
        temp_w += boost
        sim_w -= boost

    # Precision intent: "What's the EXACT learning rate?"
    if precision_hits > 0:
        boost = min(0.15, precision_hits * 0.08)
        sim_w += boost
        temp_w -= boost * 0.5
        imp_w -= boost * 0.5

    # Breadth intent: "Tell me EVERYTHING about the project"
    if breadth_hits > 0:
        # Flatten all weights — don't over-prioritize any factor
        sim_w = 0.50
        temp_w = 0.20
        reinf_w = 0.15
        imp_w = 0.15

    # Context anchors: "Team Beta", "Project Phoenix"
    context_terms = []
    for m in _CONTEXT_ANCHOR_PATTERN.finditer(query):
        context_terms.append(m.group(1).lower())

    # Also extract proper nouns as potential context anchors
    _SKIP_WORDS = {'what', 'when', 'where', 'which', 'who', 'how', 'why',
                   'the', 'this', 'that', 'from', 'with', 'about', 'tell',
                   'does', 'did', 'has', 'have', 'can', 'could', 'would',
                   'should', 'will', 'may', 'might'}
    for m in _PROPER_NOUN_PATTERN.finditer(query):
        name = m.group(1)
        if len(name) > 2 and name.lower() not in _SKIP_WORDS:
            context_terms.append(name.lower())

    # Clamp weights to valid range
    total = sim_w + temp_w + reinf_w + imp_w
    if total > 0:
        sim_w /= total
        temp_w /= total
        reinf_w /= total
        imp_w /= total

    return ScoringIntent(
        similarity_weight=sim_w,
        temporal_weight=temp_w,
        reinforcement_weight=reinf_w,
        importance_weight=imp_w,
        context_boost_terms=context_terms,
    )
