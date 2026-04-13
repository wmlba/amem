"""Smart Deduplication — looks at what's DIFFERENT, not just what's similar.

Standard dedup: cosine > 0.95 → duplicate.
Problem: "Working on Kubernetes pod scheduling" and "Working on React state
management" share the template "Working on... implementing feature..." and may
have cosine > 0.95 despite containing DIFFERENT information.

Smart dedup: cosine > threshold AND distinctive tokens overlap.
If two chunks are structurally similar but contain different named entities,
numbers, or technical terms, they are NOT duplicates.

This is like a semantic diff — high overall similarity but different specifics
means the information is distinct and worth keeping.
"""

from __future__ import annotations

import re
from typing import Set

import numpy as np


# Tokens that carry distinctive information (not stopwords, not structure)
_DISTINCTIVE_PATTERN = re.compile(
    r'\b('
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'  # Proper nouns
    r'|\d+(?:\.\d+)?(?:\s*(?:percent|%|ms|MB|GB|TB|k|M|B))?'  # Numbers with units
    r'|[A-Z]{2,}'  # Acronyms
    r'|[a-z]+(?:_[a-z]+)+'  # snake_case identifiers
    r'|[a-z]+(?:[A-Z][a-z]+)+'  # camelCase identifiers
    r')\b'
)

_KNOWN_TECH = {
    'kubernetes', 'docker', 'react', 'python', 'postgresql', 'redis',
    'elasticsearch', 'terraform', 'graphql', 'lambda', 'spark', 'kafka',
    'mongodb', 'nginx', 'pytorch', 'tensorflow', 'fastapi', 'django',
    'flask', 'express', 'nextjs', 'typescript', 'javascript', 'golang',
    'rust', 'java', 'scala', 'swift', 'kotlin', 'github', 'gitlab',
    'aws', 'gcp', 'azure', 'datadog', 'prometheus', 'grafana',
}


def extract_distinctive_tokens(text: str) -> Set[str]:
    """Extract tokens that carry unique information from text."""
    tokens = set()

    # Named entities and patterns
    for m in _DISTINCTIVE_PATTERN.finditer(text):
        tokens.add(m.group().lower())

    # Known tech terms (case-insensitive)
    for word in text.lower().split():
        clean = re.sub(r'[^\w]', '', word)
        if clean in _KNOWN_TECH:
            tokens.add(clean)

    return tokens


def is_true_duplicate(
    text_a: str,
    text_b: str,
    cosine_sim: float,
    cosine_threshold: float = 0.95,
    distinctive_overlap_min: float = 0.5,
) -> bool:
    """Determine if two texts are TRUE duplicates (not just structurally similar).

    Returns True only if:
    1. Cosine similarity exceeds threshold (overall semantic similarity)
    AND
    2. Distinctive tokens overlap sufficiently (same specific content)

    This prevents merging "Working on Kubernetes..." with "Working on React..."
    even when the template structure pushes cosine > 0.95.
    """
    if cosine_sim < cosine_threshold:
        return False

    tokens_a = extract_distinctive_tokens(text_a)
    tokens_b = extract_distinctive_tokens(text_b)

    # If neither has distinctive tokens, fall back to cosine only
    if not tokens_a and not tokens_b:
        return True

    # If one has distinctive tokens and the other doesn't, not a duplicate
    if bool(tokens_a) != bool(tokens_b):
        return False

    # Compute Jaccard overlap of distinctive tokens
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    if union == 0:
        return True

    overlap = intersection / union
    return overlap >= distinctive_overlap_min
