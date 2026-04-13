"""Accurate token counting via tiktoken.

Falls back to word-count heuristic if tiktoken unavailable.
"""

from __future__ import annotations

from functools import lru_cache

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


@lru_cache(maxsize=1)
def _get_encoder():
    """Get the cl100k_base encoder (used by GPT-4, Claude-compat)."""
    if _HAS_TIKTOKEN:
        return tiktoken.get_encoding("cl100k_base")
    return None


def count_tokens(text: str) -> int:
    """Count tokens accurately using tiktoken, fallback to heuristic."""
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate: words * 4/3. Used as fallback."""
    return max(1, len(text.split()) * 4 // 3)
