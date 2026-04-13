"""Text chunking for episodic memory ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A text chunk with positional info."""
    text: str
    start_sentence: int
    end_sentence: int
    char_offset: int
    char_length: int


class SentenceChunker:
    """Split text into overlapping sentence-based chunks."""

    # Sentence boundary pattern: period/exclamation/question followed by space or end
    _SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$')

    def __init__(self, sentences_per_chunk: int = 3, overlap: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Use regex to split on sentence boundaries
        sentences = self._SENTENCE_RE.split(text.strip())
        # Filter empty
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk text into overlapping sentence groups."""
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        chunks = []
        step = max(1, self.sentences_per_chunk - self.overlap)
        char_pos = 0

        for i in range(0, len(sentences), step):
            window = sentences[i:i + self.sentences_per_chunk]
            if not window:
                break

            chunk_text = " ".join(window)
            # Find char offset in original text
            offset = text.find(window[0][0:20], char_pos) if window[0] else char_pos
            if offset == -1:
                offset = char_pos

            chunks.append(Chunk(
                text=chunk_text,
                start_sentence=i,
                end_sentence=i + len(window) - 1,
                char_offset=max(0, offset),
                char_length=len(chunk_text),
            ))

            # Don't create a tiny trailing chunk
            if i + self.sentences_per_chunk >= len(sentences):
                break

        return chunks
