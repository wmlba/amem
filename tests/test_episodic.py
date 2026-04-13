"""Tests for the episodic store and chunker."""

from __future__ import annotations

import numpy as np
import pytest

from amem.episodic.chunker import SentenceChunker


class TestSentenceChunker:
    def test_basic_chunking(self):
        chunker = SentenceChunker(sentences_per_chunk=2, overlap=0)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert "First sentence" in chunks[0].text

    def test_overlap_chunking(self):
        chunker = SentenceChunker(sentences_per_chunk=2, overlap=1)
        text = "One. Two. Three. Four."
        chunks = chunker.chunk(text)
        # With overlap, more chunks than without
        assert len(chunks) >= 2

    def test_single_sentence(self):
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        text = "Just one sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Just one sentence."

    def test_empty_text(self):
        chunker = SentenceChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_chunk_positions(self):
        chunker = SentenceChunker(sentences_per_chunk=2, overlap=0)
        text = "Alpha sentence. Beta sentence. Gamma sentence."
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert chunk.start_sentence >= 0
            assert chunk.end_sentence >= chunk.start_sentence
