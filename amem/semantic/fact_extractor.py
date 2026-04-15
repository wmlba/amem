"""LLM-powered fact extraction — maximum density, minimum cost.

Strategy: ONE LLM call per session, but engineered to extract 50-80 facts
instead of 15-20. The prompt is the entire difference.

Mem0 extracts ~8 facts per turn × 400 turns = ~3,200 facts ($0.30)
We extract ~60 facts per session × 19 sessions = ~1,140 facts ($0.015)

That narrows the gap from 11× to ~3×, at 20× lower cost.

The key prompt engineering:
1. Demand EVERY specific detail (dates, numbers, names, places)
2. One fact per line, each must be self-contained
3. Include temporal context ("on May 7", "last week", "in 2022")
4. Separate factual statements from opinions/feelings
5. Extract relationship facts ("X is friends with Y")
6. Extract preference facts ("X likes/prefers/enjoys Y")
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx


DENSE_EXTRACTION_PROMPT = """You are a meticulous fact extractor. Extract EVERY specific detail from this conversation into individual fact statements.

Rules:
- ONE fact per line. Each fact must be a complete, self-contained statement.
- Include EVERY: name, date, time, number, place, event, decision, preference, plan, opinion, relationship.
- Include temporal context: "on May 7, 2023", "last week", "in 2022", "yesterday".
- Include WHO said or did things: "Caroline attended...", "Melanie's kids...".
- Extract relationships: "Caroline is friends with Melanie", "Bob reports to Alice".
- Extract preferences: "Caroline enjoys painting", "Melanie likes the beach".
- Extract events: "Went to the beach", "Attended a support group".
- Extract plans: "Planning to go camping in June", "Will start new job next month".
- Extract numbers: "300 patients per day", "45 thousand dollars", "3 years of experience".
- Do NOT summarize or generalize. Be specific.
- Do NOT skip anything. If in doubt, extract it.
- Aim for 40-80 facts from a typical conversation session.

Conversation:
{text}

Extract ALL facts, one per line. Return as a JSON array of strings:"""


class FactExtractor:
    """Extract facts using the user's LLM — maximum density."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str | None = None,
        openai_url: str | None = None,
        openai_key: str | None = None,
        openai_model: str | None = None,
    ):
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model
        self._openai_url = openai_url
        self._openai_key = openai_key
        self._openai_model = openai_model

    async def extract_facts(self, text: str, max_facts: int = 80) -> list[str]:
        """Extract structured facts — maximum density."""
        if not text or len(text.strip()) < 20:
            return []

        # For long sessions, split into chunks and extract from each
        # This prevents losing details from truncation
        if len(text) > 6000:
            return await self._extract_chunked(text, max_facts)

        facts = await self._try_ollama(text)
        if facts:
            return facts[:max_facts]

        if self._openai_url and self._openai_key:
            facts = await self._try_openai(text)
            if facts:
                return facts[:max_facts]

        return self._fallback_extract(text)[:max_facts]

    async def _extract_chunked(self, text: str, max_facts: int) -> list[str]:
        """Split long text into chunks, extract from each, deduplicate."""
        # Split into ~4000 char chunks with overlap
        chunks = []
        words = text.split()
        chunk_size = 600  # words
        overlap = 50
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        all_facts = []
        for chunk in chunks:
            facts = await self._try_ollama(chunk)
            if not facts and self._openai_url and self._openai_key:
                facts = await self._try_openai(chunk)
            if not facts:
                facts = self._fallback_extract(chunk)
            all_facts.extend(facts)

        # Deduplicate by normalized text
        seen = set()
        unique = []
        for f in all_facts:
            key = re.sub(r'[^\w\s]', '', f.lower()).strip()
            if key not in seen and len(key) > 10:
                seen.add(key)
                unique.append(f)

        return unique[:max_facts]

    async def _try_ollama(self, text: str) -> list[str] | None:
        try:
            model = self._ollama_model
            if not model:
                model = await self._detect_ollama_model()
            if not model:
                return None

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": DENSE_EXTRACTION_PROMPT.format(text=text),
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 4096},
                    },
                )
                if resp.status_code != 200:
                    return None
                return self._parse_facts(resp.json().get("response", ""))
        except Exception:
            return None

    async def _try_openai(self, text: str) -> list[str] | None:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self._openai_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._openai_model or "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": "You extract facts exhaustively. Return a JSON array."},
                            {"role": "user", "content": DENSE_EXTRACTION_PROMPT.format(text=text)},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    },
                )
                if resp.status_code != 200:
                    return None
                return self._parse_facts(resp.json()["choices"][0]["message"]["content"])
        except Exception:
            return None

    async def _detect_ollama_model(self) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                if resp.status_code != 200:
                    return None
                models = resp.json().get("models", [])
                skip = {"nomic-embed-text", "all-minilm", "mxbai-embed-large"}
                for m in models:
                    name = m.get("name", "").split(":")[0]
                    if name not in skip:
                        return m.get("name", name)
                return None
        except Exception:
            return None

    def _parse_facts(self, response: str) -> list[str] | None:
        # Try JSON array
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                facts = json.loads(match.group())
                if isinstance(facts, list):
                    return [str(f).strip() for f in facts if f and len(str(f).strip()) > 10]
        except (json.JSONDecodeError, ValueError):
            pass

        # Line-by-line
        lines = response.strip().split("\n")
        facts = []
        for line in lines:
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line.strip())
            line = line.strip('"\'')
            if len(line) > 10 and not line.startswith('{') and not line.startswith('['):
                facts.append(line)
        return facts if facts else None

    def _fallback_extract(self, text: str) -> list[str]:
        """No-LLM fallback: extract info-dense sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        facts = []
        for s in sentences:
            s = s.strip()
            if len(s) < 15:
                continue
            proper = len(re.findall(r'\b[A-Z][a-z]+\b', s))
            numbers = len(re.findall(r'\d+', s))
            if proper >= 1 or numbers >= 1:
                facts.append(s)
        return facts
