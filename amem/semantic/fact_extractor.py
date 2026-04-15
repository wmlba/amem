"""LLM-powered fact extraction — uses whatever model the user has.

Called ONCE per session end (not per turn). Extracts structured facts
from the session's conversation, then stores each fact as a
searchable memory in the episodic index.

Supports: Ollama (local), OpenAI, any OpenAI-compatible endpoint.
Falls back gracefully if no LLM available.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx


EXTRACTION_PROMPT = """Extract all important facts from this conversation as a JSON array.
Each fact should be a single, self-contained statement that someone might want to recall later.
Include: who did what, when, where, specific numbers, decisions made, preferences stated, plans discussed.
Include dates and times if mentioned.

Conversation:
{text}

Return ONLY a JSON array of strings. Example:
["Alice works as an ML engineer at Google", "The project deadline is May 17th", "They decided to use PyTorch instead of TensorFlow"]

Facts:"""


class FactExtractor:
    """Extract facts from conversation text using the user's LLM.

    Tries (in order):
    1. Ollama (local, free)
    2. OpenAI-compatible endpoint (if configured)
    3. Fallback: simple sentence extraction (no LLM)
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str | None = None,
        openai_url: str | None = None,
        openai_key: str | None = None,
        openai_model: str | None = None,
    ):
        self._ollama_url = ollama_url
        self._ollama_model = ollama_model  # Auto-detect if None
        self._openai_url = openai_url
        self._openai_key = openai_key
        self._openai_model = openai_model

    async def extract_facts(self, text: str, max_facts: int = 30) -> list[str]:
        """Extract structured facts from conversation text.

        Returns list of fact strings, each a self-contained statement.
        """
        if not text or len(text.strip()) < 20:
            return []

        # Truncate very long text to avoid context limits
        if len(text) > 8000:
            text = text[:8000]

        # Try Ollama first
        facts = await self._try_ollama(text)
        if facts:
            return facts[:max_facts]

        # Try OpenAI-compatible
        if self._openai_url and self._openai_key:
            facts = await self._try_openai(text)
            if facts:
                return facts[:max_facts]

        # Fallback: extractive
        return self._fallback_extract(text)[:max_facts]

    async def _try_ollama(self, text: str) -> list[str] | None:
        """Try extracting facts via Ollama."""
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
                        "prompt": EXTRACTION_PROMPT.format(text=text),
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 2048},
                    },
                )
                if resp.status_code != 200:
                    return None

                response_text = resp.json().get("response", "")
                return self._parse_facts(response_text)
        except (httpx.ConnectError, httpx.TimeoutException, Exception):
            return None

    async def _try_openai(self, text: str) -> list[str] | None:
        """Try extracting facts via OpenAI-compatible endpoint."""
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
                            {"role": "system", "content": "You extract facts from conversations. Return only a JSON array of fact strings."},
                            {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2048,
                    },
                )
                if resp.status_code != 200:
                    return None

                content = resp.json()["choices"][0]["message"]["content"]
                return self._parse_facts(content)
        except Exception:
            return None

    async def _detect_ollama_model(self) -> str | None:
        """Find a chat-capable model on Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._ollama_url}/api/tags")
                if resp.status_code != 200:
                    return None
                models = resp.json().get("models", [])
                # Prefer chat models (skip embedding-only models)
                skip = {"nomic-embed-text", "all-minilm", "mxbai-embed-large"}
                for m in models:
                    name = m.get("name", "").split(":")[0]
                    if name not in skip:
                        return m.get("name", name)
                return None
        except Exception:
            return None

    def _parse_facts(self, response: str) -> list[str] | None:
        """Parse LLM response into list of fact strings."""
        # Try JSON parsing
        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                facts = json.loads(match.group())
                if isinstance(facts, list):
                    return [str(f).strip() for f in facts if f and len(str(f).strip()) > 5]
        except (json.JSONDecodeError, ValueError):
            pass

        # Try line-by-line parsing
        lines = response.strip().split("\n")
        facts = []
        for line in lines:
            line = re.sub(r'^[\d\-\*\.]+\s*', '', line.strip())  # Strip list markers
            line = line.strip('"\'')
            if len(line) > 10 and not line.startswith('{') and not line.startswith('['):
                facts.append(line)
        return facts if facts else None

    def _fallback_extract(self, text: str) -> list[str]:
        """No-LLM fallback: extract sentences that look like facts."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        facts = []
        for s in sentences:
            s = s.strip()
            if len(s) < 15:
                continue
            # Score: contains proper nouns, numbers, or specific details
            proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', s))
            numbers = len(re.findall(r'\d+', s))
            if proper_nouns >= 1 or numbers >= 1:
                facts.append(s)
        return facts
