"""LLM-assisted entity and relation extraction via Ollama.

Replaces heuristic regex NER with structured LLM extraction for dramatically
higher accuracy on real conversational text. Falls back to heuristic if
Ollama is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import httpx

from amem.config import OllamaConfig
from amem.semantic.extractor import EntityExtractor, ExtractionResult, ExtractedEntity, ExtractedRelation


EXTRACTION_PROMPT = """Extract all entities and relationships from the following text. Return valid JSON only.

Rules:
- Entities: people, organizations, tools/technologies, projects, concepts, locations
- Relations: subject-predicate-object triples (e.g., "Alice works_on ML Pipeline")
- Common predicates: works_on, works_at, leads, uses, researches, manages, is_a, reports_to, prefers, located_in, created, part_of
- Include confidence (0.0-1.0) for each relation based on how explicitly it's stated
- Extract temporal markers if present (e.g., "since 2023", "last month", "recently")

Return format:
{
  "entities": [
    {"name": "string", "type": "person|org|tool|project|concept|location", "attributes": {}}
  ],
  "relations": [
    {"subject": "string", "predicate": "string", "object": "string", "confidence": 0.8}
  ],
  "temporal_markers": [
    {"text": "last month", "entity": "string", "context": "string"}
  ]
}

Text to extract from:
"""


class LLMExtractor:
    """LLM-powered entity/relation extraction with caching and fallback.

    Uses Ollama chat models to extract structured knowledge from text.
    Falls back to regex-based EntityExtractor if Ollama is unavailable.
    """

    def __init__(
        self,
        config: OllamaConfig | None = None,
        model: str | None = None,
        cache_enabled: bool = True,
    ):
        self._config = config or OllamaConfig()
        # Use a chat model for extraction, not the embedding model
        self._model = model or "llama3.2"
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=120.0,
        )
        self._fallback = EntityExtractor()
        self._cache: dict[str, ExtractionResult] = {}
        self._cache_enabled = cache_enabled
        self._available: bool | None = None  # None = unknown, check on first call

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relations using LLM, with cache and fallback."""
        if not text or not text.strip():
            return ExtractionResult()

        # Check cache
        if self._cache_enabled:
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Try LLM extraction
        if self._available is not False:
            try:
                result = await self._llm_extract(text)
                self._available = True
                if self._cache_enabled:
                    self._cache[cache_key] = result
                return result
            except (httpx.HTTPError, httpx.ConnectError, json.JSONDecodeError, KeyError) as e:
                self._available = False
                # Fall through to heuristic

        # Fallback to heuristic extraction
        return self._fallback.extract(text)

    async def _llm_extract(self, text: str) -> ExtractionResult:
        """Call Ollama to extract structured knowledge."""
        prompt = EXTRACTION_PROMPT + text

        resp = await self._client.post(
            "/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2048,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "")

        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> ExtractionResult:
        """Parse LLM JSON response into ExtractionResult."""
        # Try to extract JSON from the response
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                parsed = json.loads(match.group())
            else:
                return ExtractionResult()

        entities = []
        for ent in parsed.get("entities", []):
            if isinstance(ent, dict) and "name" in ent:
                entities.append(ExtractedEntity(
                    name=ent["name"],
                    entity_type=ent.get("type", "concept"),
                    mentions=1,
                ))

        relations = []
        for rel in parsed.get("relations", []):
            if isinstance(rel, dict) and all(k in rel for k in ("subject", "predicate", "object")):
                relations.append(ExtractedRelation(
                    subject=rel["subject"],
                    predicate=self._normalize_predicate(rel["predicate"]),
                    object=rel["object"],
                    confidence=float(rel.get("confidence", 0.8)),
                ))

        return ExtractionResult(entities=entities, relations=relations)

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate to snake_case standard form."""
        pred = predicate.lower().strip()
        pred = re.sub(r'\s+', '_', pred)
        # Map common variations
        mappings = {
            "works_on": "works_on",
            "working_on": "works_on",
            "works_at": "works_at",
            "employed_at": "works_at",
            "employed_by": "works_at",
            "leads": "leads",
            "leading": "leads",
            "manages": "manages",
            "managing": "manages",
            "uses": "uses",
            "using": "uses",
            "researches": "researches",
            "researching": "researches",
            "is_a": "is_a",
            "is_an": "is_a",
            "reports_to": "reports_to",
            "located_in": "located_in",
            "based_in": "located_in",
            "part_of": "part_of",
            "member_of": "part_of",
            "created": "created",
            "built": "created",
            "prefers": "prefers",
            "likes": "prefers",
            "left": "left",
            "quit": "left",
            "stopped_working_on": "stopped_working_on",
        }
        return mappings.get(pred, pred)

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def check_availability(self) -> bool:
        """Check if Ollama is available with the configured model."""
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            self._available = self._model.split(":")[0] in models
            return self._available
        except (httpx.HTTPError, httpx.ConnectError):
            self._available = False
            return False

    async def close(self):
        await self._client.aclose()

    @property
    def is_available(self) -> bool | None:
        return self._available

    @property
    def cache_size(self) -> int:
        return len(self._cache)
