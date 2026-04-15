"""Cross-layer retrieval orchestrator.

Queries all memory layers, merges, deduplicates, ranks, and budget-constrains
the results into a single MemoryContext for injection.

Features:
- Dynamic budget allocation across layers based on relevance signals
- Behavioral feedback loop — profile modulates context format and depth
- Entity resolution across layers
- Contradiction-aware fact filtering
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from amem.config import Config
from amem.utils.tokenizer import count_tokens
from amem.embeddings.base import EmbeddingProvider
from amem.episodic.store import EpisodicStore
from amem.semantic.graph import SemanticGraph
from amem.semantic.extractor import EntityExtractor
from amem.semantic.decay import ConfidenceDecay
from amem.semantic.contradictions import FactStatus
from amem.behavioral.profile import UserProfile
from amem.working.session import WorkingMemory
from amem.explicit.store import ExplicitStore
from amem.persistence.sqlite import SQLiteStore
from amem.retrieval.intent import analyze_intent
from amem.retrieval.knowledge_assembler import KnowledgeContext, assemble_knowledge
from amem.maintenance.summarizer import summarize_session, summarize_session_simple


@dataclass
class LayerBudget:
    """Token budget allocation for each layer."""
    episodic: int = 0
    semantic: int = 0
    explicit: int = 0
    behavioral: int = 0
    working: int = 0

    @property
    def total(self) -> int:
        return self.episodic + self.semantic + self.explicit + self.behavioral + self.working


@dataclass
class MemoryContext:
    """Assembled memory context ready for injection into a conversation."""
    episodic_chunks: list[dict] = field(default_factory=list)
    semantic_facts: list[dict] = field(default_factory=list)
    behavioral_priors: dict[str, dict] = field(default_factory=dict)
    explicit_entries: list[dict] = field(default_factory=list)
    working_context: dict = field(default_factory=dict)
    contradictions: list[dict] = field(default_factory=list)
    total_tokens_estimate: int = 0
    budget_allocation: dict = field(default_factory=dict)

    def to_injection_text(self, profile: UserProfile | None = None) -> str:
        """Format as text for context window injection.

        Uses behavioral profile to modulate verbosity and depth.
        """
        parts = []
        verbose = True
        if profile:
            priors = profile.get_priors()
            verbose = priors.get("verbosity_preference", {}).get("value", 0.5) > 0.4

        # Explicit — always first, highest priority
        if self.explicit_entries:
            parts.append("## Explicit Memory (User-Defined)")
            for e in self.explicit_entries:
                parts.append(f"- [{e['entry_type']}] {e['key']}: {e['value']}")

        # Semantic facts
        if self.semantic_facts:
            parts.append("\n## Known Facts")
            for f in self.semantic_facts:
                status_tag = ""
                if f.get("status") == FactStatus.SUPERSEDED.value:
                    status_tag = " [SUPERSEDED]"
                elif f.get("status") == FactStatus.CONFLICTED.value:
                    status_tag = " [CONFLICTED]"
                conf = f['confidence']
                line = f"- {f['subject']} —[{f['predicate']}]→ {f['object']}"
                if verbose:
                    line += f" (confidence: {conf}, mentions: {f.get('mention_count', 1)}){status_tag}"
                parts.append(line)

        # Contradictions — surface unresolved ones
        if self.contradictions:
            parts.append("\n## Unresolved Contradictions")
            for c in self.contradictions:
                fa = c.get("fact_a", {})
                fb = c.get("fact_b", {})
                parts.append(
                    f"- CONFLICT: \"{fa.get('subject')} {fa.get('predicate')} {fa.get('object')}\" "
                    f"vs \"{fb.get('subject')} {fb.get('predicate')} {fb.get('object')}\" "
                    f"({c.get('contradiction_type', 'unknown')})"
                )

        # Episodic chunks
        if self.episodic_chunks:
            parts.append("\n## Relevant Past Context")
            max_text_len = 300 if verbose else 150
            for c in self.episodic_chunks:
                score = c.get('score', 0)
                text = c['text'][:max_text_len]
                if len(c.get('text', '')) > max_text_len:
                    text += '...'
                line = f"- [{c.get('conversation_id', '?')[:8]}] {text}"
                if verbose:
                    line += f" (relevance: {score:.3f})"
                parts.append(line)

        # Working memory
        wc = self.working_context
        if wc and (wc.get("goals") or wc.get("facts") or wc.get("open_threads")):
            parts.append("\n## Current Session")
            if wc.get("goals"):
                parts.append("Goals: " + "; ".join(wc["goals"]))
            if wc.get("facts"):
                parts.append("Established: " + "; ".join(wc["facts"]))
            if wc.get("open_threads"):
                parts.append("Open threads: " + "; ".join(wc["open_threads"]))

        # Behavioral profile — compact summary
        if self.behavioral_priors and verbose:
            parts.append("\n## User Profile")
            for dim, data in self.behavioral_priors.items():
                conf = data.get('confidence', 0)
                if conf > 0.2:  # only show dimensions with enough signal
                    parts.append(f"- {dim}: {data.get('value', 0.5):.2f} (confidence: {conf:.2f})")

        text = "\n".join(parts)
        self.total_tokens_estimate = count_tokens(text)
        return text


class MemoryOrchestrator:
    """Central query pipeline across all memory layers.

    Features:
    - Dynamic per-layer budget allocation based on signal quality
    - Behavioral feedback modulates context injection
    - Entity resolution across layers
    - Contradiction-aware retrieval
    """

    @classmethod
    def from_config(cls, config: Config, user_id: str = "default") -> "MemoryOrchestrator":
        """Create an orchestrator from config using the embedding factory.

        This is the recommended way to create an orchestrator. The factory
        auto-detects the best available embedding provider, or uses the
        one specified in config.embedding.

        Usage:
            config = load_config("config.yaml")
            orch = MemoryOrchestrator.from_config(config)
        """
        from amem.embeddings.factory import create_embedder
        # Try new unified config first, fall back to legacy Ollama config
        if config.embedding.provider != "auto" or config.embedding.model:
            embedder = create_embedder(config.embedding)
        else:
            embedder = create_embedder(config.ollama)
        return cls(embedder, config, user_id)

    def __init__(self, embedder: EmbeddingProvider, config: Config, user_id: str = "default"):
        self._embedder = embedder
        self._config = config
        self._user_id = user_id
        self._write_lock: asyncio.Lock | None = None

        # Initialize layers
        self.episodic = EpisodicStore(embedder, config)
        self.semantic = SemanticGraph(
            decay=ConfidenceDecay(
                decay_lambda=config.semantic.decay_lambda,
                min_confidence=config.semantic.min_confidence,
            )
        )
        self.behavioral = UserProfile()
        self.working = WorkingMemory()
        self.explicit = ExplicitStore()

        self._extractor = EntityExtractor()
        self._db: SQLiteStore | None = None

        # Session summaries
        self._session_summaries: list[dict] = []

        # Session conversation buffer (collects turns for end-of-session extraction)
        self._session_turns: list[str] = []

        # LLM-powered fact extractor (uses whatever model the user has)
        from amem.semantic.fact_extractor import FactExtractor
        from amem.semantic.selective_extractor import SelectiveExtractor
        import os
        self._fact_extractor = FactExtractor(
            ollama_url=config.ollama.base_url,
            openai_url="https://api.openai.com/v1" if os.environ.get("OPENAI_API_KEY") else None,
            openai_key=os.environ.get("OPENAI_API_KEY"),
        )

        # NOVEL: Embedding-Gated Selective Extraction
        # Uses the same embedding model to decide WHICH turns need LLM extraction.
        # Phatic turns ("Hey!") → skip. Novel factual turns → extract.
        # ~12-16× fewer LLM calls than per-turn, ~85-90% fact coverage.
        self._selective_extractor = SelectiveExtractor(
            embedder=embedder,
            fact_extractor=self._fact_extractor,
            novelty_threshold=0.75,
            batch_size=5,
        )

        # Wire embedding-powered extractor (same model as episodic store)
        from amem.semantic.embedding_extractor import EmbeddingExtractor
        self._embedding_extractor = EmbeddingExtractor(embedder)
        self.semantic.set_embedding_extractor(self._embedding_extractor)

    def init_db(self, db_path: Path | None = None):
        """Initialize SQLite backend and wire to all layers."""
        if db_path is None:
            db_path = Path(self._config.storage.data_dir) / "amem.db"
        self._db = SQLiteStore(db_path, user_id=self._user_id)
        self.episodic.set_db(self._db)
        self.explicit.set_db(self._db)
        self.behavioral.set_db(self._db)
        self.semantic.set_db(self._db)

    def load_from_db(self):
        """Load all layers from SQLite."""
        if self._db is None:
            return
        self.episodic.load_from_db()
        self.explicit.load_from_db()
        self.behavioral.load_from_db()
        self.semantic.load_from_db()

    async def ingest(
        self,
        text: str,
        conversation_id: str | None = None,
        speaker: str = "",
        timestamp=None,
    ) -> dict:
        """Ingest text into all relevant layers. Thread-safe via write lock."""
        if not text or not text.strip():
            return {"chunks_stored": 0, "entities_extracted": 0, "relations_extracted": 0, "contradictions": []}

        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        async with self._write_lock:
            return await self._ingest_unsafe(text, conversation_id, speaker, timestamp)

    async def _ingest_unsafe(self, text, conversation_id, speaker, timestamp) -> dict:

        # Buffer turn text for session-end extraction (fallback)
        self._session_turns.append(text)

        # Episodic: chunk, embed, store (raw chunks for detail retrieval)
        chunk_ids = await self.episodic.ingest(
            text=text,
            conversation_id=conversation_id,
            speaker=speaker,
            timestamp=timestamp,
        )

        # ── NOVEL: Selective Extraction ──────────────────────────
        # Embed turn (already done in episodic ingest — reuse it).
        # Check novelty. Only call LLM for turns with new factual content.
        turn_embedding = await self._embedder.embed(text)
        extracted_facts = await self._selective_extractor.process_turn(
            text, turn_embedding, self.episodic.tai,
        )
        # Store extracted facts as searchable memories
        for fact in extracted_facts:
            await self.episodic.ingest(
                text=fact,
                conversation_id=f"facts-{conversation_id}",
                speaker="fact",
            )
            # Also feed into semantic graph
            await self.semantic.ingest_text_async(fact)

        # Semantic: extract entities/relations via embedding extractor
        extraction = await self.semantic.ingest_text_async(text)

        # Behavioral: update profile from user messages
        if speaker.lower() in ("user", "human", ""):
            self.behavioral.update_from_message(text)

        # Working: note entities
        for ent in extraction.entities:
            self.working.note_entity(ent.name)

        # Collect any contradictions that were detected
        contradictions = self.semantic.get_contradictions()

        return {
            "chunks_stored": len(chunk_ids),
            "entities_extracted": len(extraction.entities),
            "relations_extracted": len(extraction.relations),
            "contradictions": [c for c in contradictions[-5:]],  # last 5
        }

    async def ingest_conversation(self, messages: list[dict], conversation_id: str | None = None) -> dict:
        """Ingest a full conversation."""
        total = {"chunks_stored": 0, "entities_extracted": 0, "relations_extracted": 0, "contradictions": []}
        for msg in messages:
            result = await self.ingest(
                text=msg.get("text", ""),
                conversation_id=conversation_id,
                speaker=msg.get("speaker", ""),
            )
            total["chunks_stored"] += result["chunks_stored"]
            total["entities_extracted"] += result["entities_extracted"]
            total["relations_extracted"] += result["relations_extracted"]
            total["contradictions"].extend(result.get("contradictions", []))
        return total

    def _compute_budget(
        self,
        total_budget: int,
        episodic_signal: float,
        semantic_signal: float,
        has_explicit: bool,
        has_working: bool,
    ) -> LayerBudget:
        """Dynamically allocate token budget across layers.

        Allocation is based on:
        1. Base weights from config
        2. Signal quality — if episodic results are highly relevant, give them more budget
        3. Explicit and working memory get guaranteed minimums
        4. Behavioral gets a small fixed allocation
        """
        # Guaranteed minimums
        behavioral_budget = min(200, total_budget // 20)
        explicit_budget = 0
        working_budget = 0

        if has_explicit:
            # Explicit always gets enough for all entries (high priority)
            explicit_budget = min(total_budget // 4, max(200, total_budget // 8))
        if has_working:
            working_budget = min(total_budget // 6, max(150, total_budget // 10))

        remaining = total_budget - behavioral_budget - explicit_budget - working_budget

        # Dynamic split between episodic and semantic based on signal quality
        # episodic_signal: avg similarity score of top results (0-1)
        # semantic_signal: avg confidence of retrieved facts (0-1)
        total_signal = episodic_signal + semantic_signal
        if total_signal > 0:
            ep_ratio = episodic_signal / total_signal
            sem_ratio = semantic_signal / total_signal
        else:
            # Default: slight bias toward episodic
            ep_ratio = 0.6
            sem_ratio = 0.4

        # Apply config weights as a blend
        cfg_ep = self._config.retrieval.episodic_weight
        cfg_sem = self._config.retrieval.semantic_weight
        cfg_total = cfg_ep + cfg_sem

        # Blend signal-based and config-based ratios (60/40 signal vs config)
        blend_ep = 0.6 * ep_ratio + 0.4 * (cfg_ep / cfg_total)
        blend_sem = 1.0 - blend_ep

        episodic_budget = int(remaining * blend_ep)
        semantic_budget = remaining - episodic_budget

        return LayerBudget(
            episodic=max(0, episodic_budget),
            semantic=max(0, semantic_budget),
            explicit=explicit_budget,
            behavioral=behavioral_budget,
            working=working_budget,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Accurate token count via tiktoken."""
        return count_tokens(text)

    def _trim_to_budget(self, items: list[dict], budget: int, text_key: str = "text") -> list[dict]:
        """Trim a list of items to fit within a token budget."""
        result = []
        used = 0
        for item in items:
            text = str(item.get(text_key, ""))
            tokens = self._estimate_tokens(text)
            if used + tokens > budget:
                break
            result.append(item)
            used += tokens
        return result

    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        token_budget: int | None = None,
    ) -> MemoryContext:
        """Query all layers and assemble context.

        Pipeline:
        1. Extract entities from query
        2. Retrieve from all layers
        3. Compute dynamic budget based on signal quality
        4. Trim each layer to budget
        5. Assemble with behavioral modulation
        """
        if token_budget is None:
            token_budget = self._config.retrieval.token_budget

        # ─── NOVEL: Intent-Aware Dynamic Scoring ───────────────────
        # Analyze query to determine what the user actually wants:
        # recency, precision, breadth, or context-specific retrieval.
        intent = analyze_intent(query_text)

        # Extract entities from query for semantic graph traversal
        extraction = self._extractor.extract(query_text)
        entity_names = [e.name for e in extraction.entities]

        # Also resolve entities through the semantic graph's resolver
        resolved_names = []
        for name in entity_names:
            resolved = self.semantic.resolve_entity_name(name)
            if resolved not in resolved_names:
                resolved_names.append(resolved)

        # Combine entity names for graph queries
        query_entities = resolved_names if resolved_names else entity_names

        # Episodic retrieval with intent-aware weights
        episodic_results = await self.episodic.retrieve(
            query_text, top_k=top_k,
            temporal_weight=intent.temporal_weight,
        )

        # Graph-aware query expansion: also search for related entities
        if query_entities:
            expanded_facts = self.semantic.query(query_entities, max_depth=1)
            expanded_entities = set()
            for f in expanded_facts[:5]:
                expanded_entities.add(f.get("object", ""))
                expanded_entities.add(f.get("subject", ""))
            # Remove entities we already have
            expanded_entities -= set(query_entities) | set(resolved_names)
            expanded_entities.discard("")
            # Search episodic for each expanded entity
            for exp_entity in list(expanded_entities)[:3]:  # limit expansion
                exp_results = await self.episodic.retrieve(exp_entity, top_k=3)
                for r in exp_results:
                    # Only add if not already in results and has decent score
                    if r.chunk_id not in {er.chunk_id for er in episodic_results}:
                        episodic_results.append(r)

        # ─── NOVEL: Context-Anchor Boosting ──────────────────────
        # If the query mentions specific teams/projects/names, boost
        # chunks from matching conversations. This prevents cross-context
        # bleed (Team Alpha content showing up for Team Beta queries).
        # ─── NOVEL: Hybrid Re-ranking ────────────────────────────
        # Vector similarity gets the right neighborhood. Lexical overlap
        # anchors the specific result. This fuses both signals.
        query_terms = set(query_text.lower().split())
        query_terms -= {'what', 'is', 'the', 'a', 'an', 'on', 'in', 'at', 'to',
                        'of', 'for', 'and', 'or', 'my', 'me', 'i', 'do', 'does',
                        'did', 'how', 'why', 'when', 'where', 'which', 'who',
                        'tell', 'about', 'with', 'from', 'that', 'this', 'are'}

        for r in episodic_results:
            meta = getattr(r, 'meta', None) or getattr(r, 'metadata', None)
            if not meta:
                continue
            text_lower = meta.text.lower()
            cid_lower = getattr(meta, 'conversation_id', '').lower()

            # Lexical overlap boost: how many query keywords appear in the chunk?
            chunk_words = set(text_lower.split())
            overlap = len(query_terms & chunk_words)
            if overlap > 0:
                r.final_score *= (1.0 + overlap * 0.08)  # 8% per matching keyword

            # Context anchor boost: named entities/teams in query match chunk metadata
            if intent.context_boost_terms:
                for term in intent.context_boost_terms:
                    if term in text_lower or term in cid_lower:
                        r.final_score *= 1.15  # 15% per anchor match

        # Sort all results by (potentially boosted) score
        episodic_results.sort(key=lambda r: r.final_score, reverse=True)
        episodic_results = episodic_results[:top_k]

        episodic_chunks = []
        for r in episodic_results:
            meta = getattr(r, 'meta', None) or getattr(r, 'metadata', None)
            sim = getattr(r, 'similarity', None) or getattr(r, 'raw_similarity', 0)
            temporal = getattr(r, 'temporal_score', None) or getattr(r, 'temporal_factor', 0)
            episodic_chunks.append({
                "chunk_id": r.chunk_id,
                "text": meta.text if meta else "",
                "conversation_id": getattr(meta, 'conversation_id', ''),
                "speaker": getattr(meta, 'speaker', ''),
                "score": r.final_score,
                "raw_similarity": sim,
                "temporal_factor": temporal,
            })

        # Semantic retrieval (using resolved entity names)
        semantic_facts = self.semantic.query(
            query_entities,
            max_depth=self._config.semantic.max_traversal_depth,
        )

        # Behavioral priors
        behavioral_priors = self.behavioral.get_priors()

        # Explicit entries (always included)
        explicit_entries = self.explicit.get_all_for_context()

        # Working memory context
        working_context = self.working.get_context() if not self.working.is_empty else {}

        # Unresolved contradictions
        contradictions = self.semantic.get_unresolved_contradictions()

        # Compute signal quality for dynamic budget allocation
        episodic_signal = 0.0
        if episodic_chunks:
            episodic_signal = sum(c["score"] for c in episodic_chunks[:5]) / min(5, len(episodic_chunks))
            episodic_signal = max(0.0, min(1.0, episodic_signal))

        semantic_signal = 0.0
        if semantic_facts:
            semantic_signal = sum(f["confidence"] for f in semantic_facts[:5]) / min(5, len(semantic_facts))

        budget = self._compute_budget(
            total_budget=token_budget,
            episodic_signal=episodic_signal,
            semantic_signal=semantic_signal,
            has_explicit=len(explicit_entries) > 0,
            has_working=bool(working_context),
        )

        # Trim each layer to its budget
        trimmed_episodic = self._trim_to_budget(episodic_chunks, budget.episodic, "text")
        trimmed_semantic = self._trim_to_budget(
            [{"text": f"{f['subject']} {f['predicate']} {f['object']}", **f} for f in semantic_facts],
            budget.semantic,
            "text",
        )
        # Remove the added "text" key from semantic facts
        trimmed_semantic = [{k: v for k, v in f.items() if k != "text"} for f in trimmed_semantic]

        ctx = MemoryContext(
            episodic_chunks=trimmed_episodic,
            semantic_facts=trimmed_semantic,
            behavioral_priors=behavioral_priors,
            explicit_entries=explicit_entries,
            working_context=working_context,
            contradictions=contradictions,
            budget_allocation={
                "episodic": budget.episodic,
                "semantic": budget.semantic,
                "explicit": budget.explicit,
                "behavioral": budget.behavioral,
                "working": budget.working,
                "total": budget.total,
            },
        )

        # Generate injection text with behavioral modulation
        ctx.to_injection_text(profile=self.behavioral)

        return ctx

    async def query_knowledge(
        self,
        query_text: str,
        top_k: int = 20,
        token_budget: int | None = None,
        current_session_id: str | None = None,
    ) -> KnowledgeContext:
        """Query for CROSS-SESSION KNOWLEDGE only.

        This is the correct way to use memory with an LLM:
        - The LLM already has the current conversation in its context
        - We provide what it DOESN'T have: identity, past facts, history

        Returns a KnowledgeContext optimized for injection alongside
        the current conversation, not replacing it.
        """
        if token_budget is None:
            token_budget = self._config.retrieval.token_budget

        # If no session ID provided, use the active working memory session
        if current_session_id is None and not self.working.is_empty:
            current_session_id = self.working.session_id

        # Intent analysis
        intent = analyze_intent(query_text)

        # Entity extraction + resolution
        extraction = self._extractor.extract(query_text)
        entity_names = [e.name for e in extraction.entities]
        resolved_names = []
        for name in entity_names:
            resolved = self.semantic.resolve_entity_name(name)
            if resolved not in resolved_names:
                resolved_names.append(resolved)
        query_entities = resolved_names if resolved_names else entity_names

        # Retrieve from each layer
        episodic_results = await self.episodic.retrieve(
            query_text, top_k=top_k,
            temporal_weight=intent.temporal_weight,
        )
        episodic_chunks = []
        for r in episodic_results:
            meta = getattr(r, 'meta', None) or getattr(r, 'metadata', None)
            if meta:
                episodic_chunks.append({
                    "text": meta.text,
                    "conversation_id": getattr(meta, 'conversation_id', ''),
                    "speaker": getattr(meta, 'speaker', ''),
                    "score": r.final_score,
                })

        semantic_facts = self.semantic.query(
            query_entities,
            max_depth=self._config.semantic.max_traversal_depth,
        )

        behavioral_priors = self.behavioral.get_priors()
        explicit_entries = self.explicit.get_all_for_context()

        # Assemble cross-session knowledge
        knowledge = assemble_knowledge(
            explicit_entries=explicit_entries,
            semantic_facts=semantic_facts,
            episodic_chunks=episodic_chunks,
            behavioral_priors=behavioral_priors,
            session_summaries=self._session_summaries,
            current_session_id=current_session_id,
            token_budget=token_budget,
        )

        return knowledge

    def start_session(self, session_id: str | None = None):
        """Start a new working memory session."""
        self.working = WorkingMemory(session_id)

    async def end_session(self) -> str | None:
        """End session: flush selective extractor + summary.

        Selective extraction happened DURING the conversation (per-turn embedding
        gating). At session end, we just flush any remaining pending turns and
        create the session summary.
        """
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        async with self._write_lock:
            session_id = self.working.session_id
            session_text = "\n".join(self._session_turns)

            if not session_text.strip() and self.working.is_empty:
                self._session_turns.clear()
                return None

            # Flush any remaining turns from selective extractor
            remaining_facts = await self._selective_extractor.flush()
            for fact in remaining_facts:
                await self.episodic.ingest(
                    text=fact,
                    conversation_id=f"facts-{session_id}",
                    speaker="fact",
                )
                await self.semantic.ingest_text_async(fact)

            # Create session summary
            stats = self._selective_extractor.stats
            summary = session_text[:300] if session_text else ""
            self._session_summaries.append({
                "session_id": session_id,
                "summary": summary,
                "facts_count": stats.get("turns_extracted", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Also flush working memory if it has content
            if not self.working.is_empty:
                wm_text = self.working.get_all_text()
                if wm_text.strip():
                    await self.episodic.ingest(
                        text=wm_text,
                        conversation_id=f"session-{session_id}",
                        speaker="system",
                    )

            self.working.clear()
            self._session_turns.clear()
            return session_id

    def decay_pass(self):
        """Run decay on all layers."""
        self.episodic.index.decay_pass()
        self.semantic.decay_pass()

    # --- Entity resolution shortcuts ---

    def merge_entities(self, name_a: str, name_b: str) -> bool:
        """Merge two entities in the semantic graph."""
        return self.semantic.merge_entities(name_a, name_b)

    def add_entity_alias(self, canonical_name: str, alias: str):
        """Add an alias for a known entity. Registers the canonical if needed."""
        self.semantic.resolver.resolve_or_create(canonical_name)
        self.semantic.resolver.add_alias(canonical_name, alias)

    def retract_fact(self, subject: str, predicate: str, obj: str) -> bool:
        """Explicitly retract a fact from the semantic graph."""
        return self.semantic.retract_fact(subject, predicate, obj)

    # --- Persistence ---

    def save(self, base_path: Path | None = None):
        """Persist all layers.

        If SQLite is initialized, episodic/explicit/behavioral are already
        written incrementally. Only semantic graph still uses file-based save.
        """
        if base_path is None:
            base_path = Path(self._config.storage.data_dir)
        # Semantic graph still uses file-based persistence (NetworkX JSON)
        self.semantic.save(base_path / "semantic")
        # Legacy: save episodic/explicit/behavioral to files if no SQLite
        if self._db is None:
            self.episodic.save(base_path / "episodic")
            self.behavioral.save(base_path / "behavioral")
            self.explicit.save(base_path / "explicit")

    def load(self, base_path: Path | None = None):
        """Load all layers. Prefers SQLite if initialized, falls back to files."""
        if base_path is None:
            base_path = Path(self._config.storage.data_dir)
        # Semantic graph always loads from files
        self.semantic.load(base_path / "semantic")
        if self._db is not None:
            self.load_from_db()
        else:
            self.episodic.load(base_path / "episodic")
            self.behavioral.load(base_path / "behavioral")
            self.explicit.load(base_path / "explicit")

    def close(self):
        """Clean shutdown."""
        if self._db:
            self._db.close()

    def stats(self) -> dict:
        s = {
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
            "explicit": {"count": self.explicit.count},
            "behavioral": self.behavioral.get_summary(),
            "working": {"active": not self.working.is_empty},
            "persistence": "sqlite" if self._db else "file",
        }
        if self._db:
            s["db_stats"] = self._db.stats()
        return s
