"""Semantic knowledge graph — Layer 2.

Stores structured facts as a directed multigraph with:
- Entity resolution (fuzzy matching, aliases, vector similarity)
- Contradiction detection and temporal reasoning
- Confidence decay with reinforcement
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from amem.semantic.decay import ConfidenceDecay
from amem.semantic.extractor import EntityExtractor, ExtractionResult
from amem.semantic.resolver import EntityResolver
from amem.semantic.contradictions import ContradictionDetector, FactStatus, Contradiction


@dataclass
class Entity:
    name: str
    entity_type: str  # person, project, tool, concept, org, location
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mention_count: int = 1
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_chunks: list[str] = field(default_factory=list)
    mention_count: int = 1


class SemanticGraph:
    """Knowledge graph for structured fact storage with entity resolution,
    contradiction detection, and temporal decay."""

    def __init__(self, decay: ConfidenceDecay | None = None):
        self._graph = nx.MultiDiGraph()
        self._decay = decay or ConfidenceDecay()
        self._extractor = EntityExtractor()
        self._llm_extractor = None  # Set via set_llm_extractor()
        self._resolver = EntityResolver()
        self._contradictions = ContradictionDetector()
        self._db = None  # SQLiteStore, set via set_db()

    def set_llm_extractor(self, extractor):
        """Attach an LLM-based extractor for higher-quality extraction."""
        self._llm_extractor = extractor

    def set_embedding_extractor(self, extractor):
        """Attach embedding-powered extractor (uses same model as episodic store)."""
        self._embedding_extractor = extractor

    def set_db(self, db):
        """Attach SQLite backend for durable persistence."""
        self._db = db

    @property
    def entity_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def relation_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def resolver(self) -> EntityResolver:
        return self._resolver

    @property
    def contradiction_detector(self) -> ContradictionDetector:
        return self._contradictions

    def _persist_entity(self, key: str, data: dict):
        """Write entity to SQLite if available."""
        if self._db:
            self._db.save_entity(key, data)

    def _persist_relation(self, subj_key: str, obj_key: str, data: dict):
        """Write relation to SQLite if available."""
        if self._db:
            self._db.save_relation(subj_key, obj_key, data)

    def resolve_entity_name(self, name: str) -> str:
        """Resolve a name to its canonical form via entity resolver."""
        canonical = self._resolver.resolve(name)
        if canonical is not None:
            return canonical.canonical_name
        return name

    def add_entity(self, entity: Entity):
        """Add or update an entity node, resolving to canonical form."""
        # Resolve to canonical name
        canonical = self._resolver.resolve_or_create(
            entity.name,
            entity_type=entity.entity_type,
        )
        key = canonical.canonical_name.lower()

        if self._graph.has_node(key):
            data = self._graph.nodes[key]
            data["last_seen"] = entity.last_seen.isoformat() if isinstance(entity.last_seen, datetime) else entity.last_seen
            data["mention_count"] = data.get("mention_count", 0) + 1
            data["attributes"].update(entity.attributes)
            # Keep track of aliases
            if entity.name.lower() != key:
                aliases = set(data.get("aliases", []))
                aliases.add(entity.name)
                data["aliases"] = list(aliases)
        else:
            self._graph.add_node(key, **{
                "name": canonical.canonical_name,
                "entity_type": entity.entity_type,
                "first_seen": entity.first_seen.isoformat(),
                "last_seen": entity.last_seen.isoformat(),
                "mention_count": entity.mention_count,
                "attributes": entity.attributes,
                "aliases": list(canonical.aliases),
            })

        # Persist to SQLite
        self._persist_entity(key, dict(self._graph.nodes[key]))

    def add_relation(self, relation: Relation) -> list[Contradiction]:
        """Add or reinforce a relation edge with contradiction checking.

        Returns list of any contradictions detected.
        """
        # Resolve entity names
        resolved_subj = self.resolve_entity_name(relation.subject)
        resolved_obj = self.resolve_entity_name(relation.object)
        subj = resolved_subj.lower()
        obj = resolved_obj.lower()

        # Ensure nodes exist
        if not self._graph.has_node(subj):
            self.add_entity(Entity(name=resolved_subj, entity_type="concept"))
        if not self._graph.has_node(obj):
            self.add_entity(Entity(name=resolved_obj, entity_type="concept"))

        # Check for contradictions before adding
        new_fact = {
            "subject": resolved_subj,
            "predicate": relation.predicate,
            "object": resolved_obj,
            "confidence": relation.confidence,
            "timestamp": relation.last_seen.isoformat() if isinstance(relation.last_seen, datetime) else relation.last_seen,
            "last_seen": relation.last_seen.isoformat() if isinstance(relation.last_seen, datetime) else relation.last_seen,
        }

        # Get existing facts for this subject to check contradictions
        existing_facts = self._get_facts_for_subject(subj)
        contradictions = self._contradictions.check_and_resolve(new_fact, existing_facts)

        # Handle superseded facts — reduce their confidence
        for c in contradictions:
            if c.resolution != "unresolved":
                loser = c.fact_a if c.winner == "b" else c.fact_b if c.winner == "a" else None
                if loser:
                    self._demote_fact(
                        loser.get("subject", "").lower(),
                        loser.get("predicate", ""),
                        loser.get("object", "").lower(),
                    )

        # Check for existing edge with same predicate
        existing = None
        if self._graph.has_edge(subj, obj):
            for k, data in self._graph[subj][obj].items():
                if data.get("predicate") == relation.predicate:
                    existing = k
                    break

        now = datetime.now(timezone.utc)
        if existing is not None:
            data = self._graph[subj][obj][existing]
            data["confidence"] = self._decay.reinforce(data["confidence"])
            data["last_seen"] = now.isoformat()
            data["mention_count"] = data.get("mention_count", 0) + 1
            data["source_chunks"].extend(relation.source_chunks)
            data["status"] = FactStatus.ACTIVE.value
        else:
            self._graph.add_edge(subj, obj,
                predicate=relation.predicate,
                confidence=relation.confidence,
                first_seen=relation.first_seen.isoformat(),
                last_seen=relation.last_seen.isoformat(),
                source_chunks=relation.source_chunks,
                mention_count=relation.mention_count,
                status=FactStatus.ACTIVE.value,
            )

        # Persist to SQLite
        edge_data = {"predicate": relation.predicate}
        if existing is not None:
            edge_data = dict(self._graph[subj][obj][existing])
        else:
            edge_data = {
                "predicate": relation.predicate,
                "confidence": relation.confidence,
                "first_seen": relation.first_seen.isoformat(),
                "last_seen": relation.last_seen.isoformat(),
                "source_chunks": relation.source_chunks,
                "mention_count": relation.mention_count,
                "status": FactStatus.ACTIVE.value,
            }
        self._persist_relation(subj, obj, edge_data)

        return contradictions

    def _get_facts_for_subject(self, subject_key: str) -> list[dict]:
        """Get all facts where subject_key is the subject."""
        facts = []
        for _, target, _, data in self._graph.out_edges(subject_key, keys=True, data=True):
            tgt_data = self._graph.nodes.get(target, {})
            facts.append({
                "subject": self._graph.nodes.get(subject_key, {}).get("name", subject_key),
                "predicate": data.get("predicate", ""),
                "object": tgt_data.get("name", target),
                "confidence": data.get("confidence", 0.5),
                "last_seen": data.get("last_seen", ""),
                "status": data.get("status", FactStatus.ACTIVE.value),
            })
        return facts

    def _demote_fact(self, subject_key: str, predicate: str, object_key: str):
        """Reduce confidence of a superseded fact."""
        if self._graph.has_edge(subject_key, object_key):
            for k, data in self._graph[subject_key][object_key].items():
                if data.get("predicate") == predicate:
                    data["confidence"] = max(0.01, data["confidence"] * 0.3)
                    data["status"] = FactStatus.SUPERSEDED.value
                    break

    def retract_fact(self, subject: str, predicate: str, obj: str) -> bool:
        """Explicitly retract a fact (user-driven)."""
        subj_key = self.resolve_entity_name(subject).lower()
        obj_key = self.resolve_entity_name(obj).lower()

        self._contradictions.retract_fact(subject, predicate, obj)

        if self._graph.has_edge(subj_key, obj_key):
            for k, data in self._graph[subj_key][obj_key].items():
                if data.get("predicate") == predicate:
                    data["status"] = FactStatus.RETRACTED.value
                    data["confidence"] = 0.0
                    return True
        return False

    def merge_entities(self, name_a: str, name_b: str) -> bool:
        """Merge two entities — name_a becomes canonical, name_b becomes alias.

        All edges from/to name_b are repointed to name_a.
        """
        canonical = self._resolver.merge(name_a, name_b)
        if canonical is None:
            return False

        key_a = canonical.canonical_name.lower()
        key_b = name_b.lower()

        if not self._graph.has_node(key_b):
            return True  # nothing to merge in graph

        # Move edges from key_b to key_a
        # Outgoing
        for _, target, k, data in list(self._graph.out_edges(key_b, keys=True, data=True)):
            if target == key_a:
                continue  # skip self-loops
            self._graph.add_edge(key_a, target, **data)
        # Incoming
        for source, _, k, data in list(self._graph.in_edges(key_b, keys=True, data=True)):
            if source == key_a:
                continue
            self._graph.add_edge(source, key_a, **data)

        # Merge node attributes
        if self._graph.has_node(key_a) and self._graph.has_node(key_b):
            data_a = self._graph.nodes[key_a]
            data_b = self._graph.nodes[key_b]
            data_a["mention_count"] = data_a.get("mention_count", 0) + data_b.get("mention_count", 0)
            aliases = set(data_a.get("aliases", []))
            aliases.add(name_b)
            aliases.update(data_b.get("aliases", []))
            data_a["aliases"] = list(aliases)

        # Remove old node
        self._graph.remove_node(key_b)
        return True

    async def ingest_text_async(self, text: str, source_chunk_id: str = "") -> ExtractionResult:
        """Extract entities and relations (async).

        Priority: embedding extractor > LLM extractor > heuristic fallback.
        Embedding extractor uses the SAME model as the episodic store.
        """
        if not text or not text.strip():
            return ExtractionResult()

        if hasattr(self, '_embedding_extractor') and self._embedding_extractor is not None:
            result = await self._embedding_extractor.extract(text)
        elif self._llm_extractor is not None:
            result = await self._llm_extractor.extract(text)
        else:
            result = self._extractor.extract(text)

        return self._apply_extraction(result, source_chunk_id)

    def ingest_text(self, text: str, source_chunk_id: str = "") -> ExtractionResult:
        """Extract entities and relations from text and add to graph (sync, heuristic only)."""
        if not text or not text.strip():
            return ExtractionResult()

        result = self._extractor.extract(text)
        return self._apply_extraction(result, source_chunk_id)

    def _apply_extraction(self, result: ExtractionResult, source_chunk_id: str = "") -> ExtractionResult:
        """Apply extracted entities/relations to the graph."""
        now = datetime.now(timezone.utc)

        for ent in result.entities:
            self.add_entity(Entity(
                name=ent.name,
                entity_type=ent.entity_type,
                first_seen=now,
                last_seen=now,
                mention_count=ent.mentions,
            ))

        all_contradictions = []
        for rel in result.relations:
            contradictions = self.add_relation(Relation(
                subject=rel.subject,
                predicate=rel.predicate,
                object=rel.object,
                confidence=rel.confidence,
                first_seen=now,
                last_seen=now,
                source_chunks=[source_chunk_id] if source_chunk_id else [],
            ))
            all_contradictions.extend(contradictions)

        return result

    def query(self, entity_names: list[str], max_depth: int = 2,
              include_superseded: bool = False) -> list[dict]:
        """Query graph for facts related to given entities.

        Returns list of fact dicts with status and confidence.
        BFS from each entity to max_depth.
        Filters out superseded/retracted facts by default.
        """
        facts = []
        visited_edges: set[tuple] = set()
        now = datetime.now(timezone.utc)

        for name in entity_names:
            # Resolve through entity resolver
            resolved = self.resolve_entity_name(name)
            key = resolved.lower()
            if not self._graph.has_node(key):
                continue

            # BFS traversal
            queue: deque[tuple[str, int]] = deque([(key, 0)])
            visited_nodes: set[str] = {key}

            while queue:
                node, depth = queue.popleft()
                if depth >= max_depth:
                    continue

                # Outgoing edges
                for _, target, k, data in self._graph.out_edges(node, keys=True, data=True):
                    edge_key = (node, target, data.get("predicate", ""))
                    if edge_key in visited_edges:
                        continue
                    visited_edges.add(edge_key)

                    # Check fact status
                    status = data.get("status", FactStatus.ACTIVE.value)
                    if not include_superseded and status in (FactStatus.SUPERSEDED.value, FactStatus.RETRACTED.value):
                        continue

                    # Compute decayed confidence
                    last_seen = datetime.fromisoformat(data["last_seen"]) if isinstance(data["last_seen"], str) else data["last_seen"]
                    conf = self._decay.compute(data["confidence"], last_seen, now)

                    if not self._decay.should_prune(conf):
                        src_data = self._graph.nodes[node]
                        tgt_data = self._graph.nodes[target]
                        facts.append({
                            "subject": src_data.get("name", node),
                            "predicate": data["predicate"],
                            "object": tgt_data.get("name", target),
                            "confidence": round(conf, 4),
                            "last_seen": data["last_seen"],
                            "status": status,
                            "mention_count": data.get("mention_count", 1),
                        })

                    if target not in visited_nodes:
                        visited_nodes.add(target)
                        queue.append((target, depth + 1))

                # Incoming edges
                for source, _, k, data in self._graph.in_edges(node, keys=True, data=True):
                    edge_key = (source, node, data.get("predicate", ""))
                    if edge_key in visited_edges:
                        continue
                    visited_edges.add(edge_key)

                    status = data.get("status", FactStatus.ACTIVE.value)
                    if not include_superseded and status in (FactStatus.SUPERSEDED.value, FactStatus.RETRACTED.value):
                        continue

                    last_seen = datetime.fromisoformat(data["last_seen"]) if isinstance(data["last_seen"], str) else data["last_seen"]
                    conf = self._decay.compute(data["confidence"], last_seen, now)

                    if not self._decay.should_prune(conf):
                        src_data = self._graph.nodes[source]
                        tgt_data = self._graph.nodes[node]
                        facts.append({
                            "subject": src_data.get("name", source),
                            "predicate": data["predicate"],
                            "object": tgt_data.get("name", node),
                            "confidence": round(conf, 4),
                            "last_seen": data["last_seen"],
                            "status": status,
                            "mention_count": data.get("mention_count", 1),
                        })

                    if source not in visited_nodes:
                        visited_nodes.add(source)
                        queue.append((source, depth + 1))

        # Sort by confidence descending
        facts.sort(key=lambda f: f["confidence"], reverse=True)
        return facts

    def get_entity(self, name: str) -> dict | None:
        """Get entity data by name (resolves through entity resolver)."""
        resolved = self.resolve_entity_name(name)
        key = resolved.lower()
        if not self._graph.has_node(key):
            return None
        return dict(self._graph.nodes[key])

    def get_entities(self) -> list[dict]:
        """Get all entities."""
        return [{"key": n, **dict(d)} for n, d in self._graph.nodes(data=True)]

    def get_contradictions(self, subject: str | None = None) -> list[dict]:
        """Get contradictions, optionally filtered by subject."""
        contradictions = self._contradictions.get_contradictions(subject)
        return [c.to_dict() for c in contradictions]

    def get_unresolved_contradictions(self) -> list[dict]:
        return [c.to_dict() for c in self._contradictions.get_unresolved()]

    def decay_pass(self, now: datetime | None = None):
        """Run decay on all edges, prune those below threshold."""
        if now is None:
            now = datetime.now(timezone.utc)

        to_remove = []
        for u, v, k, data in self._graph.edges(keys=True, data=True):
            last_seen = datetime.fromisoformat(data["last_seen"]) if isinstance(data["last_seen"], str) else data["last_seen"]
            conf = self._decay.compute(data["confidence"], last_seen, now)
            if self._decay.should_prune(conf):
                to_remove.append((u, v, k))
            else:
                data["confidence"] = conf

        for u, v, k in to_remove:
            self._graph.remove_edge(u, v, key=k)

        # Remove orphan nodes
        orphans = [n for n in self._graph.nodes() if self._graph.degree(n) == 0]
        self._graph.remove_nodes_from(orphans)

    def save(self, path: Path):
        """Save graph, resolver, and contradiction state.

        If SQLite is attached, entities/relations are already persisted
        incrementally. File-based save is kept for resolver + contradictions
        (which have complex structures not yet in SQLite tables).
        """
        path.mkdir(parents=True, exist_ok=True)
        if not self._db:
            # Legacy: full dump to JSON
            data = nx.node_link_data(self._graph)
            with open(path / "graph.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
        # Resolver and contradictions always save to file (compact, infrequent)
        with open(path / "resolver.json", "w") as f:
            json.dump(self._resolver.to_dict(), f, indent=2, default=str)
        with open(path / "contradictions.json", "w") as f:
            json.dump(self._contradictions.to_dict(), f, indent=2, default=str)

    def load(self, path: Path):
        """Load graph, resolver, and contradiction state."""
        if self._db:
            self.load_from_db()
        else:
            graph_file = path / "graph.json"
            if graph_file.exists():
                with open(graph_file) as f:
                    data = json.load(f)
                self._graph = nx.node_link_graph(data, directed=True, multigraph=True)

        resolver_file = path / "resolver.json"
        if resolver_file.exists():
            with open(resolver_file) as f:
                data = json.load(f)
            self._resolver = EntityResolver.from_dict(data)

        contradictions_file = path / "contradictions.json"
        if contradictions_file.exists():
            with open(contradictions_file) as f:
                data = json.load(f)
            self._contradictions = ContradictionDetector.from_dict(data)

    def load_from_db(self):
        """Rebuild in-memory graph from SQLite."""
        if not self._db:
            return
        self._graph = nx.MultiDiGraph()

        # Load entities
        for ent in self._db.load_all_entities():
            key = ent["key"]
            self._graph.add_node(key, **{k: v for k, v in ent.items() if k != "key"})

        # Load relations
        for rel in self._db.load_all_relations():
            subj = rel.pop("subject_key")
            obj = rel.pop("object_key")
            # Ensure nodes exist
            if not self._graph.has_node(subj):
                self._graph.add_node(subj, name=subj, entity_type="concept",
                                      first_seen="", last_seen="", mention_count=1,
                                      attributes={}, aliases=[])
            if not self._graph.has_node(obj):
                self._graph.add_node(obj, name=obj, entity_type="concept",
                                      first_seen="", last_seen="", mention_count=1,
                                      attributes={}, aliases=[])
            self._graph.add_edge(subj, obj, **rel)

    def stats(self) -> dict:
        return {
            "entities": self.entity_count,
            "relations": self.relation_count,
            "components": nx.number_weakly_connected_components(self._graph),
            "resolved_entities": self._resolver.entity_count,
            "contradictions_total": self._contradictions.contradiction_count,
            "contradictions_unresolved": self._contradictions.unresolved_count,
        }
