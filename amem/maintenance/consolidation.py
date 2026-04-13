"""Memory consolidation engine — inspired by human sleep.

Periodic process that:
1. Reviews episodic chunks from a time window
2. Identifies frequently-mentioned entities/facts → promotes to semantic graph
3. Detects emerging topic clusters
4. Evicts dead memories (low confidence, old, never accessed)

This is what makes the system LEARN, not just store.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amem.retrieval.orchestrator import MemoryOrchestrator
    from amem.semantic.graph import Entity, Relation


class MemoryConsolidator:
    """Consolidates episodic memory into semantic knowledge.

    Like human memory consolidation during sleep:
    - Frequently accessed episodic memories → crystallize into semantic facts
    - Rarely accessed memories → decay and eventually evict
    - Patterns across episodes → extracted as general knowledge
    """

    def __init__(
        self,
        min_mentions_to_promote: int = 3,
        promotion_confidence: float = 0.85,
        eviction_age_days: int = 90,
        eviction_min_access: int = 0,
    ):
        self.min_mentions = min_mentions_to_promote
        self.promotion_confidence = promotion_confidence
        self.eviction_age_days = eviction_age_days
        self.eviction_min_access = eviction_min_access

    async def consolidate(self, orch: "MemoryOrchestrator") -> dict:
        """Run a full consolidation pass.

        Returns summary of actions taken.
        """
        results = {
            "entities_promoted": 0,
            "relations_promoted": 0,
            "chunks_evicted": 0,
            "topics_detected": 0,
        }

        # 1. Analyze episodic chunks for recurring entities/relations
        promotion_results = self._promote_frequent_knowledge(orch)
        results["entities_promoted"] = promotion_results["entities"]
        results["relations_promoted"] = promotion_results["relations"]

        # 2. Evict dead memories
        results["chunks_evicted"] = self._evict_dead_memories(orch)

        # 3. Detect topic clusters
        results["topics_detected"] = self._detect_topic_clusters(orch)

        return results

    def _promote_frequent_knowledge(self, orch: "MemoryOrchestrator") -> dict:
        """Scan episodic chunks for entities/relations mentioned frequently.

        If an entity appears in N+ separate conversations, promote to
        semantic graph with high confidence.
        """
        from amem.semantic.graph import Entity, Relation
        from amem.semantic.extractor import EntityExtractor

        extractor = EntityExtractor()
        entity_counter: Counter = Counter()
        relation_counter: Counter = Counter()
        entity_types: dict[str, str] = {}

        # Scan all chunks in hot + warm tiers (recent + medium-term)
        for shard in (orch.episodic.tai._hot, orch.episodic.tai._warm):
            for i in range(shard.count):
                meta = shard._metadata[i]
                result = extractor.extract(meta.text)

                for ent in result.entities:
                    key = ent.name.lower()
                    entity_counter[key] += 1
                    entity_types[key] = ent.entity_type

                for rel in result.relations:
                    rel_key = (rel.subject.lower(), rel.predicate, rel.object.lower())
                    relation_counter[rel_key] += 1

        # Promote entities that appear frequently
        promoted_entities = 0
        for name, count in entity_counter.items():
            if count >= self.min_mentions:
                etype = entity_types.get(name, "concept")
                orch.semantic.add_entity(Entity(
                    name=name.title(),
                    entity_type=etype,
                    mention_count=count,
                ))
                promoted_entities += 1

        # Promote relations that appear frequently
        promoted_relations = 0
        for (subj, pred, obj), count in relation_counter.items():
            if count >= self.min_mentions:
                orch.semantic.add_relation(Relation(
                    subject=subj.title(),
                    predicate=pred,
                    object=obj.title(),
                    confidence=min(1.0, self.promotion_confidence + count * 0.02),
                    mention_count=count,
                ))
                promoted_relations += 1

        return {"entities": promoted_entities, "relations": promoted_relations}

    def _evict_dead_memories(self, orch: "MemoryOrchestrator") -> int:
        """Remove chunks that are old, low-confidence, and never accessed."""
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - (self.eviction_age_days * 86400)
        evicted = 0

        # Only evict from cold tier
        cold = orch.episodic.tai._cold
        candidates = cold.get_eviction_candidates(threshold=0.05)

        for chunk_id in candidates:
            meta = cold.get_meta(chunk_id)
            if meta and meta.timestamp < cutoff and meta.access_count <= self.eviction_min_access:
                cold.remove(chunk_id)
                evicted += 1

        return evicted

    def _detect_topic_clusters(self, orch: "MemoryOrchestrator") -> int:
        """Detect emerging topic clusters from co-retrieval patterns.

        Chunks that are frequently retrieved together form a topic cluster.
        We detect these and tag them for faster future retrieval.
        """
        coretrieval = orch.episodic.tai._coretrieval

        # Find clusters: groups of chunks with mutual co-retrieval > threshold
        clusters = []
        visited = set()

        for chunk_id, links in coretrieval.items():
            if chunk_id in visited:
                continue
            # Find strongly connected peers
            strong_peers = [peer for peer, count in links.items() if count >= 3]
            if len(strong_peers) >= 2:
                cluster = {chunk_id} | set(strong_peers)
                visited.update(cluster)
                clusters.append(cluster)

        # Tag cluster members with a topic label (based on shared entities)
        for i, cluster in enumerate(clusters):
            topic_tag = f"cluster_{i}"
            for chunk_id in cluster:
                for shard in (orch.episodic.tai._hot, orch.episodic.tai._warm):
                    meta = shard.get_meta(chunk_id)
                    if meta and topic_tag not in meta.topic_tags:
                        meta.topic_tags.append(topic_tag)

        return len(clusters)
