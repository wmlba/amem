#!/usr/bin/env python3
"""
=======================================================================
  ASSOCIATIVE MEMORY — FULL END-TO-END DEMO
=======================================================================

This demo simulates a realistic multi-session user journey:

  Session 1: User introduces themselves, sets preferences
  Session 2: User discusses work projects
  Session 3: User changes jobs (triggers contradiction detection)
  Session 4: Query time — watch all 5 memory layers work together

Each step prints what's happening under the hood.
"""

import asyncio
import json
import tempfile
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from amem.config import Config
from amem.embeddings.base import EmbeddingProvider
from amem.retrieval.orchestrator import MemoryOrchestrator
from amem.feedback.relevance import RelevanceFeedback
from amem.maintenance.consolidation import MemoryConsolidator


# ─── Embedding Setup ─────────────────────────────────────────────────

def _check_ollama():
    """Check if Ollama is available."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


class SemanticMockEmbedder(EmbeddingProvider):
    """Mock embedder that produces semantically-meaningful vectors.

    Uses keyword hashing to create vectors where texts about similar
    topics produce similar embeddings. Not as good as real models,
    but demonstrates the system's behavior.
    """
    TOPIC_SEEDS = {
        "ml": [0, 1, 2], "machine": [0, 1], "learning": [0, 2], "model": [0, 3],
        "python": [4, 5], "pytorch": [4, 6], "code": [4, 7], "programming": [4, 5, 7],
        "team": [8, 9], "lead": [8, 10], "manage": [8, 11], "engineer": [8, 12],
        "gpu": [13, 14], "cuda": [13, 15], "blackwell": [13, 16], "workstation": [13, 17],
        "kitchen": [18, 19], "granite": [18, 20], "cabinet": [18, 21], "renovation": [18, 22],
        "finance": [23, 24], "revenue": [23, 25], "quarter": [23, 26],
        "alice": [27, 28], "bob": [29, 30], "will": [31, 32],
        "oci": [33, 34], "anthropic": [35, 36], "google": [37, 38],
        "infrastructure": [39, 40], "kubernetes": [39, 41], "docker": [39, 42],
        "research": [43, 44], "paper": [43, 45], "experiment": [43, 46],
    }

    def __init__(self, dim=64):
        self._dim = dim

    @property
    def dimension(self):
        return self._dim

    async def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            for key, indices in self.TOPIC_SEEDS.items():
                if key in word:
                    for idx in indices:
                        if idx < self._dim:
                            vec[idx] += 1.0
        # Add some uniqueness from hash
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        vec += rng.standard_normal(self._dim).astype(np.float32) * 0.3
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]


# ─── Pretty Printing ─────────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def banner(text):
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

def section(text):
    print(f"\n{CYAN}{BOLD}--- {text} ---{RESET}\n")

def info(text):
    print(f"  {DIM}{text}{RESET}")

def result(label, value):
    print(f"  {GREEN}{label}:{RESET} {value}")

def warn(text):
    print(f"  {YELLOW}⚠ {text}{RESET}")

def user_says(text):
    print(f"  {BLUE}👤 User:{RESET} \"{text}\"")

def system_does(text):
    print(f"  {DIM}🔧 {text}{RESET}")


# ─── Demo ─────────────────────────────────────────────────────────────

async def main():
    banner("ASSOCIATIVE MEMORY — END-TO-END DEMO")

    use_ollama = _check_ollama()
    if use_ollama:
        print(f"  {GREEN}✓ Ollama detected — using real embeddings{RESET}")
        from amem.embeddings.ollama import OllamaEmbedding
        config = Config()
        embedder = OllamaEmbedding(config.ollama)
        # Probe dimension
        test_vec = await embedder.embed("test")
        config.ollama.embedding_dim = len(test_vec)
    else:
        print(f"  {YELLOW}⚠ Ollama not available — using semantic mock embeddings{RESET}")
        print(f"  {DIM}  (Install Ollama + run 'ollama pull nomic-embed-text' for real embeddings){RESET}")
        config = Config()
        embedder = SemanticMockEmbedder(dim=64)

    # Create orchestrator with SQLite persistence
    tmpdir = tempfile.mkdtemp()
    config.storage.data_dir = tmpdir
    orch = MemoryOrchestrator(embedder, config)
    orch.init_db(Path(tmpdir) / "amem.db")

    feedback = RelevanceFeedback()
    consolidator = MemoryConsolidator(min_mentions_to_promote=2)

    # ═══════════════════════════════════════════════════════════════════
    # SESSION 1: User introduces themselves
    # ═══════════════════════════════════════════════════════════════════
    banner("SESSION 1: User Introduction")

    orch.start_session("session-1")

    msg1 = "I'm Will, a senior ML engineer at OCI. I lead the MEA SCE team and we work on GPU infrastructure."
    user_says(msg1)
    system_does("Ingesting into all 5 memory layers...")
    r = await orch.ingest(text=msg1, conversation_id="conv-1", speaker="user")
    result("Chunks stored", r["chunks_stored"])
    result("Entities extracted", r["entities_extracted"])
    result("Relations extracted", r["relations_extracted"])

    msg2 = "We use the GB10 Blackwell workstation for training experiments. I prefer concise, technical responses."
    user_says(msg2)
    r = await orch.ingest(text=msg2, conversation_id="conv-1", speaker="user")
    result("Chunks stored", r["chunks_stored"])

    section("Setting Explicit Memories (Layer 5)")
    orch.explicit.set("name", "Will", entry_type="fact", priority=10)
    orch.explicit.set("communication_style", "concise technical responses", entry_type="preference", priority=8)
    orch.explicit.set("always_do", "Show code examples when explaining concepts", entry_type="instruction", priority=9)
    result("Explicit memories", orch.explicit.count)
    for e in orch.explicit.list_all():
        info(f"  [{e.entry_type}] {e.key} = {e.value}")

    section("Entity Resolution (Layer 2)")
    orch.add_entity_alias("GB10", "Blackwell workstation")
    orch.add_entity_alias("GB10", "the workstation")
    resolved = orch.semantic.resolver.resolve("Blackwell workstation")
    if resolved:
        result("'Blackwell workstation' resolves to", resolved.canonical_name)
        result("Aliases", list(resolved.aliases))

    section("Working Memory (Layer 4)")
    orch.working.add_goal("Set up memory system")
    orch.working.add_fact("User is Will, ML engineer at OCI")
    info(f"Active session: {orch.working.session_id}")
    info(f"Goals: {orch.working.get_context()['goals']}")

    system_does("Ending session → flushing working memory to episodic store...")
    await orch.end_session()
    result("Session flushed", "✓")

    # ═══════════════════════════════════════════════════════════════════
    # SESSION 2: Work projects
    # ═══════════════════════════════════════════════════════════════════
    banner("SESSION 2: Work Projects")

    orch.start_session("session-2")

    msg3 = "I've been researching H2RR, an improvement over Euclidean HRR with 38 to 43 percent better performance on associative memory tasks."
    user_says(msg3)
    r = await orch.ingest(text=msg3, conversation_id="conv-2", speaker="user")
    result("Chunks stored", r["chunks_stored"])

    msg4 = "We use Python and PyTorch for most experiments. The GB10 workstation runs our training jobs on CUDA."
    user_says(msg4)
    r = await orch.ingest(text=msg4, conversation_id="conv-2", speaker="user")
    result("Chunks stored", r["chunks_stored"])

    msg5 = "Alice from the infrastructure team helps us with Kubernetes deployments and Docker containers."
    user_says(msg5)
    r = await orch.ingest(text=msg5, conversation_id="conv-2", speaker="user")
    result("Chunks stored", r["chunks_stored"])

    section("Behavioral Profile (Layer 3) — Built From Messages")
    summary = orch.behavioral.get_summary()
    for dim, desc in summary.items():
        result(dim, desc)

    section("Knowledge Graph Snapshot (Layer 2)")
    entities = orch.semantic.get_entities()
    result("Total entities", len(entities))
    for e in entities[:8]:
        info(f"  {e.get('name', e.get('key', '?'))} ({e.get('entity_type', '?')})")

    await orch.end_session()

    # ═══════════════════════════════════════════════════════════════════
    # SESSION 3: Job change → CONTRADICTION
    # ═══════════════════════════════════════════════════════════════════
    banner("SESSION 3: Job Change (Contradiction Detection)")

    orch.start_session("session-3")

    msg6 = "I left OCI last month. I now work at Anthropic as a research scientist."
    user_says(msg6)
    warn("This should trigger contradiction detection: 'works at OCI' vs 'works at Anthropic'")
    r = await orch.ingest(text=msg6, conversation_id="conv-3", speaker="user")
    result("Chunks stored", r["chunks_stored"])
    result("Entities extracted", r["entities_extracted"])

    section("Contradiction Detection Results")
    contradictions = orch.semantic.get_contradictions()
    if contradictions:
        for c in contradictions:
            fa = c.get("fact_a", {})
            fb = c.get("fact_b", {})
            result("Type", c.get("contradiction_type", "unknown"))
            info(f"  Fact A: {fa.get('subject', '?')} {fa.get('predicate', '?')} {fa.get('object', '?')}")
            info(f"  Fact B: {fb.get('subject', '?')} {fb.get('predicate', '?')} {fb.get('object', '?')}")
            result("Resolution", c.get("resolution", "unresolved"))
            result("Winner", c.get("winner", "none"))
    else:
        info("(No contradictions detected — entity extraction may not have captured 'works_at' relation)")

    section("Explicit Fact Retraction")
    system_does("Retracting: Will works_at OCI")
    orch.retract_fact("Will", "works_at", "OCI")
    result("Retracted", "✓")

    await orch.end_session()

    # ═══════════════════════════════════════════════════════════════════
    # QUERY TIME: Watch all layers work together
    # ═══════════════════════════════════════════════════════════════════
    banner("QUERY: 'What does Will work on?'")

    section("Retrieval Pipeline Executing...")
    system_does("1. Extracting entities from query → ['Will']")
    system_does("2. Resolving entities through entity resolver")
    system_does("3. Searching episodic store (Temporal Associative Index)")
    system_does("4. Traversing knowledge graph for related facts")
    system_does("5. Computing dynamic budget allocation")
    system_does("6. Assembling context with behavioral modulation")

    ctx = await orch.query("What does Will work on?", top_k=5)

    section("Layer 5: Explicit Memories (highest priority, always injected)")
    for e in ctx.explicit_entries:
        result(e["key"], e["value"])

    section("Layer 2: Semantic Facts (knowledge graph)")
    if ctx.semantic_facts:
        for f in ctx.semantic_facts[:5]:
            status_tag = f" [{f.get('status', '')}]" if f.get('status') != 'active' else ""
            info(f"  {f['subject']} —[{f['predicate']}]→ {f['object']}  (conf: {f['confidence']}){status_tag}")
    else:
        info("  (No semantic facts retrieved for this query)")

    section("Layer 1: Episodic Chunks (vector similarity + temporal + reinforcement)")
    if ctx.episodic_chunks:
        for i, c in enumerate(ctx.episodic_chunks[:5]):
            info(f"  [{i+1}] score={c['score']:.3f} | {c['text'][:80]}...")
    else:
        info("  (No episodic chunks retrieved)")

    section("Layer 3: Behavioral Profile")
    for dim, data in ctx.behavioral_priors.items():
        bar_len = int(data['value'] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        info(f"  {dim:25s} [{bar}] {data['value']:.2f}")

    section("Layer 4: Working Memory")
    if ctx.working_context:
        for k, v in ctx.working_context.items():
            if v:
                info(f"  {k}: {v}")
    else:
        info("  (No active session)")

    section("Dynamic Budget Allocation")
    ba = ctx.budget_allocation
    total = max(ba.get('total', 1), 1)
    for layer, tokens in ba.items():
        if layer != 'total':
            pct = tokens / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            info(f"  {layer:12s} [{bar}] {tokens} tokens ({pct:.0f}%)")
    result("Total budget", f"{ba.get('total', 0)} tokens")
    result("Tokens used", ctx.total_tokens_estimate)

    section("Full Injection Text (what the LLM sees)")
    text = ctx.to_injection_text(profile=orch.behavioral)
    print(f"\n{DIM}{'─'*60}{RESET}")
    for line in text.split("\n"):
        print(f"  {line}")
    print(f"{DIM}{'─'*60}{RESET}")

    # ═══════════════════════════════════════════════════════════════════
    # QUERY 2: Different topic — budget should shift
    # ═══════════════════════════════════════════════════════════════════
    banner("QUERY: 'Tell me about GPU workstations'")

    ctx2 = await orch.query("Tell me about the GPU workstations and CUDA training", top_k=5)

    section("Episodic Results (should favor GPU/CUDA content)")
    for i, c in enumerate(ctx2.episodic_chunks[:3]):
        info(f"  [{i+1}] score={c['score']:.3f} | {c['text'][:80]}...")

    # ═══════════════════════════════════════════════════════════════════
    # RELEVANCE FEEDBACK
    # ═══════════════════════════════════════════════════════════════════
    banner("RELEVANCE FEEDBACK: Learning What's Useful")

    section("Simulating LLM response that uses retrieved context...")
    simulated_response = (
        "Will works on GPU infrastructure, specifically using the GB10 Blackwell "
        "workstation for training ML models with PyTorch and CUDA. He leads the "
        "MEA SCE team and has been researching H2RR for associative memory tasks."
    )
    info(f"Simulated LLM response: \"{simulated_response[:80]}...\"")

    signals = feedback.compute_overlap(ctx.episodic_chunks, simulated_response)
    section("Feedback Signals")
    for s in signals:
        status = f"{GREEN}USED{RESET}" if s.was_used else f"{RED}IGNORED{RESET}"
        info(f"  {status} (overlap: {s.overlap_score:.3f}) | {s.chunk_text[:60]}...")

    fb_result = feedback.apply_feedback(signals, orch.episodic.tai)
    result("Reinforced chunks", fb_result["reinforced"])
    result("Demoted chunks", fb_result["demoted"])
    result("Feedback rate", f"{feedback.get_feedback_rate()['avg_used_ratio']:.0%} chunks useful")

    # ═══════════════════════════════════════════════════════════════════
    # MEMORY CONSOLIDATION
    # ═══════════════════════════════════════════════════════════════════
    banner("MEMORY CONSOLIDATION (like human sleep)")

    section("Running consolidation pass...")
    system_does("Scanning episodic chunks for recurring patterns...")
    system_does("Promoting frequently-mentioned entities to semantic graph...")
    system_does("Checking for dead memories to evict...")
    system_does("Detecting topic clusters from co-retrieval...")

    consolidation_result = await consolidator.consolidate(orch)
    result("Entities promoted", consolidation_result["entities_promoted"])
    result("Relations promoted", consolidation_result["relations_promoted"])
    result("Chunks evicted", consolidation_result["chunks_evicted"])
    result("Topic clusters detected", consolidation_result["topics_detected"])

    # ═══════════════════════════════════════════════════════════════════
    # SYSTEM STATS
    # ═══════════════════════════════════════════════════════════════════
    banner("FINAL SYSTEM STATS")

    stats = orch.stats()
    ep = stats.get("episodic", {})
    sem = stats.get("semantic", {})
    tai = ep.get("tai", {})

    result("Persistence", stats.get("persistence", "file"))
    print()
    result("Episodic chunks (legacy index)", ep.get("count", 0))
    result("Episodic chunks (TAI total)", tai.get("total", 0))
    result("  TAI hot shard", tai.get("hot", 0))
    result("  TAI warm shard", tai.get("warm", 0))
    result("  TAI cold shard", tai.get("cold", 0))
    result("  Co-retrieval links", tai.get("coretrieval_links", 0))
    print()
    result("Semantic entities", sem.get("entities", 0))
    result("Semantic relations", sem.get("relations", 0))
    result("Resolved entities", sem.get("resolved_entities", 0))
    result("Contradictions", f"{sem.get('contradictions_total', 0)} total, {sem.get('contradictions_unresolved', 0)} unresolved")
    print()
    result("Explicit memories", stats.get("explicit", {}).get("count", 0))
    result("Behavioral profile", json.dumps(stats.get("behavioral", {}), indent=None))
    print()

    section("Temporal Associative Index Performance")
    info(f"Search: fused vectorized scoring in single numpy pass")
    info(f"Insert: O(d) amortized with auto-compaction")
    info(f"Dedup+Score: single batch matmul (not N separate searches)")
    info(f"Decay: vectorized exp() across all {tai.get('total', 0)} chunks")

    # Cleanup
    orch.save()
    orch.close()
    if use_ollama:
        await embedder.close()

    banner("DEMO COMPLETE")
    print(f"  Data stored in: {tmpdir}")
    print(f"  To explore: python3 -m cli.main --config config.yaml status")
    print(f"  To serve API: python3 -m cli.main serve")
    print(f"  Dashboard at: http://localhost:8420/dashboard")
    print()


if __name__ == "__main__":
    asyncio.run(main())
