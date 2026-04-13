#!/usr/bin/env python3
"""
Full realistic test against live Ollama with real embeddings.
No mocks. No shortcuts. Proves every claim.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator
from amem.feedback.relevance import RelevanceFeedback
from amem.maintenance.consolidation import MemoryConsolidator
from amem.semantic.graph import Relation

B = "\033[1m"; G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; D = "\033[2m"; X = "\033[0m"; C = "\033[96m"
passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  {G}PASS{X}  {name}")
        passed += 1
    else:
        print(f"  {R}FAIL{X}  {name}  {D}{detail}{X}")
        failed += 1


async def main():
    global passed, failed

    print(f"\n{B}{'='*70}{X}")
    print(f"{B}  FULL REALISTIC TEST — LIVE OLLAMA, REAL EMBEDDINGS{X}")
    print(f"{B}{'='*70}{X}\n")

    # ── Setup ─────────────────────────────────────────────────────
    config = Config()
    tmpdir = tempfile.mkdtemp()
    config.storage.data_dir = tmpdir

    t0 = time.monotonic()
    embedder = create_embedder(config.ollama)
    orch = MemoryOrchestrator(embedder, config)
    orch.init_db(Path(tmpdir) / "amem.db")
    setup_time = time.monotonic() - t0
    print(f"  {D}Setup: {setup_time:.2f}s | DB: {tmpdir}/amem.db{X}\n")

    # ══════════════════════════════════════════════════════════════
    print(f"{C}{B}  1. SEMANTIC RETRIEVAL — does similarity actually work?{X}\n")
    # ══════════════════════════════════════════════════════════════

    await orch.ingest(text="Machine learning models require careful hyperparameter tuning. Grid search and Bayesian optimization are standard approaches for finding optimal learning rates.", conversation_id="ml", speaker="user")
    await orch.ingest(text="The kitchen renovation costs forty-five thousand dollars. We chose granite countertops and custom oak cabinets with soft-close hinges.", conversation_id="kitchen", speaker="user")
    await orch.ingest(text="My stock portfolio is sixty percent equities and forty percent bonds. I rebalance quarterly using a momentum strategy.", conversation_id="finance", speaker="user")
    await orch.ingest(text="I practice yoga three times a week and do guided meditation every morning for twenty minutes.", conversation_id="yoga", speaker="user")

    ctx_ml = await orch.query("How should I tune my model's learning rate?", top_k=4)
    top_ml = ctx_ml.episodic_chunks[0]["text"] if ctx_ml.episodic_chunks else ""
    test("ML query retrieves ML content", "hyperparameter" in top_ml.lower() or "learning" in top_ml.lower(), f"got: {top_ml[:60]}")

    ctx_kitchen = await orch.query("What did the granite countertops and oak cabinets cost?", top_k=4)
    top_kitchen = ctx_kitchen.episodic_chunks[0]["text"] if ctx_kitchen.episodic_chunks else ""
    test("Kitchen query retrieves kitchen content", "kitchen" in top_kitchen.lower() or "granite" in top_kitchen.lower() or "cabinet" in top_kitchen.lower(), f"got: {top_kitchen[:60]}")

    ctx_fin = await orch.query("stocks bonds equities portfolio rebalance quarterly momentum", top_k=4)
    all_fin = " ".join(c["text"] for c in ctx_fin.episodic_chunks[:2]).lower()
    test("Finance query finds finance content in top 2", "portfolio" in all_fin or "equit" in all_fin or "bond" in all_fin, f"got: {all_fin[:80]}")

    ctx_yoga = await orch.query("yoga meditation practice morning guided session mindfulness", top_k=4)
    all_yoga = " ".join(c["text"] for c in ctx_yoga.episodic_chunks[:2]).lower()
    test("Yoga query finds yoga content in top 2", "yoga" in all_yoga or "meditation" in all_yoga, f"got: {all_yoga[:80]}")

    # Verify irrelevant content is NOT in top result
    test("ML query does NOT return kitchen content", "kitchen" not in top_ml.lower() and "granite" not in top_ml.lower(), f"got: {top_ml[:60]}")
    test("Kitchen query does NOT return finance content", "portfolio" not in top_kitchen.lower() and "bond" not in top_kitchen.lower())

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  2. DETAIL PRESERVATION — no lossy summarization{X}\n")
    # ══════════════════════════════════════════════════════════════

    await orch.ingest(text="Our training config uses learning rate 3e-4 with cosine annealing, batch size 256, on 4x A100 80GB GPUs with mixed precision BF16.", conversation_id="config", speaker="user")

    ctx_detail = await orch.query("learning rate 3e-4 batch size A100 GPU BF16 cosine annealing config", top_k=5)
    detail_text = " ".join(c["text"] for c in ctx_detail.episodic_chunks)
    test("Learning rate 3e-4 preserved", "3e-4" in detail_text, f"chunks: {len(ctx_detail.episodic_chunks)}")
    test("Batch size 256 preserved", "256" in detail_text)
    test("A100 80GB preserved", "A100" in detail_text and "80GB" in detail_text)
    test("BF16 preserved", "BF16" in detail_text)

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  3. CONTEXT EFFICIENCY — only relevant tokens injected{X}\n")
    # ══════════════════════════════════════════════════════════════

    ctx_narrow = await orch.query("meditation routine", top_k=3)
    injection = ctx_narrow.to_injection_text(profile=orch.behavioral)
    test("Injection text < 1000 tokens for narrow query", ctx_narrow.total_tokens_estimate < 1000, f"used {ctx_narrow.total_tokens_estimate}")
    test("Budget allocation computed", ctx_narrow.budget_allocation.get("total", 0) > 0)
    test("Episodic budget > 0", ctx_narrow.budget_allocation.get("episodic", 0) > 0)

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  4. EXPLICIT MEMORY — highest priority, always present{X}\n")
    # ══════════════════════════════════════════════════════════════

    orch.explicit.set("name", "Will", entry_type="fact", priority=10)
    orch.explicit.set("role", "Senior ML Engineer", entry_type="fact", priority=9)
    orch.explicit.set("instruction", "Always show code examples", entry_type="instruction", priority=8)

    ctx_any = await orch.query("random unrelated query about butterflies", top_k=3)
    injection_any = ctx_any.to_injection_text(profile=orch.behavioral)
    test("Name appears in ANY query", "Will" in injection_any)
    test("Role appears in ANY query", "Senior ML Engineer" in injection_any)
    test("Instruction appears in ANY query", "code examples" in injection_any)
    test("Explicit entries count correct", len(ctx_any.explicit_entries) == 3)

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  5. CONTRADICTION DETECTION — temporal reasoning{X}\n")
    # ══════════════════════════════════════════════════════════════

    now = datetime.now(timezone.utc)
    old = now - timedelta(days=180)

    orch.semantic.add_relation(Relation(subject="Will", predicate="works_at", object="OCI", confidence=0.8, first_seen=old, last_seen=old))
    contradictions = orch.semantic.add_relation(Relation(subject="Will", predicate="works_at", object="Anthropic", confidence=0.8, first_seen=now, last_seen=now))

    test("Contradiction detected", len(contradictions) > 0, f"got {len(contradictions)} contradictions")
    if contradictions:
        test("Type is direct", contradictions[0].contradiction_type == "direct")
        test("Resolution is newer_wins", contradictions[0].resolution == "newer_wins")
        test("Winner is B (newer)", contradictions[0].winner == "b")

    facts = orch.semantic.query(["Will"], max_depth=1)
    active_works_at = [f for f in facts if f["predicate"] == "works_at" and f.get("status", "active") == "active"]
    test("Active fact is Anthropic", any("Anthropic" in f["object"] for f in active_works_at), f"active: {active_works_at}")

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  6. ENTITY EXTRACTION — using same embedding model{X}\n")
    # ══════════════════════════════════════════════════════════════

    r = await orch.ingest(text="Alice from Google uses Python and PyTorch for deep learning research on transformer architectures.", speaker="user")
    test("Entities extracted > 0", r["entities_extracted"] > 0, f"got {r['entities_extracted']}")

    entities = orch.semantic.get_entities()
    names_lower = [e.get("name", "").lower() for e in entities]
    test("Python detected as entity", any("python" in n for n in names_lower), f"entities: {names_lower[:10]}")

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  7. ENTITY RESOLUTION — aliases and fuzzy matching{X}\n")
    # ══════════════════════════════════════════════════════════════

    orch.semantic.resolver.register("GB10", entity_type="tool")
    orch.add_entity_alias("GB10", "Blackwell workstation")
    orch.add_entity_alias("GB10", "the GPU workstation")

    resolved1 = orch.semantic.resolver.resolve("Blackwell workstation")
    test("'Blackwell workstation' resolves to GB10", resolved1 is not None and resolved1.canonical_name == "GB10")

    resolved2 = orch.semantic.resolver.resolve("the GPU workstation")
    test("'the GPU workstation' resolves to GB10", resolved2 is not None and resolved2.canonical_name == "GB10")

    resolved3 = orch.semantic.resolver.resolve("GB10")
    test("'GB10' resolves to itself", resolved3 is not None and resolved3.canonical_name == "GB10")

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  8. FACT RETRACTION — user corrections{X}\n")
    # ══════════════════════════════════════════════════════════════

    orch.semantic.add_relation(Relation(subject="Will", predicate="uses", object="TensorFlow"))
    retracted = orch.retract_fact("Will", "uses", "TensorFlow")
    test("Retraction succeeds", retracted)

    facts_after = orch.semantic.query(["Will"])
    tf_active = [f for f in facts_after if f["object"] == "TensorFlow" and f.get("status") == "active"]
    test("Retracted fact not in active results", len(tf_active) == 0)

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  9. RELEVANCE FEEDBACK — learning from LLM usage{X}\n")
    # ══════════════════════════════════════════════════════════════

    feedback = RelevanceFeedback(min_overlap_to_count_as_used=0.08)
    ctx_fb = await orch.query("machine learning hyperparameter tuning", top_k=5)

    # Simulate LLM response that reuses words from the retrieved chunks
    chunk_words = set()
    for c in ctx_fb.episodic_chunks[:3]:
        chunk_words.update(c["text"].lower().split())
    # Build a response that shares significant vocabulary with retrieved chunks
    shared = list(chunk_words)[:20]
    simulated_response = "Based on what I know: " + " ".join(shared) + ". That covers the key points about your setup."
    signals = feedback.compute_overlap(ctx_fb.episodic_chunks, simulated_response)

    used_count = sum(1 for s in signals if s.was_used)
    test("Feedback detects used chunks", used_count > 0, f"used: {used_count}/{len(signals)}, overlap: {[f'{s.overlap_score:.3f}' for s in signals[:3]]}")

    if signals:
        fb_result = feedback.apply_feedback(signals, orch.episodic.tai)
        test("Reinforcement applied", fb_result["reinforced"] > 0 or fb_result["demoted"] > 0)

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  10. WORKING MEMORY — session scratchpad{X}\n")
    # ══════════════════════════════════════════════════════════════

    orch.start_session("test-session")
    orch.working.add_goal("Debug the training pipeline")
    orch.working.add_fact("Loss spiked at epoch 47")
    orch.working.add_thread("Investigating gradient explosion")

    ctx_wm = await orch.query("What are we debugging?", top_k=3)
    wm = ctx_wm.working_context
    test("Working memory has goals", len(wm.get("goals", [])) > 0)
    test("Working memory has facts", len(wm.get("facts", [])) > 0)

    injection_wm = ctx_wm.to_injection_text(profile=orch.behavioral)
    test("Working memory in injection text", "Debug the training pipeline" in injection_wm)
    test("Session facts in injection text", "epoch 47" in injection_wm)

    sid = await orch.end_session()
    test("Session flushed to episodic", sid == "test-session")
    test("Working memory cleared", orch.working.is_empty)

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  11. BEHAVIORAL PROFILE — adapts to user{X}\n")
    # ══════════════════════════════════════════════════════════════

    for _ in range(10):
        await orch.ingest(text="Optimize the distributed training pipeline by implementing gradient accumulation with ZeRO stage 3 and activation checkpointing to reduce memory footprint across the cluster nodes.", speaker="user")

    priors = orch.behavioral.get_priors()
    test("Domain expertise signal built", priors["domain_expertise"]["confidence"] > 0.2)
    test("Domain expertise value elevated", priors["domain_expertise"]["value"] > 0.1)

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  12. PERSISTENCE — survives restart{X}\n")
    # ══════════════════════════════════════════════════════════════

    db_path = Path(tmpdir) / "amem.db"
    chunk_count_before = orch.episodic.tai.count
    explicit_count_before = orch.explicit.count
    orch.save()
    orch.close()

    # New orchestrator, same DB
    embedder2 = create_embedder(config.ollama)
    orch2 = MemoryOrchestrator(embedder2, config)
    orch2.init_db(db_path)
    orch2.load()

    test("Episodic chunks survive restart", orch2.episodic.index.count > 0, f"count: {orch2.episodic.index.count}")
    test("Explicit memories survive restart", orch2.explicit.count == explicit_count_before)
    test("Explicit value correct after restart", orch2.explicit.get("name").value == "Will")

    ctx_after = await orch2.query("What GPU setup do we use?", top_k=3)
    test("Queries work after restart", len(ctx_after.episodic_chunks) > 0)
    orch2.close()
    await embedder.close()
    await embedder2.close()

    # ══════════════════════════════════════════════════════════════
    print(f"\n{C}{B}  13. PERFORMANCE{X}\n")
    # ══════════════════════════════════════════════════════════════

    embedder3 = create_embedder(config.ollama)
    orch3 = MemoryOrchestrator(embedder3, config)
    orch3.init_db(Path(tmpdir) / "perf.db")

    # Ingest latency
    t0 = time.monotonic()
    for i in range(10):
        await orch3.ingest(text=f"Performance test message number {i} about distributed systems and GPU training pipelines.", speaker="user")
    ingest_time = (time.monotonic() - t0) / 10
    test(f"Avg ingest latency < 2s", ingest_time < 2.0, f"{ingest_time:.3f}s")

    # Query latency
    t0 = time.monotonic()
    for _ in range(5):
        await orch3.query("distributed training performance")
    query_time = (time.monotonic() - t0) / 5
    test(f"Avg query latency < 1s", query_time < 1.0, f"{query_time:.3f}s")

    orch3.close()
    await embedder3.close()

    # ══════════════════════════════════════════════════════════════
    print(f"\n{B}{'='*70}{X}")
    total = passed + failed
    if failed == 0:
        print(f"{G}{B}  ALL {total} TESTS PASSED{X}")
    else:
        print(f"{R}{B}  {failed} FAILED{X} / {G}{passed} PASSED{X} / {total} total")
    print(f"{B}{'='*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
