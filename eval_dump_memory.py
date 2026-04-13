#!/usr/bin/env python3
"""
Runs one conversation (the Startup Founder), then dumps
the COMPLETE memory state — every chunk, every entity,
every relation, every explicit entry, the full behavioral
profile, and the raw SQLite tables.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator
from amem.persistence.sqlite import SQLiteStore

B="\033[1m"; G="\033[92m"; Y="\033[93m"; D="\033[2m"; X="\033[0m"; C="\033[96m"; M="\033[95m"; BL="\033[94m"; R="\033[91m"


async def main():
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  FULL MEMORY DUMP — Everything stored after a conversation{X}")
    print(f"{B}{'═'*70}{X}")

    config = Config()
    tmpdir = tempfile.mkdtemp()
    config.storage.data_dir = tmpdir
    db_path = Path(tmpdir) / "amem.db"
    embedder = create_embedder(config.ollama)
    orch = MemoryOrchestrator(embedder, config)
    orch.init_db(db_path)

    # ── Run a conversation ──────────────────────────────────────
    print(f"\n{D}  Running conversation (Startup Founder, 4 turns)...{X}")

    orch.start_session("session-1")

    # Turn 1
    await orch.ingest(text="I'm building an AI tutoring startup called LearnLoop. We use LLMs to create personalized lesson plans for K-12 students. Our differentiator is that we adapt in real-time based on student mistakes.", conversation_id="conv-1", speaker="user")
    await orch.ingest(text="LearnLoop sounds promising. What subjects are you starting with? What's your go-to-market — direct to schools or parents? How are you handling COPPA?", conversation_id="conv-1", speaker="assistant")

    orch.explicit.set("company", "LearnLoop — AI tutoring for K-12", entry_type="fact", priority=10)
    orch.explicit.set("differentiator", "Real-time adaptation based on student mistakes", entry_type="fact", priority=9)
    orch.explicit.set("style", "Be direct, use bullet points, cite metrics", entry_type="instruction", priority=8)

    # Turn 2
    await orch.ingest(text="Starting with math for grades 6-8. Going direct to parents first — schools have 18-month sales cycles. We're COPPA compliant with parental consent. We have 200 beta users and 40 percent weekly retention.", conversation_id="conv-2", speaker="user")
    await orch.ingest(text="40 percent weekly retention is above the industry benchmark of 25-30 percent. Your direct-to-parent strategy avoids the long school sales cycle. Track time-to-first-lesson-completion and mistake pattern diversity.", conversation_id="conv-2", speaker="assistant")

    orch.working.add_fact("200 beta users, 40% weekly retention")
    orch.working.add_fact("Math grades 6-8, direct to parents")
    orch.working.add_goal("Prepare seed round pitch")

    # Turn 3
    await orch.ingest(text="We're raising 2 million at a 10 million pre-money valuation. Our monthly burn is 45K. The team is 4 people: me as CEO, a CTO, and two ML engineers.", conversation_id="conv-3", speaker="user")
    await orch.ingest(text="At 2M on 10M pre-money you get 12M post-money, roughly 44 months runway at 45K burn. The valuation is defensible for edtech seed rounds. Key risk: 200 users is small — show a path to 2000 in 6 months.", conversation_id="conv-3", speaker="assistant")

    # Turn 4
    await orch.ingest(text="Our CAC is 35 dollars through Instagram ads targeting parents of middle schoolers. LTV is estimated at 180 dollars based on 6-month average subscription length at 30 dollars per month.", conversation_id="conv-4", speaker="user")
    await orch.ingest(text="CAC of 35 and LTV of 180 gives you a 5.1x LTV/CAC ratio — that's excellent, well above the 3x benchmark for healthy unit economics. This will be a strong slide in your pitch deck.", conversation_id="conv-4", speaker="assistant")

    orch.working.add_fact("CAC $35, LTV $180, LTV/CAC 5.1x")
    orch.working.add_fact("Team: CEO, CTO, 2 ML engineers")

    await orch.end_session()
    orch.save()

    print(f"  {G}Done. Now dumping complete memory state.{X}\n")

    # ══════════════════════════════════════════════════════════════
    # DUMP EVERYTHING
    # ══════════════════════════════════════════════════════════════

    # ── 1. EXPLICIT MEMORY ──────────────────────────────────────
    print(f"{Y}{B}{'─'*70}{X}")
    print(f"{Y}{B}  LAYER 5: EXPLICIT MEMORY (user-controlled, never decays){X}")
    print(f"{Y}{B}{'─'*70}{X}")
    for e in orch.explicit.list_all():
        print(f"  {Y}key:{X}        {e.key}")
        print(f"  {Y}value:{X}      {e.value}")
        print(f"  {Y}type:{X}       {e.entry_type}")
        print(f"  {Y}priority:{X}   {e.priority}")
        print(f"  {Y}created:{X}    {e.created}")
        print(f"  {Y}updated:{X}    {e.updated}")
        print()

    # ── 2. EPISODIC MEMORY (TAI) ────────────────────────────────
    print(f"{G}{B}{'─'*70}{X}")
    print(f"{G}{B}  LAYER 1: EPISODIC MEMORY (every stored chunk){X}")
    print(f"{G}{B}{'─'*70}{X}")

    tai = orch.episodic.tai
    stats = tai.stats()
    print(f"  {D}Total chunks: {stats['total']} (hot: {stats['hot']}, warm: {stats['warm']}, cold: {stats['cold']}){X}")
    print(f"  {D}Co-retrieval links: {stats['coretrieval_links']}{X}")
    print()

    chunk_num = 0
    for shard_name, shard in [("HOT", tai._hot), ("WARM", tai._warm), ("COLD", tai._cold)]:
        for i in range(shard.count):
            chunk_num += 1
            meta = shard._metadata[i]
            print(f"  {G}Chunk {chunk_num} [{shard_name}]{X}")
            print(f"  {G}id:{X}              {meta.chunk_id}")
            print(f"  {G}text:{X}            {meta.text[:100]}{'...' if len(meta.text) > 100 else ''}")
            print(f"  {G}conversation:{X}    {meta.conversation_id}")
            print(f"  {G}speaker:{X}         {meta.speaker}")
            print(f"  {G}timestamp:{X}       {datetime.fromtimestamp(meta.timestamp, tz=timezone.utc).isoformat()}")
            print(f"  {G}confidence:{X}      {meta.confidence:.4f}")
            print(f"  {G}importance:{X}      {meta.importance:.4f}")
            print(f"  {G}access_count:{X}    {meta.access_count}")
            print(f"  {G}entities:{X}        {meta.entity_mentions}")
            print(f"  {G}tags:{X}            {meta.topic_tags}")
            print()

    # ── 3. SEMANTIC GRAPH ───────────────────────────────────────
    print(f"{C}{B}{'─'*70}{X}")
    print(f"{C}{B}  LAYER 2: SEMANTIC GRAPH (entities + relations){X}")
    print(f"{C}{B}{'─'*70}{X}")

    entities = orch.semantic.get_entities()
    print(f"  {D}Entities: {len(entities)}{X}")
    print()
    for e in entities:
        print(f"  {C}Entity:{X} {e.get('name', e.get('key', '?'))}")
        print(f"    type:       {e.get('entity_type', '?')}")
        print(f"    mentions:   {e.get('mention_count', 1)}")
        print(f"    aliases:    {e.get('aliases', [])}")
        print(f"    first_seen: {e.get('first_seen', '?')}")
        print(f"    last_seen:  {e.get('last_seen', '?')}")
        print()

    # Relations
    sem_stats = orch.semantic.stats()
    print(f"  {D}Relations: {sem_stats.get('relations', 0)}{X}")
    # Query all entities for their relations
    all_facts = []
    for e in entities:
        name = e.get('name', e.get('key', ''))
        facts = orch.semantic.query([name], max_depth=1)
        for f in facts:
            fact_key = (f['subject'], f['predicate'], f['object'])
            if fact_key not in [(af['subject'], af['predicate'], af['object']) for af in all_facts]:
                all_facts.append(f)
    print()
    for f in all_facts:
        status_tag = f"  {R}[{f.get('status','')}]{X}" if f.get('status', 'active') != 'active' else ""
        print(f"  {C}{f['subject']} ─[{f['predicate']}]→ {f['object']}{X}")
        print(f"    confidence: {f['confidence']}")
        print(f"    mentions:   {f.get('mention_count', 1)}")
        print(f"    last_seen:  {f.get('last_seen', '?')}{status_tag}")
        print()

    # Entity resolver
    print(f"  {D}Resolved canonical entities: {orch.semantic.resolver.entity_count}{X}")
    for ce in orch.semantic.resolver.get_all_entities():
        if ce.aliases:
            print(f"    {ce.canonical_name} → aliases: {ce.aliases}")

    # Contradictions
    contras = orch.semantic.get_contradictions()
    if contras:
        print(f"\n  {R}Contradictions: {len(contras)}{X}")
        for c in contras:
            print(f"    {c}")

    # ── 4. BEHAVIORAL PROFILE ───────────────────────────────────
    print(f"\n{BL}{B}{'─'*70}{X}")
    print(f"{BL}{B}  LAYER 3: BEHAVIORAL PROFILE{X}")
    print(f"{BL}{B}{'─'*70}{X}")

    priors = orch.behavioral.get_priors()
    summary = orch.behavioral.get_summary()
    for dim in priors:
        data = priors[dim]
        desc = summary[dim]
        bar_len = int(data['value'] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {BL}{dim}{X}")
        print(f"    value:      {data['value']:.4f}")
        print(f"    confidence: {data['confidence']:.4f}")
        print(f"    signals:    {data['n_signals']} samples")
        print(f"    visual:     [{bar}]")
        print(f"    summary:    {desc}")
        print()

    # ── 5. WORKING MEMORY ───────────────────────────────────────
    print(f"{R}{B}{'─'*70}{X}")
    print(f"{R}{B}  LAYER 4: WORKING MEMORY (flushed to episodic at session end){X}")
    print(f"{R}{B}{'─'*70}{X}")
    print(f"  {D}Session ended — working memory was flushed to episodic store.{X}")
    print(f"  {D}The flushed content is now in episodic chunks above (speaker='system').{X}")

    # ── 6. SQLite RAW ───────────────────────────────────────────
    print(f"\n{M}{B}{'─'*70}{X}")
    print(f"{M}{B}  SQLite DATABASE (raw table counts){X}")
    print(f"{M}{B}{'─'*70}{X}")

    db = SQLiteStore(db_path)
    db_stats = db.stats()
    for table, count in db_stats.items():
        print(f"  {M}{table:25s}{X} {count} rows")
    print(f"\n  {D}Database file: {db_path}{X}")
    import os
    size = os.path.getsize(db_path)
    print(f"  {D}Database size: {size / 1024:.1f} KB{X}")
    db.close()

    # ── 7. Query demo — show what gets injected ─────────────────
    print(f"\n{B}{'─'*70}{X}")
    print(f"{B}  QUERY DEMO: 'What are LearnLoop\\'s unit economics?'{X}")
    print(f"{B}{'─'*70}{X}")

    ctx = await orch.query("What are LearnLoop's unit economics and fundraising plan?", top_k=5)
    injection = ctx.to_injection_text(profile=orch.behavioral)

    print(f"\n  {D}Tokens used: {ctx.total_tokens_estimate} / {ctx.budget_allocation.get('total', 4000)}{X}")
    print(f"  {D}Budget: episodic={ctx.budget_allocation.get('episodic',0)} semantic={ctx.budget_allocation.get('semantic',0)} explicit={ctx.budget_allocation.get('explicit',0)}{X}")
    print(f"\n  {B}FULL INJECTION TEXT:{X}")
    print(f"  {D}{'─'*60}{X}")
    for line in injection.split("\n"):
        print(f"  {line}")
    print(f"  {D}{'─'*60}{X}")

    orch.close()
    await embedder.close()

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  MEMORY DUMP COMPLETE{X}")
    print(f"{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
