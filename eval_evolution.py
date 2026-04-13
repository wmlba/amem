#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  MEMORY CONTEXT EVOLUTION — Watch the brain grow
═══════════════════════════════════════════════════════════════════════

Three users. Three completely different domains.
Each has a multi-turn conversation.
After each turn, we show what the memory system would inject
into the LLM's context — watching it evolve from empty to rich.
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator
from amem.semantic.graph import Relation

B="\033[1m"; G="\033[92m"; R="\033[91m"; Y="\033[93m"; D="\033[2m"; X="\033[0m"
C="\033[96m"; M="\033[95m"; BL="\033[94m"

def header(text):
    print(f"\n{B}{'━'*70}{X}")
    print(f"{B}  {text}{X}")
    print(f"{B}{'━'*70}{X}")

def turn_header(n, role, text):
    icon = "👤" if role == "user" else "🤖"
    color = BL if role == "user" else G
    print(f"\n  {color}{B}Turn {n} ({role}):{X}")
    # Show first 120 chars
    display = text[:120] + ("..." if len(text) > 120 else "")
    print(f"  {color}{icon} {display}{X}")

def show_memory_state(ctx, orch, label):
    print(f"\n  {M}{B}╔══ MEMORY STATE: {label} ══╗{X}")

    ep = ctx.episodic_chunks
    sem = ctx.semantic_facts
    exp = ctx.explicit_entries
    beh = ctx.behavioral_priors
    wm = ctx.working_context
    ba = ctx.budget_allocation

    # Stats bar
    stats = orch.stats()
    tai = stats.get("episodic", {}).get("tai", {})
    print(f"  {M}║{X} {D}Chunks: {tai.get('total', '?')} | Entities: {stats.get('semantic',{}).get('entities',0)} | Relations: {stats.get('semantic',{}).get('relations',0)} | Explicit: {stats.get('explicit',{}).get('count',0)}{X}")
    print(f"  {M}║{X} {D}Tokens used: {ctx.total_tokens_estimate} / {ba.get('total', 4000)} budget{X}")

    # Explicit
    if exp:
        print(f"  {M}║{X}")
        print(f"  {M}║{X} {Y}{B}EXPLICIT (always present):{X}")
        for e in exp:
            print(f"  {M}║{X}   {Y}• [{e['entry_type']}] {e['key']}: {e['value']}{X}")

    # Semantic
    if sem:
        print(f"  {M}║{X}")
        print(f"  {M}║{X} {C}{B}KNOWLEDGE GRAPH:{X}")
        for f in sem[:5]:
            status = f" {R}[{f.get('status','')}]{X}" if f.get('status','active') != 'active' else ""
            print(f"  {M}║{X}   {C}• {f['subject']} ─[{f['predicate']}]→ {f['object']}  {D}(conf: {f['confidence']}){X}{status}")

    # Episodic
    if ep:
        print(f"  {M}║{X}")
        print(f"  {M}║{X} {G}{B}EPISODIC (top {len(ep)}):{X}")
        for i, c in enumerate(ep[:4]):
            text = c['text'][:90] + ("..." if len(c['text']) > 90 else "")
            print(f"  {M}║{X}   {G}• [{c.get('conversation_id','')[:8]}] {text}{X}")
            print(f"  {M}║{X}     {D}score={c['score']:.3f}{X}")

    # Behavioral
    if beh:
        dims_with_signal = {k: v for k, v in beh.items() if v.get('confidence', 0) > 0.05}
        if dims_with_signal:
            print(f"  {M}║{X}")
            print(f"  {M}║{X} {BL}{B}BEHAVIORAL PROFILE:{X}")
            for dim, data in dims_with_signal.items():
                val = data['value']
                bar_len = int(val * 15)
                bar = "█" * bar_len + "░" * (15 - bar_len)
                print(f"  {M}║{X}   {BL}• {dim:22s} [{bar}] {val:.2f}{X}")

    # Working Memory
    if wm and (wm.get('goals') or wm.get('facts')):
        print(f"  {M}║{X}")
        print(f"  {M}║{X} {R}{B}WORKING MEMORY (this session):{X}")
        for g in wm.get('goals', []):
            print(f"  {M}║{X}   {R}🎯 {g}{X}")
        for f in wm.get('facts', []):
            print(f"  {M}║{X}   {R}📌 {f}{X}")

    print(f"  {M}╚{'═'*50}╝{X}")


async def make_orch():
    config = Config()
    tmpdir = tempfile.mkdtemp()
    config.storage.data_dir = tmpdir
    embedder = create_embedder(config.ollama)
    orch = MemoryOrchestrator(embedder, config)
    orch.init_db(Path(tmpdir) / "amem.db")
    return orch, embedder


async def main():
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  MEMORY CONTEXT EVOLUTION — WATCH THE BRAIN GROW{X}")
    print(f"{B}{'═'*70}{X}")

    # ══════════════════════════════════════════════════════════════
    # TOPIC 1: STARTUP CTO
    # ══════════════════════════════════════════════════════════════
    header("TOPIC 1: STARTUP CTO — Building a fintech product")

    orch, emb = await make_orch()
    now = datetime.now(timezone.utc)
    orch.start_session("cto-session")

    # Turn 1
    msg = "I'm the CTO of a fintech startup called PayFlow. We're building a real-time payment processing platform. Our stack is Go for the backend, React for the frontend, and PostgreSQL for the database."
    turn_header(1, "user", msg)
    await orch.ingest(text=msg, conversation_id="cto-1", speaker="user", timestamp=now - timedelta(minutes=10))
    orch.explicit.set("name", "CTO of PayFlow", entry_type="fact", priority=10)
    orch.explicit.set("company", "PayFlow — fintech startup", entry_type="fact", priority=9)
    orch.working.add_goal("Discuss architecture decisions")

    ctx = await orch.query("What's our tech stack?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 1")

    # Turn 2
    msg = "We're processing 50,000 transactions per second at peak. The main bottleneck is the fraud detection pipeline — it adds 120ms latency to each transaction. We need to get that under 50ms."
    turn_header(2, "user", msg)
    await orch.ingest(text=msg, conversation_id="cto-2", speaker="user", timestamp=now - timedelta(minutes=8))
    orch.working.add_fact("Fraud detection pipeline: 120ms latency, target 50ms")
    orch.working.add_fact("Peak throughput: 50,000 TPS")

    ctx = await orch.query("What's our performance bottleneck?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 2")

    # Turn 3
    msg = "I've decided to move the fraud detection from synchronous to asynchronous. We'll use Kafka for event streaming and run the ML fraud model in a separate consumer group. Transactions will be approved optimistically and flagged post-hoc."
    turn_header(3, "user", msg)
    await orch.ingest(text=msg, conversation_id="cto-3", speaker="user", timestamp=now - timedelta(minutes=5))

    ctx = await orch.query("What's our plan for the fraud detection latency?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 3")

    # Turn 4: Assistant response
    msg = "That's a solid approach. The async pattern with Kafka will decouple the critical transaction path from the fraud analysis. You should consider implementing a circuit breaker between the payment service and the fraud consumer, and add a dead letter queue for failed fraud checks. Also make sure to implement idempotency keys since optimistic approval means you might need to reverse transactions."
    turn_header(4, "assistant", msg)
    await orch.ingest(text=msg, conversation_id="cto-4", speaker="assistant", timestamp=now - timedelta(minutes=3))

    # Turn 5: Follow-up
    msg = "Good points. We already have idempotency keys implemented. The dead letter queue is a great idea — we'll use Kafka's built-in DLQ support. One more thing: we're considering adding Apple Pay and Google Pay. How complex is that integration?"
    turn_header(5, "user", msg)
    await orch.ingest(text=msg, conversation_id="cto-5", speaker="user", timestamp=now)

    ctx = await orch.query("What architecture decisions have we made so far?", top_k=5)
    show_memory_state(ctx, orch, "After Turn 5 — Full Context")

    await orch.end_session()
    orch.close(); await emb.close()

    # ══════════════════════════════════════════════════════════════
    # TOPIC 2: PHD STUDENT
    # ══════════════════════════════════════════════════════════════
    header("TOPIC 2: PHD STUDENT — Writing a dissertation on climate modeling")

    orch, emb = await make_orch()
    now = datetime.now(timezone.utc)
    orch.start_session("phd-session")

    # Turn 1
    msg = "I'm a third-year PhD student studying computational climate science at MIT. My dissertation focuses on improving the resolution of global climate models from 100km to 25km grid spacing using machine learning-based downscaling."
    turn_header(1, "user", msg)
    await orch.ingest(text=msg, conversation_id="phd-1", speaker="user", timestamp=now - timedelta(minutes=15))
    orch.explicit.set("field", "Computational climate science, MIT PhD", entry_type="fact", priority=10)
    orch.explicit.set("dissertation", "ML-based downscaling of global climate models (100km→25km)", entry_type="fact", priority=9)
    orch.working.add_goal("Get feedback on methodology chapter")

    ctx = await orch.query("What's my research about?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 1")

    # Turn 2
    msg = "My approach uses a conditional diffusion model trained on pairs of ERA5 reanalysis data at different resolutions. The model learns to generate high-resolution fields conditioned on low-resolution inputs. I'm getting a RMSE reduction of 34 percent compared to bilinear interpolation."
    turn_header(2, "user", msg)
    await orch.ingest(text=msg, conversation_id="phd-2", speaker="user", timestamp=now - timedelta(minutes=12))
    orch.working.add_fact("Method: conditional diffusion model on ERA5 data")
    orch.working.add_fact("Result: 34% RMSE reduction vs bilinear interpolation")

    ctx = await orch.query("What results am I getting with the diffusion model?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 2")

    # Turn 3
    msg = "My advisor Professor Chen thinks the 34 percent improvement is strong enough for a NeurIPS submission. But she wants me to add ablation studies comparing against U-Net and GAN-based baselines. The deadline is May 17th."
    turn_header(3, "user", msg)
    await orch.ingest(text=msg, conversation_id="phd-3", speaker="user", timestamp=now - timedelta(minutes=9))
    orch.working.add_fact("Advisor: Professor Chen")
    orch.working.add_fact("NeurIPS deadline: May 17th")
    orch.working.add_thread("Need ablation studies: U-Net and GAN baselines")

    ctx = await orch.query("What does my advisor want me to do before the deadline?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 3")

    # Turn 4: Assistant
    msg = "For the ablation studies, I'd suggest structuring them as: (1) U-Net baseline with the same ERA5 training data, (2) a StyleGAN-based approach since it handles spatial coherence well, and (3) a simple CNN regression as a lower bound. Make sure to use the same test set across all methods and report both pixel-level RMSE and perceptual metrics like SSIM."
    turn_header(4, "assistant", msg)
    await orch.ingest(text=msg, conversation_id="phd-4", speaker="assistant", timestamp=now - timedelta(minutes=6))

    # Turn 5
    msg = "Perfect. I'll also add FID score since we're comparing generative approaches. One concern: my diffusion model takes 48 hours to train on 4 A100 GPUs. The U-Net should be much faster. Should I match the compute budget or just let each method train to convergence?"
    turn_header(5, "user", msg)
    await orch.ingest(text=msg, conversation_id="phd-5", speaker="user", timestamp=now)

    ctx = await orch.query("What's the full picture of my research, deadlines, and next steps?", top_k=5)
    show_memory_state(ctx, orch, "After Turn 5 — Full Context")

    await orch.end_session()
    orch.close(); await emb.close()

    # ══════════════════════════════════════════════════════════════
    # TOPIC 3: EMERGENCY PHYSICIAN
    # ══════════════════════════════════════════════════════════════
    header("TOPIC 3: EMERGENCY PHYSICIAN — Clinical workflow optimization")

    orch, emb = await make_orch()
    now = datetime.now(timezone.utc)
    orch.start_session("er-session")

    # Turn 1
    msg = "I'm an attending physician in the emergency department at Mass General Hospital. Our ED sees about 300 patients per day. I'm leading a quality improvement project to reduce door-to-doctor time from 45 minutes to under 20 minutes."
    turn_header(1, "user", msg)
    await orch.ingest(text=msg, conversation_id="er-1", speaker="user", timestamp=now - timedelta(minutes=20))
    orch.explicit.set("role", "Attending physician, Mass General ED", entry_type="fact", priority=10)
    orch.explicit.set("project", "Reduce door-to-doctor time from 45min to <20min", entry_type="fact", priority=9)
    orch.explicit.set("preference", "Evidence-based recommendations with citation years", entry_type="instruction", priority=8)
    orch.working.add_goal("Design the QI intervention")

    ctx = await orch.query("What am I working on?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 1")

    # Turn 2
    msg = "The main bottleneck we've identified is the triage process. Our current system uses the Emergency Severity Index version 4. Nurses spend an average of 8 minutes per patient on triage documentation. The waiting room backs up during shift changes between 3pm and 4pm."
    turn_header(2, "user", msg)
    await orch.ingest(text=msg, conversation_id="er-2", speaker="user", timestamp=now - timedelta(minutes=15))
    orch.working.add_fact("Bottleneck: triage — 8min per patient, ESI v4")
    orch.working.add_fact("Peak backup: 3pm-4pm shift changes")

    ctx = await orch.query("What's causing the delay in our emergency department?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 2")

    # Turn 3
    msg = "We're piloting a split-flow model where ESI level 4 and 5 patients go directly to a fast-track area staffed by a PA. Initial data from the first two weeks shows a 31 percent reduction in overall door-to-doctor time. But the nurses are concerned about patient safety with less comprehensive triage."
    turn_header(3, "user", msg)
    await orch.ingest(text=msg, conversation_id="er-3", speaker="user", timestamp=now - timedelta(minutes=10))
    orch.working.add_fact("Split-flow pilot: 31% reduction in door-to-doctor time")
    orch.working.add_thread("Nursing concern: patient safety with abbreviated triage")

    ctx = await orch.query("What are the results of our pilot program and what concerns have been raised?", top_k=3)
    show_memory_state(ctx, orch, "After Turn 3")

    # Turn 4: Assistant
    msg = "The 31 percent reduction is promising and aligns with published literature on split-flow models. To address the nursing concerns, consider implementing a standardized safety checklist for the fast-track pathway — focusing on vital signs, chief complaint red flags, and medication reconciliation. Also track bounce-back rates for fast-track patients who return within 72 hours."
    turn_header(4, "assistant", msg)
    await orch.ingest(text=msg, conversation_id="er-4", speaker="assistant", timestamp=now - timedelta(minutes=5))

    # Turn 5
    msg = "Great suggestion on the bounce-back tracking. Our current 72-hour return rate is 4.2 percent overall. I'll need to segment that by ESI level to see if fast-track patients have a higher rate. Also, our chief of medicine Dr. Williams wants to present the preliminary results at the hospital quality committee meeting next Thursday."
    turn_header(5, "user", msg)
    await orch.ingest(text=msg, conversation_id="er-5", speaker="user", timestamp=now)

    ctx = await orch.query("Summarize our QI project: goals, results, concerns, and next steps", top_k=5)
    show_memory_state(ctx, orch, "After Turn 5 — Full Context")

    # ── Show the full injection text ──
    print(f"\n  {B}FULL CONTEXT INJECTION (what the LLM would see):{X}")
    injection = ctx.to_injection_text(profile=orch.behavioral)
    print(f"  {D}{'─'*60}{X}")
    for line in injection.split("\n"):
        print(f"  {line}")
    print(f"  {D}{'─'*60}{X}")
    print(f"  {D}Total: {ctx.total_tokens_estimate} tokens{X}")

    await orch.end_session()
    orch.close(); await emb.close()

    # ══════════════════════════════════════════════════════════════
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  WHAT YOU JUST SAW{X}")
    print(f"{B}{'═'*70}{X}")
    print(f"""
  Three completely different domains. Five turns each.

  After each turn, the memory state evolved:
    Turn 1: Basic identity + first topic context
    Turn 2: Working facts accumulate, behavioral profile starts building
    Turn 3: Knowledge graph grows, working threads track open questions
    Turn 4: Assistant responses ingested — system remembers what it said
    Turn 5: Full rich context with history, facts, profile, and session state

  Key observations:
  • {G}Explicit memory{X} (name, role, preferences) appeared in EVERY query
  • {C}Knowledge graph{X} built entity relationships across turns
  • {G}Episodic chunks{X} ranked by relevance to each specific query
  • {BL}Behavioral profile{X} adapted to each user's communication style
  • {R}Working memory{X} tracked session goals and established facts

  The LLM never sees raw conversation history.
  It sees a {B}structured, query-conditioned memory context{X}
  that evolves with every interaction.
""")


if __name__ == "__main__":
    asyncio.run(main())
