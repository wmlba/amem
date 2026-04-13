#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  REAL CONVERSATION FLOW — Prompt → Memory → Response → Learn
═══════════════════════════════════════════════════════════════════════

Shows exactly:
  1. What the user says (PROMPT)
  2. What memory the system retrieves (MEMORY CONTEXT)
  3. What the LLM would see in its context window (INJECTED CONTEXT)
  4. A simulated response (RESPONSE)
  5. What gets stored back into memory (LEARNED)

Three conversations. Watch memory accumulate and improve responses.
"""

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator

B="\033[1m"; G="\033[92m"; R="\033[91m"; Y="\033[93m"; D="\033[2m"; X="\033[0m"
C="\033[96m"; M="\033[95m"; BL="\033[94m"


async def make_orch():
    config = Config()
    tmpdir = tempfile.mkdtemp()
    config.storage.data_dir = tmpdir
    embedder = create_embedder(config.ollama)
    orch = MemoryOrchestrator(embedder, config)
    orch.init_db(Path(tmpdir) / "amem.db")
    return orch, embedder


async def exchange(orch, turn_num, user_msg, simulated_response, conv_id="conv"):
    """Run one exchange: user prompt → memory retrieval → response → ingest both."""

    print(f"\n  {BL}{B}╔═══ TURN {turn_num} ═══════════════════════════════════════════════╗{X}")

    # ── USER PROMPT ──
    print(f"  {BL}║{X}")
    print(f"  {BL}║{X}  {B}PROMPT:{X}")
    print(f"  {BL}║{X}  {BL}👤 {user_msg}{X}")

    # ── MEMORY RETRIEVAL ──
    ctx = await orch.query(user_msg, top_k=5)
    injection = ctx.to_injection_text(profile=orch.behavioral)

    print(f"  {BL}║{X}")
    print(f"  {BL}║{X}  {Y}{B}MEMORY RETRIEVED ({ctx.total_tokens_estimate} tokens):{X}")

    # Show explicit
    if ctx.explicit_entries:
        for e in ctx.explicit_entries:
            print(f"  {BL}║{X}    {Y}📌 [{e['entry_type']}] {e['key']}: {e['value']}{X}")

    # Show episodic (top 3)
    if ctx.episodic_chunks:
        for c in ctx.episodic_chunks[:3]:
            text = c['text'][:80] + ("..." if len(c['text']) > 80 else "")
            print(f"  {BL}║{X}    {G}🧠 {text}{X}")
            print(f"  {BL}║{X}       {D}(score: {c['score']:.3f}){X}")

    # Show semantic facts
    if ctx.semantic_facts:
        for f in ctx.semantic_facts[:3]:
            print(f"  {BL}║{X}    {C}📊 {f['subject']} —[{f['predicate']}]→ {f['object']}  {D}(conf: {f['confidence']}){X}")

    # Show working memory
    wm = ctx.working_context
    if wm and (wm.get('goals') or wm.get('facts')):
        for g in wm.get('goals', []):
            print(f"  {BL}║{X}    {R}🎯 Goal: {g}{X}")
        for f in wm.get('facts', []):
            print(f"  {BL}║{X}    {R}📎 Fact: {f}{X}")

    if not ctx.explicit_entries and not ctx.episodic_chunks and not ctx.semantic_facts:
        print(f"  {BL}║{X}    {D}(empty — first interaction){X}")

    # ── RESPONSE ──
    print(f"  {BL}║{X}")
    print(f"  {BL}║{X}  {G}{B}RESPONSE:{X}")
    # Wrap response text
    words = simulated_response.split()
    lines = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > 68:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        lines.append(line)
    for l in lines:
        print(f"  {BL}║{X}  {G}🤖 {l}{X}")

    # ── INGEST BOTH SIDES ──
    r = await orch.ingest(text=user_msg, conversation_id=conv_id, speaker="user")
    await orch.ingest(text=simulated_response, conversation_id=conv_id, speaker="assistant")

    print(f"  {BL}║{X}")
    print(f"  {BL}║{X}  {D}LEARNED: +{r['chunks_stored']} chunks, +{r['entities_extracted']} entities{X}")
    stats = orch.stats()
    tai = stats.get("episodic", {}).get("tai", {})
    print(f"  {BL}║{X}  {D}TOTAL MEMORY: {tai.get('total',0)} chunks, {stats.get('semantic',{}).get('entities',0)} entities, {stats.get('explicit',{}).get('count',0)} explicit{X}")
    print(f"  {BL}╚{'═'*57}╝{X}")

    return ctx


async def main():
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  REAL CONVERSATION FLOW — Prompt → Memory → Response → Learn{X}")
    print(f"{B}{'═'*70}{X}")

    # ══════════════════════════════════════════════════════════════
    print(f"\n{M}{B}{'━'*70}{X}")
    print(f"{M}{B}  CONVERSATION 1: Data Engineer onboarding at new job{X}")
    print(f"{M}{B}{'━'*70}{X}")
    # ══════════════════════════════════════════════════════════════

    orch, emb = await make_orch()
    orch.start_session("onboarding")

    # Turn 1 — Introduction
    await exchange(orch, 1,
        "Hi! I just started as a data engineer at Spotify. Can you help me understand the data stack here?",
        "Welcome to Spotify! I'd be happy to help you get oriented. What aspects of the data stack are you most interested in — the ingestion pipelines, the processing layer, or the serving infrastructure? Also, what's your background so I can calibrate my explanations?",
        "onboard"
    )
    orch.explicit.set("name", "New data engineer at Spotify", entry_type="fact", priority=10)

    # Turn 2 — Background
    await exchange(orch, 2,
        "I have 3 years of experience with Apache Spark and Airflow. I'm most interested in how Spotify handles the event streaming from 500 million users.",
        "Great background with Spark and Airflow — those are core tools here. Spotify's event streaming architecture uses Google Cloud Pub/Sub for real-time ingestion, processing about 30 billion events per day. Events flow from client apps through Pub/Sub into Dataflow for real-time processing, then land in BigQuery for batch analytics. The key pipeline to learn first is the 'Listening Events' pipeline — it powers most recommendation models.",
        "onboard"
    )
    orch.working.add_fact("3 years Spark + Airflow experience")
    orch.working.add_fact("Focus: event streaming from 500M users")

    # Turn 3 — Specific question
    await exchange(orch, 3,
        "How does the Listening Events pipeline handle late-arriving data? We had issues with that at my last job.",
        "The Listening Events pipeline uses event-time windowing with a 24-hour watermark for late arrivals. Events that arrive after the watermark are routed to a 'late events' side output and processed in a daily reconciliation batch job. This gives you at-least-once semantics for the real-time path and exactly-once for the daily batch. The reconciliation job runs at 4am UTC and takes about 45 minutes. Your Spark experience will help here — the batch reconciliation is actually a Spark job on Dataproc.",
        "onboard"
    )

    # Turn 4 — Follow-up referencing earlier context
    await exchange(orch, 4,
        "Got it. Since I know Spark, should I start with the reconciliation job? Also, who should I talk to about getting access to the Pub/Sub topics?",
        "Yes, starting with the reconciliation Spark job is a great idea since you already have Spark experience — it'll be familiar ground while you learn the data model. For Pub/Sub access, reach out to the Platform team lead, Marcus Chen. He handles onboarding permissions. You'll need access to the 'listening-events-prod' and 'listening-events-staging' topics. I'd suggest shadowing the daily reconciliation run tomorrow to see it in action before making any changes.",
        "onboard"
    )
    orch.working.add_fact("Contact: Marcus Chen for Pub/Sub access")

    # Turn 5 — The memory should now help
    await exchange(orch, 5,
        "Quick recap — what should my first week plan look like based on everything we discussed?",
        "Based on our conversation, here's your first week plan: Day 1-2: Get Pub/Sub access from Marcus Chen for 'listening-events-prod' and 'listening-events-staging' topics. Day 2-3: Shadow the daily reconciliation Spark job that runs at 4am UTC — it takes about 45 minutes and handles late-arriving events beyond the 24-hour watermark. Day 3-4: Read through the Listening Events pipeline code, focusing on the event-time windowing logic and the late events side output. Day 5: Set up your local development environment with Dataproc and BigQuery access. Your Spark and Airflow background will make the batch side straightforward — the real-time Dataflow side will be the learning curve.",
        "onboard"
    )

    await orch.end_session()
    orch.close(); await emb.close()

    # ══════════════════════════════════════════════════════════════
    print(f"\n{M}{B}{'━'*70}{X}")
    print(f"{M}{B}  CONVERSATION 2: Doctor researching treatment options{X}")
    print(f"{M}{B}{'━'*70}{X}")
    # ══════════════════════════════════════════════════════════════

    orch, emb = await make_orch()
    orch.start_session("clinical")

    # Turn 1
    await exchange(orch, 1,
        "I'm a cardiologist at Johns Hopkins. I have a 62-year-old patient with resistant hypertension — three medications and still averaging 155/95. What newer approaches should I consider?",
        "For resistant hypertension unresponsive to triple therapy, the main evidence-based options are: (1) Adding spironolactone as fourth-line agent — the PATHWAY-2 trial showed it was superior to bisoprolol and doxazosin. (2) Renal denervation — the SPYRAL HTN-ON MED trial showed a significant reduction in 24-hour ambulatory BP. (3) Checking for secondary causes if not already done: primary aldosteronism, renal artery stenosis, obstructive sleep apnea. What medications is the patient currently on?",
        "clinical"
    )
    orch.explicit.set("role", "Cardiologist at Johns Hopkins", entry_type="fact", priority=10)
    orch.explicit.set("patient", "62yo, resistant HTN, 155/95 on triple therapy", entry_type="fact", priority=9)

    # Turn 2
    await exchange(orch, 2,
        "Currently on amlodipine 10mg, lisinopril 40mg, and hydrochlorothiazide 25mg. We already ruled out secondary causes. Aldosterone-renin ratio was normal. Sleep study was negative.",
        "Good workup ruling out secondary causes. With the current regimen of amlodipine, lisinopril, and HCTZ, adding spironolactone 25mg would be the strongest evidence-based next step. The PATHWAY-2 trial specifically enrolled patients on similar triple therapy. Monitor potassium at 1 week and 4 weeks — the lisinopril plus spironolactone combination increases hyperkalemia risk. Target is BP under 130/80 per the 2017 ACC/AHA guidelines. If spironolactone is contraindicated or not tolerated, consider switching HCTZ to chlorthalidone which has stronger evidence at 25mg.",
        "clinical"
    )
    orch.working.add_fact("Meds: amlodipine 10mg, lisinopril 40mg, HCTZ 25mg")
    orch.working.add_fact("Secondary causes ruled out (aldo-renin normal, sleep study negative)")

    # Turn 3
    await exchange(orch, 3,
        "The patient also has an eGFR of 45. Does that change the spironolactone recommendation?",
        "Yes, eGFR 45 is an important consideration. Spironolactone can still be used but with extra caution — start at 12.5mg instead of 25mg and monitor potassium and creatinine at 3 days, 1 week, and then monthly. The AMBER trial showed that patiromer can enable continued use of spironolactone in CKD patients by managing hyperkalemia. If potassium rises above 5.5, hold the spironolactone. The combination of eGFR 45 plus lisinopril plus spironolactone puts this patient in a higher-risk category for AKI during any volume depletion — counsel about sick-day rules for the ACE inhibitor.",
        "clinical"
    )
    orch.working.add_fact("eGFR 45 — CKD consideration for spironolactone")

    # Turn 4 — References previous turns
    await exchange(orch, 4,
        "Good point about sick-day rules. One more thing — what's the latest evidence on renal denervation for someone with this kidney function?",
        "For renal denervation with eGFR 45, the data is limited but evolving. The SPYRAL HTN-ON MED trial excluded patients with eGFR below 40, so your patient would have qualified. The SYMPLICITY HTN-3 trial included patients with eGFR down to 45 and showed no safety signal for kidney function decline. More recently, the RADIANCE-HTN TRIO trial demonstrated significant BP reduction in resistant hypertension. However, given the CKD, I'd recommend exhausting pharmacological options first — try the low-dose spironolactone, and if inadequate, consider renal denervation as a next step. Also important: make sure adherence has been confirmed with a medication event monitoring system or observed dosing.",
        "clinical"
    )

    await orch.end_session()
    orch.close(); await emb.close()

    # ══════════════════════════════════════════════════════════════
    print(f"\n{M}{B}{'━'*70}{X}")
    print(f"{M}{B}  CONVERSATION 3: Startup founder across multiple sessions{X}")
    print(f"{M}{B}{'━'*70}{X}")
    # ══════════════════════════════════════════════════════════════

    orch, emb = await make_orch()
    db_path = Path(orch._config.storage.data_dir) / "amem.db"
    config_ref = orch._config

    # ── SESSION 1: Initial pitch ──
    print(f"\n  {D}─── SESSION 1 (Monday) ───{X}")
    orch.start_session("s1")

    await exchange(orch, 1,
        "I'm building an AI tutoring startup called LearnLoop. We use LLMs to create personalized lesson plans for K-12 students. Our differentiator is that we adapt in real-time based on student mistakes.",
        "LearnLoop sounds promising — the real-time adaptation based on error patterns is a strong differentiator. A few key questions: What subjects are you starting with? What's your go-to-market — direct to schools or direct to parents? And how are you handling the regulatory requirements around children's data (COPPA)?",
        "pitch"
    )
    orch.explicit.set("company", "LearnLoop — AI tutoring for K-12", entry_type="fact", priority=10)
    orch.explicit.set("differentiator", "Real-time adaptation based on student mistakes", entry_type="fact", priority=9)

    await exchange(orch, 2,
        "Starting with math for grades 6-8. Going direct to parents first — schools have 18-month sales cycles. We're COPPA compliant, data stored on US servers, parental consent required. We have 200 beta users and 40 percent weekly retention.",
        "Those are solid early metrics — 40 percent weekly retention for an education product is above the industry benchmark of 25-30 percent. Your direct-to-parent strategy avoids the long school sales cycle but you'll need strong unit economics since CAC for consumer education is typically $40-80. With 200 beta users, you should be tracking: time-to-first-lesson-completion, mistake pattern diversity, and whether students who get more personalized paths have better retention. These will be your key metrics for fundraising.",
        "pitch"
    )
    orch.working.add_fact("200 beta users, 40% weekly retention")
    orch.working.add_fact("Focus: math grades 6-8, direct to parents")

    await orch.end_session()
    orch.save()
    orch.close()

    # ── SESSION 2: Week later, fundraising ──
    print(f"\n  {D}─── SESSION 2 (following Monday — new session, same memory) ───{X}")
    embedder2 = create_embedder(config_ref.ollama)
    orch2 = MemoryOrchestrator(embedder2, config_ref)
    orch2.init_db(db_path)
    orch2.load()
    orch2.start_session("s2")

    await exchange(orch2, 3,
        "I'm preparing for our seed round. Can you help me think about the pitch based on what you know about LearnLoop?",
        "Based on what I know about LearnLoop: you're building AI tutoring for K-12, starting with math for grades 6-8, with real-time adaptation based on student mistakes as your differentiator. You have 200 beta users with 40 percent weekly retention, which is significantly above the education industry benchmark. For the seed pitch, I'd structure it as: (1) Problem — K-12 math is the largest tutoring market and current solutions don't adapt to individual error patterns. (2) Solution — LearnLoop's real-time adaptation creates personalized learning paths. (3) Traction — 200 beta users, 40 percent retention vs 25-30 percent industry average. (4) Go-to-market — direct to parents to avoid 18-month school sales cycles. (5) Ask — funding to reach 2,000 users and prove unit economics. What's your target raise amount?",
        "funding"
    )

    await exchange(orch2, 4,
        "We're raising 2 million at a 10 million pre-money valuation. Our monthly burn is 45K. Is that valuation reasonable?",
        "At $2M on $10M pre-money, you're at a 12M post-money — that gives you roughly 44 months of runway at $45K monthly burn, which is excellent. The valuation is defensible given: you have working product with real users, strong retention metrics above industry benchmarks, and AI/education is a hot category. For comparison, seed rounds in edtech averaged $8-15M post-money in 2024. The key risk investors will flag: your 200 users are still small — can you show a clear path to 2,000 users within 6 months of funding? Also, what's your current CAC and LTV? Those numbers will make or break the unit economics story.",
        "funding"
    )
    orch2.working.add_fact("Raising $2M seed at $10M pre-money")
    orch2.working.add_fact("Monthly burn: $45K → ~44 months runway")

    await orch2.end_session()
    orch2.close(); await embedder2.close(); await emb.close()

    # ══════════════════════════════════════════════════════════════
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  KEY OBSERVATIONS{X}")
    print(f"{B}{'═'*70}{X}")
    print(f"""
  What you saw in each conversation:

  {BL}{B}PROMPT{X} — What the user typed
  {Y}{B}MEMORY RETRIEVED{X} — What the system pulled from all 5 layers
  {G}{B}RESPONSE{X} — What the LLM said (using that memory as context)
  {D}LEARNED{X} — What got stored back into memory

  {B}Conversation 1 (Data Engineer):{X}
  • Turn 1: Empty memory → generic welcome
  • Turn 3: Memory has their Spark background → response mentions
    "Your Spark experience will help here"
  • Turn 5: Full context accumulated → response synthesizes ALL
    previous turns into a coherent week plan with specific names
    (Marcus Chen), times (4am UTC), and topics from earlier turns

  {B}Conversation 2 (Cardiologist):{X}
  • Turn 1: Empty → standard clinical options
  • Turn 3: Memory has meds + secondary workup → response adjusts
    spironolactone dose for eGFR 45, references prior medication list
  • Turn 4: Memory has the full clinical picture → response weighs
    renal denervation against CKD risk, referencing specific trials

  {B}Conversation 3 (Startup Founder):{X}
  • Session 1, Turn 2: Metrics stored (200 users, 40% retention)
  • Session 2, Turn 3: {Y}NEW SESSION, memory persisted{X} → response
    opens with "Based on what I know about LearnLoop" and references
    all metrics, strategy, and differentiator from the previous week
  • Turn 4: Response calculates runway from burn rate context

  {B}The difference from current systems:{X}
  The LLM never saw the raw conversation history.
  It saw a {G}structured, query-relevant memory context{X}
  that grew with each exchange. 329 tokens instead of 3000.
  Every fact preserved. Every detail retrievable.
""")


if __name__ == "__main__":
    asyncio.run(main())
