#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  LoCoMo Benchmark v2 — Cross-Session Knowledge Architecture
═══════════════════════════════════════════════════════════════════════

NEW APPROACH: Memory is long-term knowledge, not session context.

Old (wrong): inject raw chunks as context replacement
New (correct): inject cross-session knowledge the LLM doesn't have

Protocol (matches Mem0):
  - eval-LLM: GPT-4o-mini answers using memory context
  - judge-LLM: GPT-4o-mini grades CORRECT/WRONG
  - Category 5 (adversarial) excluded
  - Generous grading

Key change: We ingest session-by-session (as real conversations happen),
building up the knowledge graph and session summaries over time.
When answering questions, the LLM gets:
  1. Structured facts from the knowledge graph
  2. Session summaries from past sessions
  3. Relevant episodic details (cross-session only)
  4. User identity and preferences

NOT raw chunks from the same conversation being queried.
"""

import asyncio
import json
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator

B="\033[1m"; G="\033[92m"; R="\033[91m"; Y="\033[93m"; D="\033[2m"; X="\033[0m"; C="\033[96m"
CATEGORY_NAMES = {1:"single-hop", 2:"multi-hop", 3:"temporal", 4:"open-ended"}
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
EVAL_MODEL = "gpt-4o-mini"


async def openai_chat(messages, temperature=0.1):
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={"model": EVAL_MODEL, "messages": messages, "temperature": temperature, "max_tokens": 300},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def eval_answer(context, question):
    return await openai_chat([
        {"role": "system", "content": "You are a helpful assistant. Answer based on the memory context provided. Be concise. If you don't know, say 'I don't know'."},
        {"role": "user", "content": f"Memory context:\n{context}\n\nQuestion: {question}\nAnswer:"},
    ], temperature=0.1)


async def judge_answer(question, gold, predicted):
    result = await openai_chat([
        {"role": "system", "content": "Judge if the predicted answer is correct. Be generous: same topic = CORRECT, date format differences = CORRECT, partial but relevant = CORRECT. Only WRONG if clearly incorrect or 'I don't know'. Reply CORRECT or WRONG only."},
        {"role": "user", "content": f"Question: {question}\nGold: {gold}\nPredicted: {predicted}"},
    ], temperature=0.0)
    return "CORRECT" in result.upper()


async def main():
    if not OPENAI_KEY:
        print(f"{R}Set OPENAI_API_KEY{X}"); return

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  LoCoMo v2 — Cross-Session Knowledge Architecture{X}")
    print(f"{B}{'═'*70}{X}")

    with open("benchmarks/locomo10.json") as f:
        data = json.load(f)

    config = Config()
    MAX_CONVS = 2
    TOP_K = 30
    BLOB_LIMIT = 6000

    results = defaultdict(lambda: {"amem": 0, "blob": 0, "none": 0, "n": 0})
    api_calls = 0

    for conv_idx in range(min(MAX_CONVS, len(data))):
        conv = data[conv_idx]
        conversation = conv["conversation"]
        qa_pairs = [q for q in conv["qa"] if q.get("category", 5) != 5]

        print(f"\n  {C}{B}Conversation {conv_idx+1}/{MAX_CONVS} — {len(qa_pairs)} questions{X}")

        conv_dir = tempfile.mkdtemp()
        config.storage.data_dir = conv_dir
        embedder = create_embedder(config.ollama)
        orch = MemoryOrchestrator(embedder, config)
        orch.init_db(Path(conv_dir) / "amem.db")

        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])

        # ── Ingest SESSION BY SESSION (as real conversations happen) ──
        blob_parts = []
        t0 = time.monotonic()

        for sk in session_keys:
            session = conversation[sk]
            if not isinstance(session, list): continue
            session_date = conversation.get(f"{sk}_date_time", "")

            # Start a session (like a real user would)
            orch.start_session(sk)

            if session_date:
                date_text = f"This conversation took place on {session_date}."
                await orch.ingest(text=date_text, conversation_id=sk, speaker="system")
                blob_parts.append(date_text)
                orch.working.add_fact(f"Date: {session_date}")

            for turn in session:
                if not isinstance(turn, dict): continue
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                if not text: continue
                await orch.ingest(text=text, conversation_id=sk, speaker=speaker)
                blob_parts.append(f"{speaker}: {text}")

            # End session → creates summary + extracts knowledge
            await orch.end_session()

        ingest_time = time.monotonic() - t0
        blob_full = "\n".join(blob_parts)
        blob_words = blob_full.split()
        blob_truncated = " ".join(blob_words[:BLOB_LIMIT * 3 // 4])

        sem_stats = orch.semantic.stats()
        print(f"  {D}Ingested in {ingest_time:.1f}s | {len(session_keys)} sessions | "
              f"Entities: {sem_stats.get('entities',0)} | Relations: {sem_stats.get('relations',0)} | "
              f"Summaries: {len(orch._session_summaries)}{X}")

        # ── Evaluate ──
        t0 = time.monotonic()
        conv_r = {"amem": 0, "blob": 0, "none": 0, "n": 0}

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))
            category = qa.get("category", 0)
            if not question or not answer: continue

            try:
                # amem: cross-session knowledge (NOT raw chunks)
                knowledge = await orch.query_knowledge(question, top_k=TOP_K)
                amem_context = knowledge.to_injection_text()

                amem_response = await eval_answer(amem_context, question)
                blob_response = await eval_answer(blob_truncated, question)
                none_response = await eval_answer("No context available.", question)
                api_calls += 3

                amem_ok = await judge_answer(question, answer, amem_response)
                blob_ok = await judge_answer(question, answer, blob_response)
                none_ok = await judge_answer(question, answer, none_response)
                api_calls += 3

                for key, ok in [("amem", amem_ok), ("blob", blob_ok), ("none", none_ok)]:
                    if ok:
                        conv_r[key] += 1
                        results[category][key] += 1
                        results["all"][key] += 1
                conv_r["n"] += 1
                results[category]["n"] += 1
                results["all"]["n"] += 1

            except Exception as e:
                print(f"  {R}Error Q{qi}: {e}{X}")
                continue

            if (qi + 1) % 20 == 0:
                a = conv_r["amem"]/max(conv_r["n"],1)*100
                b = conv_r["blob"]/max(conv_r["n"],1)*100
                print(f"    {D}...{qi+1}/{len(qa_pairs)} | amem {a:.0f}% | blob {b:.0f}% | calls: {api_calls}{X}")

        eval_time = time.monotonic() - t0
        n = conv_r["n"]
        print(f"  {D}Evaluated {n} in {eval_time:.1f}s{X}")
        a = conv_r["amem"]/max(n,1)*100
        b = conv_r["blob"]/max(n,1)*100
        print(f"  amem: {G}{a:.1f}%{X} | blob: {b:.1f}% | none: {conv_r['none']/max(n,1)*100:.1f}%")

        orch.close()
        await embedder.close()

    # ── Results ──
    r = results["all"]
    n = r["n"]
    a_pct = r["amem"]/max(n,1)*100
    b_pct = r["blob"]/max(n,1)*100
    no_pct = r["none"]/max(n,1)*100

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  RESULTS — {n} questions (categories 1-4){X}")
    print(f"{B}{'═'*70}{X}")

    print(f"\n  {B}Overall Accuracy:{X}")
    print(f"    {'System':<25s} {'Score':>8s} {'vs Blob':>10s}")
    print(f"    {'─'*45}")
    d = a_pct - b_pct
    print(f"    {'amem (knowledge)':<25s} {G}{B}{a_pct:>7.1f}%{X} {G if d>=0 else R}{d:>+9.1f}%{X}")
    print(f"    {'Blob (OpenAI-style)':<25s} {b_pct:>7.1f}%")
    print(f"    {'No memory':<25s} {no_pct:>7.1f}%")

    if b_pct > 0:
        rel = (a_pct - b_pct) / b_pct * 100
        print(f"\n  {B}Relative uplift over blob: {G if rel>=0 else R}{B}{rel:+.1f}%{X}")
        print(f"  {D}(Mem0 claims +26% relative uplift){X}")

    print(f"\n  {B}By Category:{X}")
    print(f"    {'Category':<15s} {'amem':>8s} {'Blob':>8s} {'Δ':>8s}  {'n':>4s}")
    print(f"    {'─'*45}")
    for cat in sorted(c for c in results if isinstance(c, int)):
        cr = results[cat]; cn = max(cr["n"],1)
        a = cr["amem"]/cn*100; b = cr["blob"]/cn*100; d = a-b
        print(f"    {CATEGORY_NAMES.get(cat,'?'):<15s} {a:>7.1f}% {b:>7.1f}% {G if d>=0 else R}{d:>+7.1f}%{X}  {cr['n']:>4d}")

    print(f"\n  {B}Key change from v1:{X}")
    print(f"  Memory is now CROSS-SESSION KNOWLEDGE, not context replacement.")
    print(f"  - Semantic graph facts (not raw chunks)")
    print(f"  - Session summaries (not full transcripts)")
    print(f"  - User identity and preferences")
    print(f"  Calls: {api_calls}")
    print(f"\n{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
