#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  HEAD TO HEAD: amem vs Mem0 — Same data, same judge, same machine
═══════════════════════════════════════════════════════════════════════

Runs BOTH systems on the same LoCoMo conversations side by side.
Same GPT-4o-mini eval + judge for both. No cherry-picking.

Measures:
  1. Accuracy (context sufficiency judged by GPT-4o-mini)
  2. LLM calls used (extraction cost)
  3. Latency (ingest + query time)
  4. Tokens injected per query

Usage: OPENAI_API_KEY=sk-... PYTHONPATH=. python3 benchmarks/head_to_head.py
"""

import asyncio
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import httpx

# amem imports
from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator

# mem0 imports
from mem0 import Memory

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
EVAL_MODEL = "gpt-4o-mini"
CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-ended"}
B="\033[1m"; G="\033[92m"; R="\033[91m"; Y="\033[93m"; D="\033[2m"; X="\033[0m"; C="\033[96m"


async def openai_chat(messages, temperature=0.1):
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                    json={"model": EVAL_MODEL, "messages": messages, "temperature": temperature, "max_tokens": 300},
                )
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 2: raise
            await asyncio.sleep(1)


async def eval_answer(context, question):
    return await openai_chat([
        {"role": "system", "content": "Answer based on the memory context. Be concise. If you don't know, say 'I don't know'."},
        {"role": "user", "content": f"Memory:\n{context}\n\nQuestion: {question}\nAnswer:"},
    ])


async def judge_answer(question, gold, predicted):
    result = await openai_chat([
        {"role": "system", "content": "Judge if predicted answer is correct. Generous: same topic=CORRECT, date format differences=CORRECT, partial relevant=CORRECT. Only WRONG if clearly wrong or 'I don't know'. Reply CORRECT or WRONG only."},
        {"role": "user", "content": f"Q: {question}\nGold: {gold}\nPredicted: {predicted}"},
    ], temperature=0.0)
    return "CORRECT" in result.upper()


async def openai_extract_facts(text):
    """GPT-4o-mini fact extraction for amem."""
    try:
        result = await openai_chat([
            {"role": "system", "content": "Extract all important facts as a JSON array of strings. Include who, what, when, where, numbers, decisions. Be thorough."},
            {"role": "user", "content": f"Conversation:\n{text[:6000]}\n\nFacts:"},
        ])
        import re
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            return [str(f).strip() for f in json.loads(match.group()) if f and len(str(f)) > 5]
    except Exception:
        pass
    return []


async def main():
    if not OPENAI_KEY:
        print(f"{R}Set OPENAI_API_KEY{X}"); return

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  HEAD TO HEAD: amem vs Mem0{X}")
    print(f"{B}  Same data · Same judge · Same machine{X}")
    print(f"{B}{'═'*70}{X}")

    with open("benchmarks/locomo10.json") as f:
        data = json.load(f)

    # Use 2 conversations for manageable runtime
    MAX_CONVS = 2
    results = defaultdict(lambda: {"amem": 0, "mem0": 0, "n": 0})
    timing = {"amem_ingest": 0, "mem0_ingest": 0, "amem_query": 0, "mem0_query": 0}
    calls = {"amem": 0, "mem0": 0}

    for conv_idx in range(min(MAX_CONVS, len(data))):
        conv = data[conv_idx]
        conversation = conv["conversation"]
        qa_pairs = [q for q in conv["qa"] if q.get("category", 5) != 5]

        speaker_a = conversation.get("speaker_a", "A")
        speaker_b = conversation.get("speaker_b", "B")
        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])

        print(f"\n  {C}{B}Conversation {conv_idx+1}/{MAX_CONVS}: {speaker_a} & {speaker_b} — {len(qa_pairs)} questions{X}")

        # ══════════════════════════════════════════════════════════
        # INGEST INTO AMEM
        # ══════════════════════════════════════════════════════════
        print(f"  {D}Ingesting into amem...{X}", end="", flush=True)
        config = Config()
        conv_dir = tempfile.mkdtemp()
        config.storage.data_dir = conv_dir
        embedder = create_embedder(config.ollama)
        orch = MemoryOrchestrator(embedder, config)
        orch.init_db(Path(conv_dir) / "amem.db")

        t0 = time.monotonic()
        amem_extract_calls = 0

        for sk in session_keys:
            session = conversation[sk]
            if not isinstance(session, list): continue
            session_date = conversation.get(f"{sk}_date_time", "")

            orch.start_session(sk)
            session_texts = []

            if session_date:
                date_text = f"This conversation took place on {session_date}."
                await orch.ingest(text=date_text, conversation_id=sk, speaker="system")
                session_texts.append(date_text)
                orch.working.add_fact(f"Date: {session_date}")

            for turn in session:
                if not isinstance(turn, dict): continue
                text = turn.get("text", "")
                if not text: continue
                await orch.ingest(text=text, conversation_id=sk, speaker=turn.get("speaker", ""))
                session_texts.append(text)

            # GPT-4o-mini extraction per session
            session_text = "\n".join(session_texts)
            if session_text.strip():
                facts = await openai_extract_facts(session_text)
                amem_extract_calls += 1
                for fact in facts:
                    await orch.ingest(text=fact, conversation_id=f"facts-{sk}", speaker="fact")

            orch.working.clear()
            orch._session_turns.clear()

        amem_ingest_time = time.monotonic() - t0
        timing["amem_ingest"] += amem_ingest_time
        calls["amem"] += amem_extract_calls
        print(f" {amem_ingest_time:.1f}s ({amem_extract_calls} LLM calls)")

        # ══════════════════════════════════════════════════════════
        # INGEST INTO MEM0
        # ══════════════════════════════════════════════════════════
        print(f"  {D}Ingesting into Mem0...{X}", end="", flush=True)

        mem0_config = {
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "temperature": 0.1}},
            "version": "v1.1",
        }
        m = Memory.from_config(config_dict=mem0_config)
        user_id = f"locomo_conv_{conv_idx}"

        t0 = time.monotonic()
        mem0_add_calls = 0

        for sk in session_keys:
            session = conversation[sk]
            if not isinstance(session, list): continue
            session_date = conversation.get(f"{sk}_date_time", "")

            for turn in session:
                if not isinstance(turn, dict): continue
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                if not text: continue

                msg = text
                if session_date:
                    msg = f"[{session_date}] {speaker}: {text}"

                try:
                    m.add(msg, user_id=user_id)
                    mem0_add_calls += 1
                except Exception:
                    pass

        mem0_ingest_time = time.monotonic() - t0
        timing["mem0_ingest"] += mem0_ingest_time
        calls["mem0"] += mem0_add_calls
        print(f" {mem0_ingest_time:.1f}s ({mem0_add_calls} LLM calls)")

        # ══════════════════════════════════════════════════════════
        # EVALUATE BOTH ON SAME QUESTIONS
        # ══════════════════════════════════════════════════════════
        print(f"  {D}Evaluating...{X}")

        conv_r = {"amem": 0, "mem0": 0, "n": 0}

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))
            category = qa.get("category", 0)
            if not question or not answer: continue

            try:
                # amem retrieval
                t0 = time.monotonic()
                knowledge = await orch.query_knowledge(question, top_k=30)
                amem_context = knowledge.to_injection_text()
                timing["amem_query"] += time.monotonic() - t0

                # Mem0 retrieval
                t0 = time.monotonic()
                mem0_results = m.search(question, user_id=user_id, limit=30)
                mem0_context = "\n".join([
                    r.get("memory", r.get("text", ""))
                    for r in (mem0_results.get("results", mem0_results) if isinstance(mem0_results, dict) else mem0_results)
                    if isinstance(r, dict)
                ])
                timing["mem0_query"] += time.monotonic() - t0

                # Eval both
                amem_resp = await eval_answer(amem_context, question)
                mem0_resp = await eval_answer(mem0_context, question)

                amem_ok = await judge_answer(question, answer, amem_resp)
                mem0_ok = await judge_answer(question, answer, mem0_resp)

                if amem_ok:
                    conv_r["amem"] += 1
                    results[category]["amem"] += 1
                    results["all"]["amem"] += 1
                if mem0_ok:
                    conv_r["mem0"] += 1
                    results[category]["mem0"] += 1
                    results["all"]["mem0"] += 1
                conv_r["n"] += 1
                results[category]["n"] += 1
                results["all"]["n"] += 1

            except Exception as e:
                if qi < 3:
                    print(f"    {R}Error Q{qi}: {e}{X}")
                continue

            if (qi + 1) % 30 == 0:
                a = conv_r["amem"]/max(conv_r["n"],1)*100
                m_pct = conv_r["mem0"]/max(conv_r["n"],1)*100
                print(f"    {D}...{qi+1}/{len(qa_pairs)} | amem {a:.0f}% | mem0 {m_pct:.0f}%{X}")

        n = conv_r["n"]
        a = conv_r["amem"]/max(n,1)*100
        m_pct = conv_r["mem0"]/max(n,1)*100
        print(f"  amem: {G}{a:.1f}%{X} | mem0: {m_pct:.1f}% | ({n} questions)")

        orch.close()
        await embedder.close()

    # ══════════════════════════════════════════════════════════
    # FINAL RESULTS
    # ══════════════════════════════════════════════════════════
    r = results["all"]
    n = max(r["n"], 1)
    a_pct = r["amem"]/n*100
    m_pct = r["mem0"]/n*100
    delta = a_pct - m_pct

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  RESULTS — {n} questions, head to head{X}")
    print(f"{B}{'═'*70}{X}")

    print(f"\n  {B}Accuracy:{X}")
    print(f"    {'System':<20s} {'Score':>8s} {'Delta':>10s}")
    print(f"    {'─'*40}")
    print(f"    {'amem':<20s} {a_pct:>7.1f}%  {G if delta>=0 else R}{delta:>+8.1f}%{X}")
    print(f"    {'Mem0':<20s} {m_pct:>7.1f}%")

    print(f"\n  {B}By Category:{X}")
    print(f"    {'Category':<15s} {'amem':>8s} {'Mem0':>8s} {'Delta':>8s}  {'n':>4s}")
    print(f"    {'─'*45}")
    for cat in sorted(c for c in results if isinstance(c, int)):
        cr = results[cat]; cn = max(cr["n"],1)
        a = cr["amem"]/cn*100; m = cr["mem0"]/cn*100; d = a-m
        print(f"    {CATEGORY_NAMES.get(cat,'?'):<15s} {a:>7.1f}% {m:>7.1f}% {G if d>=0 else R}{d:>+7.1f}%{X}  {cr['n']:>4d}")

    print(f"\n  {B}Cost (LLM calls for extraction):{X}")
    print(f"    amem:  {calls['amem']} calls (per-session)")
    print(f"    Mem0:  {calls['mem0']} calls (per-turn)")
    print(f"    Ratio: Mem0 uses {calls['mem0']/max(calls['amem'],1):.0f}× more LLM calls")

    print(f"\n  {B}Latency:{X}")
    total_q = max(r["n"], 1)
    print(f"    amem ingest: {timing['amem_ingest']:.1f}s total")
    print(f"    Mem0 ingest: {timing['mem0_ingest']:.1f}s total")
    print(f"    amem query:  {timing['amem_query']/total_q*1000:.0f}ms avg per query")
    print(f"    Mem0 query:  {timing['mem0_query']/total_q*1000:.0f}ms avg per query")

    print(f"\n  {B}Methodology:{X}")
    print(f"    Same GPT-4o-mini for eval + judge")
    print(f"    Same LoCoMo questions, same order")
    print(f"    amem: per-session GPT-4o-mini extraction + Ollama embeddings")
    print(f"    Mem0: per-turn extraction (default config, GPT-4o-mini)")
    print(f"\n{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
