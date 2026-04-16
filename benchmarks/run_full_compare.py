#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  FULL COMPARISON: amem vs Mem0 vs Blob — 10 conversations
═══════════════════════════════════════════════════════════════════════

Runs all 10 LoCoMo conversations. Saves results incrementally
so it can survive crashes.

amem: Selective extraction (embedding-gated)
Mem0: Per-turn extraction (their default)
Blob: Full text injected (OpenAI Memory baseline)

All use GPT-4o-mini for eval + judge.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import gc
from collections import defaultdict
from pathlib import Path

import httpx

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
EVAL_MODEL = "gpt-4o-mini"
TOP_K = 30
BLOB_LIMIT = 6000
CATEGORY_NAMES = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-ended"}
RESULTS_FILE = "benchmarks/results_incremental.json"

B="\033[1m"; G="\033[92m"; R="\033[91m"; D="\033[2m"; X="\033[0m"; C="\033[96m"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator


async def openai_chat(messages, temperature=0.1):
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                    json={"model": EVAL_MODEL, "messages": messages, "temperature": temperature, "max_tokens": 300},
                )
                if resp.status_code == 429:
                    wait = min(30, 2 ** attempt)
                    print(f"      {D}rate limited, waiting {wait}s{X}", flush=True)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == 4: raise
            await asyncio.sleep(2)


async def eval_answer(context, question):
    return await openai_chat([
        {"role": "system", "content": "Answer based on the memory context. Be concise. If you don't know, say 'I don't know'."},
        {"role": "user", "content": f"Memory:\n{context[:4000]}\n\nQuestion: {question}\nAnswer:"},
    ])


async def judge(question, gold, predicted):
    r = await openai_chat([
        {"role": "system", "content": "Judge correctness. Generous: same topic=CORRECT, date format differences=CORRECT. Only WRONG if clearly wrong or 'I don't know'. Reply CORRECT or WRONG only."},
        {"role": "user", "content": f"Q: {question}\nGold: {gold}\nPredicted: {predicted}"},
    ], temperature=0.0)
    return "CORRECT" in r.upper()


def run_mem0_ingest(conversation, conv_idx):
    """Run Mem0 synchronously (avoids SQLite threading issues)."""
    try:
        from mem0 import Memory
        m = Memory.from_config(config_dict={
            "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "temperature": 0.1}},
            "version": "v1.1",
        })
        user_id = f"locomo_{conv_idx}"
        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])
        calls = 0
        for sk in session_keys:
            session = conversation[sk]
            if not isinstance(session, list): continue
            date = conversation.get(f"{sk}_date_time", "")
            for turn in session:
                if not isinstance(turn, dict): continue
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                if not text: continue
                msg = f"[{date}] {speaker}: {text}" if date else f"{speaker}: {text}"
                try:
                    m.add(msg, user_id=user_id)
                    calls += 1
                except Exception:
                    pass
        return m, user_id, calls
    except Exception as e:
        print(f"      {R}Mem0 ingest error: {e}{X}")
        return None, None, 0


def mem0_search(m, user_id, question):
    """Search Mem0 synchronously."""
    try:
        results = m.search(question, user_id=user_id, limit=30)
        if isinstance(results, dict):
            items = results.get("results", results.get("memories", []))
        elif isinstance(results, list):
            items = results
        else:
            items = []
        return "\n".join([
            r.get("memory", r.get("text", str(r)))
            for r in items if isinstance(r, dict)
        ])
    except Exception:
        return ""


def save_results(all_results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


async def main():
    if not OPENAI_KEY:
        print(f"{R}Set OPENAI_API_KEY{X}"); return

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  FULL COMPARISON: amem vs Mem0 vs Blob{X}")
    print(f"{B}  10 conversations · Incremental save · GPT-4o-mini{X}")
    print(f"{B}{'═'*70}{X}")

    with open("benchmarks/locomo10.json") as f:
        data = json.load(f)

    # Load previous results (resume from crash)
    saved = load_results()
    completed_convs = set(saved.get("completed", []))
    all_results = saved.get("results", {})
    total_calls = saved.get("total_calls", 0)

    for conv_idx in range(len(data)):
        if str(conv_idx) in completed_convs:
            print(f"  {D}Conv {conv_idx+1}/10 — already done, skipping{X}")
            continue

        conv = data[conv_idx]
        conversation = conv["conversation"]
        qa_pairs = [q for q in conv["qa"] if q.get("category", 5) != 5]

        print(f"\n  {C}{B}Conv {conv_idx+1}/10 — {len(qa_pairs)} questions{X}")

        # ── AMEM INGEST ──
        print(f"  {D}amem ingesting...{X}", end="", flush=True)
        config = Config()
        conv_dir = tempfile.mkdtemp()
        config.storage.data_dir = conv_dir
        embedder = create_embedder(config.ollama)
        orch = MemoryOrchestrator(embedder, config)
        orch.init_db(Path(conv_dir) / "amem.db")

        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])
        blob_parts = []
        t0 = time.monotonic()

        for sk in session_keys:
            session = conversation[sk]
            if not isinstance(session, list): continue
            date = conversation.get(f"{sk}_date_time", "")
            orch.start_session(sk)
            if date:
                await orch.ingest(text=f"This conversation took place on {date}.", conversation_id=sk, speaker="system")
                blob_parts.append(f"[Date: {date}]")
                orch.working.add_fact(f"Date: {date}")
            for turn in session:
                if not isinstance(turn, dict): continue
                text = turn.get("text", "")
                if not text: continue
                await orch.ingest(text=text, conversation_id=sk, speaker=turn.get("speaker", ""))
                blob_parts.append(text)
            await orch.end_session()

        amem_time = time.monotonic() - t0
        amem_stats = orch._selective_extractor.stats
        print(f" {amem_time:.0f}s ({amem_stats['llm_calls']} LLM calls, {amem_stats['extraction_rate']} extracted)")

        # ── MEM0 INGEST ──
        print(f"  {D}Mem0 ingesting...{X}", end="", flush=True)
        t0 = time.monotonic()
        m, user_id, mem0_calls = run_mem0_ingest(conversation, conv_idx)
        mem0_time = time.monotonic() - t0
        mem0_available = m is not None
        print(f" {mem0_time:.0f}s ({mem0_calls} LLM calls)" if mem0_available else f" FAILED")

        # ── BLOB ──
        blob_full = "\n".join(blob_parts)
        blob_trunc = " ".join(blob_full.split()[:BLOB_LIMIT * 3 // 4])

        # ── EVALUATE ──
        print(f"  {D}Evaluating...{X}")
        conv_r = defaultdict(lambda: {"amem": 0, "mem0": 0, "blob": 0, "n": 0})

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))
            cat = qa.get("category", 0)
            if not question or not answer: continue

            try:
                # amem
                knowledge = await orch.query_knowledge(question, top_k=TOP_K)
                amem_ctx = knowledge.to_injection_text()
                amem_resp = await eval_answer(amem_ctx, question)

                # Mem0
                if mem0_available:
                    mem0_ctx = mem0_search(m, user_id, question)
                    mem0_resp = await eval_answer(mem0_ctx, question)
                else:
                    mem0_resp = "I don't know"

                # Blob
                blob_resp = await eval_answer(blob_trunc, question)

                total_calls += 3

                # Judge
                amem_ok = await judge(question, answer, amem_resp)
                mem0_ok = await judge(question, answer, mem0_resp) if mem0_available else False
                blob_ok = await judge(question, answer, blob_resp)
                total_calls += 3

                for key, ok in [("amem", amem_ok), ("mem0", mem0_ok), ("blob", blob_ok)]:
                    if ok:
                        conv_r[cat][key] += 1
                        conv_r["all"][key] += 1
                conv_r[cat]["n"] += 1
                conv_r["all"]["n"] += 1

            except Exception as e:
                if qi < 3: print(f"    {R}Error Q{qi}: {e}{X}")
                continue

            if (qi + 1) % 30 == 0:
                n = conv_r["all"]["n"]
                a = conv_r["all"]["amem"]/max(n,1)*100
                m_p = conv_r["all"]["mem0"]/max(n,1)*100
                b = conv_r["all"]["blob"]/max(n,1)*100
                print(f"    {D}...{qi+1}/{len(qa_pairs)} | amem={a:.0f}% mem0={m_p:.0f}% blob={b:.0f}%{X}")

        # Save conv results
        n = conv_r["all"]["n"]
        a = conv_r["all"]["amem"]/max(n,1)*100
        mp = conv_r["all"]["mem0"]/max(n,1)*100
        b = conv_r["all"]["blob"]/max(n,1)*100
        print(f"  {G}amem={a:.1f}%{X} mem0={mp:.1f}% blob={b:.1f}% ({n}q)")

        # Accumulate
        for cat_key, cat_data in conv_r.items():
            sk = str(cat_key)
            if sk not in all_results:
                all_results[sk] = {"amem": 0, "mem0": 0, "blob": 0, "n": 0}
            for k in ["amem", "mem0", "blob", "n"]:
                all_results[sk][k] += cat_data[k]

        completed_convs.add(str(conv_idx))
        save_results({"completed": list(completed_convs), "results": all_results, "total_calls": total_calls})

        # Cleanup memory
        orch.close()
        await embedder.close()
        del orch, embedder, m
        gc.collect()

    # ── FINAL RESULTS ──
    r = all_results.get("all", {"amem": 0, "mem0": 0, "blob": 0, "n": 1})
    n = max(r["n"], 1)

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  FINAL RESULTS — {n} questions across {len(completed_convs)} conversations{X}")
    print(f"{B}{'═'*70}{X}")

    print(f"\n  {B}{'System':<20s} {'Score':>8s} {'vs Blob':>10s} {'vs Mem0':>10s}{X}")
    print(f"  {'─'*50}")
    a = r["amem"]/n*100
    mp = r["mem0"]/n*100
    b = r["blob"]/n*100
    print(f"  {'amem':<20s} {G}{B}{a:>7.1f}%{X} {G if a>b else R}{a-b:>+9.1f}%{X} {G if a>mp else R}{a-mp:>+9.1f}%{X}")
    print(f"  {'Mem0':<20s} {mp:>7.1f}% {G if mp>b else R}{mp-b:>+9.1f}%{X}")
    print(f"  {'Blob':<20s} {b:>7.1f}%")

    print(f"\n  {B}By Category:{X}")
    print(f"  {'Category':<15s} {'amem':>8s} {'Mem0':>8s} {'Blob':>8s}  {'n':>4s}")
    print(f"  {'─'*45}")
    for cat in sorted(int(c) for c in all_results if c.isdigit()):
        cr = all_results[str(cat)]
        cn = max(cr["n"], 1)
        print(f"  {CATEGORY_NAMES.get(cat,'?'):<15s} {cr['amem']/cn*100:>7.1f}% {cr['mem0']/cn*100:>7.1f}% {cr['blob']/cn*100:>7.1f}%  {cr['n']:>4d}")

    # Mem0 published comparison
    print(f"\n  {B}vs Mem0 published (66.9%):{X}")
    print(f"  amem measured: {a:.1f}%  |  Δ: {a-66.9:+.1f}%")

    print(f"\n  API calls: {total_calls}")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"\n{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
