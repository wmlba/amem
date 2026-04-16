#!/usr/bin/env python3
"""
Run NOW — actual benchmark with selective per-turn extraction.

- Ollama qwen3.5 for fact extraction (local, free)
- Ollama nomic-embed-text for embeddings (local, free)
- GPT-4o-mini for eval + judge ONLY (same as Mem0)
- Selective extraction: embedding-gated, batched
- 2 conversations (quick signal, ~30-45 min)
"""

import asyncio
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import httpx

os.environ.setdefault("PYTHONPATH", ".")
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
CATS = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "open-ended"}
B="\033[1m"; G="\033[92m"; R="\033[91m"; D="\033[2m"; X="\033[0m"; C="\033[96m"; Y="\033[93m"


async def gpt4o(messages, temp=0.1):
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                r = await c.post("https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                    json={"model": "gpt-4o-mini", "messages": messages, "temperature": temp, "max_tokens": 300})
                if r.status_code == 429:
                    await asyncio.sleep(2 ** attempt); continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt == 4: raise
            await asyncio.sleep(1)


async def main():
    assert OPENAI_KEY, "Set OPENAI_API_KEY"

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  RUNNING NOW — Selective Per-Turn Extraction{X}")
    print(f"{B}  Extraction: Ollama qwen3.5 (local, free){X}")
    print(f"{B}  Embeddings: Ollama nomic-embed-text (local, free){X}")
    print(f"{B}  Eval/Judge: GPT-4o-mini (same as Mem0){X}")
    print(f"{B}{'═'*70}{X}")

    with open("benchmarks/locomo10.json") as f:
        data = json.load(f)

    config = Config()
    MAX_CONVS = 2
    results = defaultdict(lambda: {"amem": 0, "blob": 0, "n": 0})
    api_calls = 0
    total_extraction_calls = 0
    total_turns = 0

    for ci in range(MAX_CONVS):
        conv = data[ci]
        conversation = conv["conversation"]
        qa_pairs = [q for q in conv["qa"] if q.get("category", 5) != 5]
        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])

        print(f"\n  {C}{B}Conv {ci+1}/{MAX_CONVS} — {len(qa_pairs)} questions, {len(session_keys)} sessions{X}")

        # Fresh orchestrator
        tmpdir = tempfile.mkdtemp()
        config.storage.data_dir = tmpdir
        embedder = create_embedder(config.ollama)
        orch = MemoryOrchestrator(embedder, config)
        orch.init_db(Path(tmpdir) / "amem.db")

        # Ingest session by session with selective per-turn extraction
        blob_parts = []
        t0 = time.monotonic()
        conv_turns = 0

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
                # This calls selective extractor internally:
                # embed turn → check novelty → batch extract if novel
                await orch.ingest(text=text, conversation_id=sk, speaker=turn.get("speaker", ""))
                blob_parts.append(text)
                conv_turns += 1

            await orch.end_session()

        ingest_time = time.monotonic() - t0
        se = orch._selective_extractor.stats
        total_turns += conv_turns
        total_extraction_calls += se["llm_calls"]

        print(f"  {D}Ingested {conv_turns} turns in {ingest_time:.0f}s{X}")
        print(f"  {D}Selective extractor: {se['turns_extracted']}/{se['turns_seen']} turns extracted "
              f"({se['extraction_rate']}), {se['llm_calls']} LLM calls "
              f"({se['efficiency_vs_per_turn']} vs per-turn){X}")
        print(f"  {D}TAI chunks: {orch.episodic.tai.count} | "
              f"Entities: {orch.semantic.entity_count} | "
              f"Relations: {orch.semantic.relation_count}{X}")

        # Blob
        blob = " ".join(blob_parts)
        blob_trunc = " ".join(blob.split()[:4500])  # ~6K tokens

        # Evaluate
        t0 = time.monotonic()
        conv_r = defaultdict(lambda: {"amem": 0, "blob": 0, "n": 0})

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))
            cat = qa.get("category", 0)
            if not question or not answer: continue

            try:
                # amem: cross-session knowledge retrieval
                knowledge = await orch.query_knowledge(question, top_k=30)
                amem_ctx = knowledge.to_injection_text()

                # Eval both with GPT-4o-mini
                amem_resp = await gpt4o([
                    {"role": "system", "content": "Answer based on memory context. Be concise. If unknown say 'I don't know'."},
                    {"role": "user", "content": f"Memory:\n{amem_ctx[:4000]}\n\nQ: {question}\nA:"}])
                blob_resp = await gpt4o([
                    {"role": "system", "content": "Answer based on memory context. Be concise. If unknown say 'I don't know'."},
                    {"role": "user", "content": f"Memory:\n{blob_trunc[:4000]}\n\nQ: {question}\nA:"}])
                api_calls += 2

                # Judge both
                for label, resp in [("amem", amem_resp), ("blob", blob_resp)]:
                    judge_r = await gpt4o([
                        {"role": "system", "content": "Judge correctness. Generous: same topic=CORRECT, date format diff=CORRECT. Only WRONG if clearly wrong or 'I don't know'. Reply CORRECT or WRONG."},
                        {"role": "user", "content": f"Q: {question}\nGold: {answer}\nPredicted: {resp}"}], temp=0.0)
                    api_calls += 1
                    if "CORRECT" in judge_r.upper():
                        conv_r[cat][label] += 1
                        conv_r["all"][label] += 1
                conv_r[cat]["n"] += 1
                conv_r["all"]["n"] += 1

            except Exception as e:
                if qi < 3: print(f"    {R}Err Q{qi}: {e}{X}")
                continue

            if (qi+1) % 25 == 0:
                n = conv_r["all"]["n"]
                a = conv_r["all"]["amem"]/max(n,1)*100
                b = conv_r["all"]["blob"]/max(n,1)*100
                print(f"    {D}...{qi+1}/{len(qa_pairs)} | amem={a:.0f}% blob={b:.0f}% | {api_calls} API calls{X}")

        eval_time = time.monotonic() - t0
        n = conv_r["all"]["n"]
        a = conv_r["all"]["amem"]/max(n,1)*100
        b = conv_r["all"]["blob"]/max(n,1)*100
        delta = a - b
        print(f"  {G if delta>=0 else R}{B}amem={a:.1f}%{X} blob={b:.1f}% {G if delta>=0 else R}Δ={delta:+.1f}%{X} ({n}q, {eval_time:.0f}s)")

        # Accumulate
        for k in conv_r:
            for m in ["amem", "blob", "n"]:
                results[k][m] += conv_r[k][m]

        orch.close()
        await embedder.close()

    # ── Final ──
    r = results["all"]
    n = max(r["n"], 1)
    a = r["amem"]/n*100
    b = r["blob"]/n*100

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  RESULTS — {n} questions across {MAX_CONVS} conversations{X}")
    print(f"{B}{'═'*70}{X}")

    print(f"\n  {B}Overall:{X}")
    print(f"    amem (selective, qwen3.5):  {G}{B}{a:.1f}%{X}")
    print(f"    Blob (OpenAI Memory):       {b:.1f}%")
    print(f"    Mem0 (published):           66.9%")
    print(f"    Delta vs blob:              {G if a>b else R}{a-b:+.1f}%{X}")
    print(f"    Delta vs Mem0:              {G if a>66.9 else R}{a-66.9:+.1f}%{X}")

    print(f"\n  {B}By Category:{X}")
    print(f"    {'Category':<15s} {'amem':>8s} {'Blob':>8s} {'Mem0*':>8s} {'Δ vs Mem0':>10s}  {'n':>4s}")
    print(f"    {'─'*55}")
    mem0_cats = {1: 67.1, 2: 51.2, 3: 55.5, 4: 72.9}
    for cat in sorted(int(c) for c in results if str(c).isdigit()):
        cr = results[cat]; cn = max(cr["n"],1)
        ap = cr["amem"]/cn*100; bp = cr["blob"]/cn*100
        mp = mem0_cats.get(cat, 0)
        d = ap - mp
        print(f"    {CATS.get(cat,'?'):<15s} {ap:>7.1f}% {bp:>7.1f}% {mp:>7.1f}% {G if d>=0 else R}{d:>+9.1f}%{X}  {cr['n']:>4d}")

    print(f"\n  {B}Cost:{X}")
    print(f"    Extraction LLM calls:  {total_extraction_calls} (selective, qwen3.5 local = $0)")
    print(f"    Eval+Judge API calls:  {api_calls} (GPT-4o-mini)")
    print(f"    Total turns processed: {total_turns}")
    print(f"    Mem0 would use:        ~{total_turns} extraction calls ($0.30+ per conv)")
    print(f"    amem used:             {total_extraction_calls} extraction calls ($0)")

    print(f"\n  {Y}{B}* Mem0 numbers from arXiv:2504.19413 (10 convs, 10 runs){X}")
    print(f"  {Y}{B}  amem numbers from this run (2 convs, 1 run){X}")
    print(f"\n{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
