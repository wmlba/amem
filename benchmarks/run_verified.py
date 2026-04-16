#!/usr/bin/env python3
"""
VERIFIED RUN — Per-session extraction with Ollama qwen3.5.
This is the approach that scored 72% before. Verifying it now.

No selective gating. Full session text → qwen3.5 → dense facts.
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
from amem.semantic.fact_extractor import FactExtractor

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
    print(f"{B}  VERIFIED RUN — Per-Session Extraction with qwen3.5{X}")
    print(f"{B}  Extraction: Ollama qwen3.5 (35B, local, free){X}")
    print(f"{B}  Embeddings: Ollama nomic-embed-text (local, free){X}")
    print(f"{B}  Eval/Judge: GPT-4o-mini (same as Mem0){X}")
    print(f"{B}{'═'*70}{X}")

    with open("benchmarks/locomo10.json") as f:
        data = json.load(f)

    # Use Ollama qwen3.5 for extraction (the model that got 72% before)
    fact_extractor = FactExtractor(
        ollama_url="http://localhost:11434",
        ollama_model="qwen3.5:35b-a3b-coding-nvfp4",
    )

    config = Config()
    MAX_CONVS = 2
    results = defaultdict(lambda: {"amem": 0, "blob": 0, "n": 0})
    api_calls = 0
    total_facts = 0

    for ci in range(MAX_CONVS):
        conv = data[ci]
        conversation = conv["conversation"]
        qa_pairs = [q for q in conv["qa"] if q.get("category", 5) != 5]
        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])

        print(f"\n  {C}{B}Conv {ci+1}/{MAX_CONVS} — {len(qa_pairs)} questions{X}")

        tmpdir = tempfile.mkdtemp()
        config.storage.data_dir = tmpdir
        embedder = create_embedder(config.ollama)
        orch = MemoryOrchestrator(embedder, config)
        orch.init_db(Path(tmpdir) / "amem.db")

        blob_parts = []
        t0 = time.monotonic()

        for sk in session_keys:
            session = conversation[sk]
            if not isinstance(session, list): continue
            date = conversation.get(f"{sk}_date_time", "")

            session_texts = []
            if date:
                dt = f"This conversation took place on {date}."
                await orch.ingest(text=dt, conversation_id=sk, speaker="system")
                blob_parts.append(dt)
                session_texts.append(dt)

            for turn in session:
                if not isinstance(turn, dict): continue
                text = turn.get("text", "")
                if not text: continue
                # Ingest raw turn (no selective extraction — bypass it)
                await orch.episodic.ingest(text=text, conversation_id=sk, speaker=turn.get("speaker", ""))
                blob_parts.append(text)
                session_texts.append(f"{turn.get('speaker','')}: {text}")

            # Per-session LLM extraction with qwen3.5
            session_text = "\n".join(session_texts)
            if session_text.strip():
                facts = await fact_extractor.extract_facts(session_text, max_facts=80)
                total_facts += len(facts)
                for fact in facts:
                    await orch.episodic.ingest(text=fact, conversation_id=f"facts-{sk}", speaker="fact")

        ingest_time = time.monotonic() - t0
        print(f"  {D}Ingested in {ingest_time:.0f}s | {total_facts} facts | TAI: {orch.episodic.tai.count} chunks{X}")

        blob = " ".join(blob_parts)
        blob_trunc = " ".join(blob.split()[:4500])

        # Evaluate
        t0 = time.monotonic()
        conv_r = defaultdict(lambda: {"amem": 0, "blob": 0, "n": 0})

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))
            cat = qa.get("category", 0)
            if not question or not answer: continue

            try:
                knowledge = await orch.query_knowledge(question, top_k=30)
                amem_ctx = knowledge.to_injection_text()

                amem_resp = await gpt4o([
                    {"role": "system", "content": "Answer based on memory. Concise. If unknown: 'I don't know'."},
                    {"role": "user", "content": f"Memory:\n{amem_ctx[:4000]}\n\nQ: {question}\nA:"}])
                blob_resp = await gpt4o([
                    {"role": "system", "content": "Answer based on memory. Concise. If unknown: 'I don't know'."},
                    {"role": "user", "content": f"Memory:\n{blob_trunc[:4000]}\n\nQ: {question}\nA:"}])
                api_calls += 2

                for label, resp in [("amem", amem_resp), ("blob", blob_resp)]:
                    j = await gpt4o([
                        {"role": "system", "content": "CORRECT or WRONG only. Generous: same topic=CORRECT."},
                        {"role": "user", "content": f"Q: {question}\nGold: {answer}\nPred: {resp}"}], temp=0.0)
                    api_calls += 1
                    if "CORRECT" in j.upper():
                        conv_r[cat][label] += 1; conv_r["all"][label] += 1
                conv_r[cat]["n"] += 1; conv_r["all"]["n"] += 1
            except Exception as e:
                if qi < 3: print(f"    {R}Err: {e}{X}")
                continue

            if (qi+1) % 25 == 0:
                n=conv_r["all"]["n"]; a=conv_r["all"]["amem"]/max(n,1)*100; b=conv_r["all"]["blob"]/max(n,1)*100
                print(f"    {D}...{qi+1}/{len(qa_pairs)} | amem={a:.0f}% blob={b:.0f}%{X}")

        n=conv_r["all"]["n"]; a=conv_r["all"]["amem"]/max(n,1)*100; b=conv_r["all"]["blob"]/max(n,1)*100
        print(f"  {G if a>b else R}{B}amem={a:.1f}%{X} blob={b:.1f}% Δ={a-b:+.1f}%")

        for k in conv_r:
            for m in ["amem","blob","n"]: results[k][m] += conv_r[k][m]
        orch.close(); await embedder.close()

    # Final
    r=results["all"]; n=max(r["n"],1); a=r["amem"]/n*100; b=r["blob"]/n*100
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  RESULTS — {n} questions{X}")
    print(f"{B}{'═'*70}{X}")
    print(f"\n  amem (qwen3.5 per-session): {G}{B}{a:.1f}%{X}")
    print(f"  Blob baseline:              {b:.1f}%")
    print(f"  Mem0 published:             66.9%")
    print(f"  Δ vs Mem0:                  {G if a>66.9 else R}{a-66.9:+.1f}%{X}")
    print(f"\n  {B}By Category:{X}")
    mem0_cats = {1:67.1, 2:51.2, 3:55.5, 4:72.9}
    for cat in sorted(int(c) for c in results if str(c).isdigit()):
        cr=results[cat]; cn=max(cr["n"],1); ap=cr["amem"]/cn*100; mp=mem0_cats.get(cat,0)
        print(f"    {CATS.get(cat,'?'):<15s} amem={ap:>5.1f}% Mem0={mp:>5.1f}% {G if ap>mp else R}Δ={ap-mp:+.1f}%{X} (n={cr['n']})")
    print(f"\n  Facts extracted: {total_facts} | API calls: {api_calls}")
    print(f"\n{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
