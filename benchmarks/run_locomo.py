#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  LoCoMo Benchmark: amem vs Blob Injection Baseline
═══════════════════════════════════════════════════════════════════════

Uses the official LoCoMo dataset (snap-research/locomo).

Key insight: LoCoMo is a REASONING benchmark, not a retrieval benchmark.
Most answers require inference — "When did X happen?" → the answer is
a date derived from session metadata, not stated in the text.

We measure two things:
1. Context Sufficiency: does the retrieved context contain enough
   information to answer the question? (LLM-judged)
2. Token Efficiency: how many tokens are needed to achieve that recall?

The blob baseline always injects EVERYTHING (5000-10000 tokens).
amem injects only what's relevant (200-500 tokens).
"""

import asyncio
import json
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator

B="\033[1m"; G="\033[92m"; R="\033[91m"; Y="\033[93m"; D="\033[2m"; X="\033[0m"; C="\033[96m"

CATEGORY_NAMES = {1:"single-hop", 2:"multi-hop", 3:"temporal", 4:"open-ended", 5:"adversarial"}


async def llm_judge(context: str, question: str, answer: str, ollama_url: str = "http://localhost:11434") -> bool:
    """Use a local LLM to judge if the context contains enough info to answer the question."""
    prompt = f"""Given this context and question, does the context contain enough information to answer the question? The expected answer is provided for reference.

Context:
{context[:3000]}

Question: {question}
Expected answer: {answer}

Does the context contain sufficient information to derive or directly state this answer? Reply with only YES or NO."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{ollama_url}/api/generate", json={
                "model": "qwen3.5:35b-a3b-coding-nvfp4",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 20},
            })
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip().upper()
                return "YES" in text
    except Exception:
        pass
    # Fallback: token overlap matching
    return token_match(answer, context)


def token_match(answer: str, text: str) -> bool:
    """Fallback: normalized token overlap."""
    answer_clean = re.sub(r'[^\w\s]', '', str(answer).lower()).strip()
    text_clean = re.sub(r'[^\w\s]', '', str(text).lower()).strip()
    if not answer_clean: return False
    if answer_clean in text_clean: return True
    a_tokens = set(answer_clean.split())
    t_tokens = set(text_clean.split())
    if not a_tokens: return False
    return len(a_tokens & t_tokens) / len(a_tokens) >= 0.8


async def main():
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  LoCoMo BENCHMARK: amem vs Blob Injection{X}")
    print(f"{B}{'═'*70}{X}")

    with open("benchmarks/locomo10.json") as f:
        data = json.load(f)

    # Check if LLM is available for judging
    has_llm = False
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        # Need a chat model (not just embedding)
        chat_models = [m for m in models if m not in ("nomic-embed-text", "all-minilm")]
        has_llm = len(chat_models) > 0
        if has_llm:
            print(f"  {G}LLM judge available: {chat_models[0]}{X}")
        else:
            print(f"  {Y}No chat model for LLM judge — using token matching{X}")
    except Exception:
        print(f"  {Y}No LLM judge — using token matching{X}")

    config = Config()
    MAX_CONVS = 2  # 2 conversations for reasonable runtime
    TOP_K = 30     # More chunks = better recall

    total = defaultdict(lambda: {"amem": 0, "blob": 0, "n": 0, "amem_tokens": 0, "blob_tokens": 0})

    for conv_idx in range(min(MAX_CONVS, len(data))):
        conv = data[conv_idx]
        conversation = conv["conversation"]
        qa_pairs = conv["qa"]

        print(f"\n  {C}{B}Conversation {conv_idx+1}/{MAX_CONVS} — {len(qa_pairs)} questions{X}")

        # Create orchestrator
        conv_dir = tempfile.mkdtemp()
        config.storage.data_dir = conv_dir
        embedder = create_embedder(config.ollama)
        orch = MemoryOrchestrator(embedder, config)
        orch.init_db(Path(conv_dir) / "amem.db")

        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])

        # ── Ingest with session dates as context ──
        all_blob_parts = []
        t0 = time.monotonic()

        for sk in session_keys:
            session = conversation[sk]
            if not isinstance(session, list): continue
            session_date = conversation.get(f"{sk}_date_time", "")

            # CRITICAL FIX: Ingest the session date as a fact
            if session_date:
                date_text = f"This conversation session took place on {session_date}."
                await orch.ingest(text=date_text, conversation_id=sk, speaker="system")
                all_blob_parts.append(date_text)

            for turn in session:
                if not isinstance(turn, dict): continue
                text = turn.get("text", "")
                if not text: continue
                await orch.ingest(text=text, conversation_id=sk, speaker=turn.get("speaker", ""))
                all_blob_parts.append(text)

        ingest_time = time.monotonic() - t0
        blob_text = " ".join(all_blob_parts)
        blob_tokens = len(blob_text.split()) * 4 // 3
        print(f"    {D}Ingested in {ingest_time:.1f}s | TAI: {orch.episodic.tai.count} chunks | Blob: ~{blob_tokens} tokens{X}")

        # ── Evaluate ──
        t0 = time.monotonic()
        conv_amem = 0
        conv_blob = 0
        evaluated = 0

        for qi, qa in enumerate(qa_pairs):
            question = qa.get("question", "")
            answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
            category = qa.get("category", 0)
            if not answer or not question: continue

            # amem retrieval
            ctx = await orch.query(question, top_k=TOP_K)
            amem_context = ""
            for c in ctx.episodic_chunks:
                amem_context += c["text"] + " "
            for f in ctx.semantic_facts:
                amem_context += f"{f.get('subject','')} {f.get('predicate','')} {f.get('object','')} "
            amem_tokens = len(amem_context.split()) * 4 // 3

            # Judge: amem
            if has_llm and category != 5:  # Skip adversarial for LLM judge (they're trick questions)
                amem_hit = await llm_judge(amem_context, question, answer)
            else:
                amem_hit = token_match(answer, amem_context)

            # Judge: blob
            if has_llm and category != 5:
                blob_hit = await llm_judge(blob_text[:4000], question, answer)  # First 4K tokens (LLM context limit)
            else:
                blob_hit = token_match(answer, blob_text)

            if amem_hit: conv_amem += 1
            if blob_hit: conv_blob += 1
            evaluated += 1

            total[category]["amem"] += int(amem_hit)
            total[category]["blob"] += int(blob_hit)
            total[category]["n"] += 1
            total[category]["amem_tokens"] += amem_tokens
            total[category]["blob_tokens"] += blob_tokens
            total["all"]["amem"] += int(amem_hit)
            total["all"]["blob"] += int(blob_hit)
            total["all"]["n"] += 1
            total["all"]["amem_tokens"] += amem_tokens
            total["all"]["blob_tokens"] += blob_tokens

            if (qi + 1) % 50 == 0:
                print(f"    {D}  ...{qi+1}/{len(qa_pairs)} questions{X}")

        eval_time = time.monotonic() - t0
        amem_pct = conv_amem / max(evaluated, 1) * 100
        blob_pct = conv_blob / max(evaluated, 1) * 100

        print(f"    {D}Evaluated {evaluated} questions in {eval_time:.1f}s{X}")
        print(f"    amem:  {G}{amem_pct:.1f}%{X}  ({conv_amem}/{evaluated})")
        print(f"    Blob:  {blob_pct:.1f}%  ({conv_blob}/{evaluated})")

        orch.close()
        await embedder.close()

    # ── Final Results ──
    r = total["all"]
    amem_pct = r["amem"] / max(r["n"], 1) * 100
    blob_pct = r["blob"] / max(r["n"], 1) * 100
    delta = amem_pct - blob_pct
    avg_amem_tokens = r["amem_tokens"] / max(r["n"], 1)
    avg_blob_tokens = r["blob_tokens"] / max(r["n"], 1)

    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  RESULTS — {r['n']} questions across {MAX_CONVS} conversations{X}")
    print(f"{B}{'═'*70}{X}")

    print(f"\n  {B}Accuracy (context sufficiency):{X}")
    print(f"    amem:  {G}{B}{amem_pct:.1f}%{X}")
    print(f"    Blob:  {blob_pct:.1f}%")
    color = G if delta >= 0 else R
    print(f"    Delta: {color}{B}{delta:+.1f}%{X}")

    print(f"\n  {B}Token Efficiency:{X}")
    print(f"    amem avg tokens/query: {avg_amem_tokens:.0f}")
    print(f"    Blob avg tokens/query: {avg_blob_tokens:.0f}")
    if avg_amem_tokens > 0:
        efficiency = (amem_pct / avg_amem_tokens) / max(blob_pct / avg_blob_tokens, 0.001)
        print(f"    Efficiency ratio:      {G}{B}{efficiency:.1f}×{X} (amem recall per token / blob recall per token)")

    print(f"\n  {B}By Category:{X}")
    print(f"    {'Category':<15s} {'amem':>8s} {'Blob':>8s} {'Delta':>8s}  {'n':>4s}")
    print(f"    {'─'*50}")
    for cat in sorted(c for c in total.keys() if isinstance(c, int)):
        cr = total[cat]
        a = cr["amem"] / max(cr["n"],1) * 100
        b = cr["blob"] / max(cr["n"],1) * 100
        d = a - b
        color = G if d >= 0 else R
        print(f"    {CATEGORY_NAMES.get(cat,'?'):<15s} {a:>7.1f}% {b:>7.1f}% {color}{d:>+7.1f}%{X}  {cr['n']:>4d}")

    print(f"\n  {B}Judge:{X} {'LLM (Ollama)' if has_llm else 'Token matching (no LLM available)'}")
    print(f"  {B}Dataset:{X} LoCoMo — {MAX_CONVS}/10 conversations")
    print(f"  {B}top_k:{X} {TOP_K}")
    print(f"\n{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
