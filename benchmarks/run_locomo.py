#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  LoCoMo Benchmark: amem vs Blob Injection Baseline
═══════════════════════════════════════════════════════════════════════

Uses the official LoCoMo dataset (snap-research/locomo) to measure
long-term conversational memory recall.

Methodology:
  1. Ingest all conversation turns into amem
  2. For each QA pair, query amem with the question
  3. Measure: does the retrieved context contain the answer?
  4. Compare against blob baseline (inject ALL summaries, always)

Metric: Recall@k — does the answer appear in the top-k retrieved chunks?
This is a pure retrieval metric (no LLM judge needed), making it
reproducible and unchallengeable.

Baseline: "Blob injection" simulates OpenAI Memory by concatenating
all session summaries and checking if the answer is in that blob.
The blob always contains everything — no filtering.
"""

import asyncio
import json
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from amem.config import Config
from amem.embeddings.factory import create_embedder
from amem.retrieval.orchestrator import MemoryOrchestrator

B="\033[1m"; G="\033[92m"; R="\033[91m"; Y="\033[93m"; D="\033[2m"; X="\033[0m"; C="\033[96m"

CATEGORY_NAMES = {1:"single-hop", 2:"multi-hop", 3:"temporal", 4:"open-ended", 5:"adversarial"}


def answer_in_text(answer: str, text: str) -> bool:
    """Check if the ground truth answer appears in the retrieved text.

    Uses normalized matching: lowercase, strip punctuation, partial match.
    """
    answer_clean = re.sub(r'[^\w\s]', '', str(answer).lower()).strip()
    text_clean = re.sub(r'[^\w\s]', '', str(text).lower()).strip()

    if not answer_clean:
        return False

    # Exact substring match
    if answer_clean in text_clean:
        return True

    # Token overlap: if 80%+ of answer tokens appear in text
    answer_tokens = set(answer_clean.split())
    text_tokens = set(text_clean.split())
    if not answer_tokens:
        return False
    overlap = len(answer_tokens & text_tokens) / len(answer_tokens)
    return overlap >= 0.8


async def main():
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  LoCoMo BENCHMARK: amem vs Blob Injection Baseline{X}")
    print(f"{B}{'═'*70}{X}")

    # Load dataset
    with open("benchmarks/locomo10.json") as f:
        data = json.load(f)

    print(f"\n  {D}Dataset: {len(data)} conversations, "
          f"{sum(len(c['qa']) for c in data)} QA pairs{X}")

    # Config
    config = Config()
    tmpdir = tempfile.mkdtemp()
    config.storage.data_dir = tmpdir

    # How many conversations to benchmark (full dataset = 10, takes ~30min)
    MAX_CONVS = 3  # Start with 3 for speed, increase to 10 for full benchmark
    TOP_K = 10

    total_amem_hits = 0
    total_blob_hits = 0
    total_questions = 0
    category_results = defaultdict(lambda: {"amem": 0, "blob": 0, "total": 0})

    for conv_idx in range(min(MAX_CONVS, len(data))):
        conv = data[conv_idx]
        conversation = conv["conversation"]
        qa_pairs = conv["qa"]
        speaker_a = conversation.get("speaker_a", "A")
        speaker_b = conversation.get("speaker_b", "B")

        print(f"\n  {C}{B}Conversation {conv_idx + 1}/{MAX_CONVS}: "
              f"{speaker_a} & {speaker_b} — {len(qa_pairs)} questions{X}")

        # ── Create fresh orchestrator for each conversation ──
        conv_dir = tempfile.mkdtemp()
        config.storage.data_dir = conv_dir
        embedder = create_embedder(config.ollama)
        orch = MemoryOrchestrator(embedder, config)
        orch.init_db(Path(conv_dir) / "amem.db")

        # ── Ingest all sessions ──
        session_keys = sorted([k for k in conversation.keys()
                               if k.startswith("session_") and not k.endswith("date_time")])

        all_text_blob = []  # For blob baseline
        ingest_count = 0
        t0 = time.monotonic()

        for sess_key in session_keys:
            session = conversation[sess_key]
            if not isinstance(session, list):
                continue
            session_date = conversation.get(f"{sess_key}_date_time", "")

            for turn in session:
                if not isinstance(turn, dict):
                    continue
                text = turn.get("text", turn.get("content", ""))
                speaker = turn.get("speaker", turn.get("role", ""))
                if not text:
                    continue

                # Ingest into amem
                await orch.ingest(
                    text=text,
                    conversation_id=sess_key,
                    speaker=speaker,
                )
                ingest_count += 1
                all_text_blob.append(text)

        ingest_time = time.monotonic() - t0
        print(f"    {D}Ingested {ingest_count} turns in {ingest_time:.1f}s "
              f"({ingest_count/max(ingest_time,0.1):.1f} turns/sec){X}")

        # ── Blob baseline: concatenate everything ──
        blob_text = " ".join(all_text_blob).lower()

        # ── Evaluate each QA pair ──
        conv_amem = 0
        conv_blob = 0
        conv_total = 0
        t0 = time.monotonic()

        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = qa.get("answer", qa.get("adversarial_answer", ""))
            category = qa.get("category", 0)

            if not answer or not question:
                continue

            # amem: query and check if answer is in retrieved context
            ctx = await orch.query(question, top_k=TOP_K)
            retrieved_text = " ".join(c["text"] for c in ctx.episodic_chunks)
            # Also include semantic facts
            for f in ctx.semantic_facts:
                retrieved_text += f" {f.get('subject','')} {f.get('predicate','')} {f.get('object','')}"
            # And explicit
            for e in ctx.explicit_entries:
                retrieved_text += f" {e.get('key','')} {e.get('value','')}"

            amem_hit = answer_in_text(answer, retrieved_text)

            # Blob baseline: check if answer is in the full blob
            blob_hit = answer_in_text(answer, blob_text)

            if amem_hit:
                conv_amem += 1
                total_amem_hits += 1
            if blob_hit:
                conv_blob += 1
                total_blob_hits += 1

            conv_total += 1
            total_questions += 1

            category_results[category]["total"] += 1
            if amem_hit:
                category_results[category]["amem"] += 1
            if blob_hit:
                category_results[category]["blob"] += 1

        eval_time = time.monotonic() - t0
        amem_pct = conv_amem / max(conv_total, 1) * 100
        blob_pct = conv_blob / max(conv_total, 1) * 100

        print(f"    {D}Evaluated {conv_total} questions in {eval_time:.1f}s{X}")
        print(f"    amem recall@{TOP_K}: {G}{amem_pct:.1f}%{X} ({conv_amem}/{conv_total})")
        print(f"    Blob recall:        {blob_pct:.1f}% ({conv_blob}/{conv_total})")
        delta = amem_pct - blob_pct
        color = G if delta >= 0 else R
        print(f"    Delta:              {color}{delta:+.1f}%{X}")

        orch.close()
        await embedder.close()

    # ── Final Results ──
    print(f"\n{B}{'═'*70}{X}")
    print(f"{B}  RESULTS — {total_questions} questions across {MAX_CONVS} conversations{X}")
    print(f"{B}{'═'*70}{X}")

    amem_total_pct = total_amem_hits / max(total_questions, 1) * 100
    blob_total_pct = total_blob_hits / max(total_questions, 1) * 100
    delta_total = amem_total_pct - blob_total_pct

    print(f"\n  {B}Overall Recall@{TOP_K}:{X}")
    print(f"    amem:          {G}{B}{amem_total_pct:.1f}%{X} ({total_amem_hits}/{total_questions})")
    print(f"    Blob baseline: {blob_total_pct:.1f}% ({total_blob_hits}/{total_questions})")
    color = G if delta_total >= 0 else R
    print(f"    Delta:         {color}{B}{delta_total:+.1f}%{X}")

    print(f"\n  {B}By Category:{X}")
    print(f"    {'Category':<15s} {'amem':>8s} {'Blob':>8s} {'Delta':>8s}  {'n':>4s}")
    print(f"    {'─'*50}")
    for cat in sorted(category_results.keys()):
        r = category_results[cat]
        a_pct = r["amem"] / max(r["total"], 1) * 100
        b_pct = r["blob"] / max(r["total"], 1) * 100
        d = a_pct - b_pct
        name = CATEGORY_NAMES.get(cat, f"cat-{cat}")
        color = G if d >= 0 else R
        print(f"    {name:<15s} {a_pct:>7.1f}% {b_pct:>7.1f}% {color}{d:>+7.1f}%{X}  {r['total']:>4d}")

    print(f"\n  {B}Methodology:{X}")
    print(f"    Dataset:    LoCoMo (snap-research/locomo) — {MAX_CONVS}/{len(data)} conversations")
    print(f"    Metric:     Recall@{TOP_K} (answer found in retrieved context)")
    print(f"    amem:       Query-conditioned retrieval from 5 memory layers")
    print(f"    Blob:       Full conversation text always available (OpenAI Memory proxy)")
    print(f"    Embedding:  Ollama nomic-embed-text (768 dims)")
    print(f"    Matching:   Normalized substring + 80% token overlap")
    print(f"\n  {D}Note: Blob baseline has access to ALL text, always. It cannot 'miss'{X}")
    print(f"  {D}retrieving content — it tests whether the answer exists in the data.{X}")
    print(f"  {D}amem tests whether query-conditioned retrieval finds the right subset.{X}")
    print(f"\n{B}{'═'*70}{X}\n")


if __name__ == "__main__":
    asyncio.run(main())
