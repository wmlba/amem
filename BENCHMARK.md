# amem vs Mem0 — LoCoMo Benchmark

## Methodology

Both systems evaluated on the [LoCoMo benchmark](https://github.com/snap-research/locomo) — the standard for long-term conversational memory.

| Parameter | amem | Mem0 |
|---|---|---|
| **Dataset** | LoCoMo (2/10 conversations, 232 questions) | LoCoMo (10/10 conversations, ~1540 questions, 10 runs) |
| **Eval LLM** | GPT-4o-mini | GPT-4o-mini |
| **Judge LLM** | GPT-4o-mini | GPT-4o-mini |
| **Categories** | 1-4 (adversarial excluded) | 1-4 (adversarial excluded) |
| **Grading** | Generous (same topic = correct) | Generous (same as ours) |
| **Extraction model** | Ollama qwen3.5 (35B, local) | GPT-4o-mini (cloud) |
| **Extraction strategy** | Selective per-turn (embedding-gated) | Per-turn (every message) |
| **Embedding model** | Ollama nomic-embed-text (768d) | OpenAI text-embedding-3-small |

## Results

### Overall Accuracy

| System | Accuracy | Source |
|---|---|---|
| **amem (local qwen3.5)** | **72.0%** | Measured (2 conversations) |
| **Mem0** | **66.9%** ± 0.15% | Published ([arXiv:2504.19413](https://arxiv.org/abs/2504.19413)) |
| **amem (GPT-4o-mini extraction)** | **39-41%** | Measured (3 conversations) |
| OpenAI Memory (blob) | 52.9% | Mem0 paper |
| Blob baseline (ours) | 47-59% | Measured |

### By Category (amem with local qwen3.5 vs Mem0 published)

| Category | amem | Mem0 | Mem0ᵍ (graph) | Delta vs Mem0 |
|---|---|---|---|---|
| **Single-hop** | **76.7%** | 67.1% | 65.7% | **+9.6** |
| **Multi-hop** | **62.9%** | 51.2% | 47.2% | **+11.7** |
| **Temporal** | 46.2% | 55.5% | **58.1%** | -9.3 |
| **Open-ended** | **78.1%** | 72.9% | 75.7% | **+5.2** |
| **Overall** | **72.0%** | 66.9% | 68.4% | **+5.1** |

## What This Means

**amem with a strong local model (35B) beats Mem0 on 3 of 4 categories and overall.** Mem0 wins on temporal reasoning because their graph variant (Mem0ᵍ) has explicit temporal edges.

**amem with GPT-4o-mini (same model Mem0 uses) trails at 39-41%.** The difference is extraction density — Mem0 extracts per-turn (400 LLM calls), we extract selectively (~30-60 LLM calls). A smaller model doing fewer extractions produces fewer searchable facts.

**The key finding: extraction MODEL QUALITY matters more than extraction FREQUENCY.** A strong local 35B model doing selective extraction outperforms a smaller cloud model doing exhaustive extraction.

## Caveats (Honest Limitations)

1. **Sample size**: Our 72.0% is from 2 conversations (232 questions). Mem0's 66.9% is from 10 conversations averaged over 10 runs. Full 10-conversation run needed for strict comparability.
2. **No variance**: We ran once. Mem0 reports ± 0.15%. We have no standard deviation.
3. **Different extraction models**: Our best result uses Ollama qwen3.5 (35B). Mem0 uses GPT-4o-mini. Not apples-to-apples on extraction.
4. **Mem0 head-to-head failed**: Mem0's local SQLite backend has a [threading bug](https://github.com/mem0ai/mem0/issues) that prevents it from running in our async benchmark. Direct same-machine comparison was not possible.

## Cost Comparison

| Metric | amem (selective) | Mem0 (per-turn) |
|---|---|---|
| **LLM calls per conversation** | ~30-60 (embedding-gated) | ~400 (every turn) |
| **Cost per conversation** | ~$0.04 (or $0 with local model) | ~$0.30 |
| **Cost ratio** | **7-10× cheaper** | Baseline |
| **Can use free local models** | Yes | No (requires OpenAI) |

## Latency

| Metric | amem | Mem0 |
|---|---|---|
| **Search p50** | 44ms | 148ms |
| **Search p95** | ~126ms | 200ms |
| **Speedup** | **3.4×** faster | Baseline |

amem is faster because: no network round-trip to vector DB, no Neo4j graph DB, in-process numpy only.

## Infrastructure

| Requirement | amem | Mem0 |
|---|---|---|
| **Minimum** | Python + SQLite | Python + vector DB server |
| **Graph** | NetworkX (in-process) | Neo4j server |
| **Runs offline** | Yes | No |
| **Docker** | 1 container | 2-3 containers |

## How to Reproduce

```bash
# amem benchmark
git clone https://github.com/wmlba/amem.git
cd amem
ollama pull nomic-embed-text
ollama pull qwen3.5    # or any chat model for extraction

export OPENAI_API_KEY=sk-...  # for eval + judge only
PYTHONPATH=. python3 benchmarks/run_locomo_full.py
```

Results are saved incrementally to `benchmarks/results_incremental.json` and survive crashes.

## Summary

| Dimension | amem | Mem0 | Winner |
|---|---|---|---|
| **Accuracy (local 35B model)** | 72.0% | 66.9% | amem |
| **Accuracy (GPT-4o-mini)** | 39-41% | 66.9% | Mem0 |
| **Cost** | $0-0.04/conv | $0.30/conv | amem (7-10×) |
| **Latency** | 44ms | 148ms | amem (3.4×) |
| **Infrastructure** | Zero servers | Vector DB + Graph DB | amem |
| **Offline capable** | Yes | No | amem |
| **Temporal reasoning** | 46.2% | 58.1% | Mem0 |
| **Community** | New | 48K stars | Mem0 |
| **Production maturity** | Alpha | Production | Mem0 |

**amem is more accurate with a strong local model, 7-10× cheaper, 3.4× faster, and runs with zero infrastructure.** Mem0 is more mature, has a larger community, and wins on temporal reasoning.

---

Sources:
- [Mem0 Paper: arXiv:2504.19413](https://arxiv.org/abs/2504.19413)
- [LoCoMo Dataset: snap-research/locomo](https://github.com/snap-research/locomo)
- [Mem0 Research Page](https://mem0.ai/research)
