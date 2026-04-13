<p align="center">
  <h1 align="center">amem</h1>
  <p align="center"><strong>The five-layer brain for AI agents.</strong></p>
  <p align="center">
    Episodic · Semantic · Behavioral · Working · Explicit<br/>
    Five memory layers. One embedding model. Zero LLM calls.
  </p>
</p>

<p align="center">
  <a href="https://github.com/wmlba/amem/actions"><img src="https://img.shields.io/badge/tests-212%20passed-brightgreen" alt="Tests"></a>
  <a href="https://github.com/wmlba/amem/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="https://pypi.org/project/amem"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python"></a>
  <a href="https://github.com/wmlba/amem"><img src="https://img.shields.io/github/stars/wmlba/amem?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#use-with-local-models">Local Models</a> ·
  <a href="#integrations">Integrations</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#docs">Docs</a>
</p>

---

## Why amem?

Current LLM memory stores everything as text blobs and injects **all** of it into **every** context window. amem gives your AI a structured brain instead:

```
┌─────────────────────────────────────────────────────────────────┐
│                        amem                                     │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ │
│  │ Episodic │ │ Semantic │ │Behavioral│ │Working │ │Explicit│ │
│  │          │ │          │ │          │ │        │ │        │ │
│  │ "I       │ │ "Alice   │ │ "You     │ │ "We're │ │ "My    │ │
│  │ remember │ │ leads    │ │ prefer   │ │ debug- │ │ name   │ │
│  │ that     │ │ the ML   │ │ concise  │ │ ging   │ │ is     │ │
│  │ convo"   │ │ team"    │ │ answers" │ │ auth"  │ │ Will"  │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ └────────┘ │
│                        │                                        │
│              Query-conditioned retrieval                        │
│              Only inject what's relevant                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 5 lines to memory

```python
from amem.retrieval.orchestrator import MemoryOrchestrator

orch = MemoryOrchestrator.from_config(config)  # auto-detects embedding provider
orch.init_db()                                  # SQLite, crash-safe, zero config

await orch.ingest(text="Alice is an ML engineer at Google.", speaker="user")
ctx = await orch.query("What does Alice do?")
print(ctx.to_injection_text())
# → Episodic: "Alice is an ML engineer at Google." (score: 0.87)
# → Semantic: Alice —[works_at]→ Google (confidence: 0.80)
```

### Install

```bash
pip install numpy networkx httpx msgpack fastapi uvicorn click pydantic pyyaml tiktoken

git clone https://github.com/wmlba/amem.git
cd amem
```

### Choose your embedding provider

<table>
<tr><td><strong>Ollama (recommended)</strong></td><td><strong>Completely offline</strong></td></tr>
<tr>
<td>

```bash
ollama pull nomic-embed-text
```
```yaml
# config.yaml
embedding:
  provider: ollama
  model: nomic-embed-text
```

</td>
<td>

```bash
pip install sentence-transformers
```
```yaml
# config.yaml
embedding:
  provider: local
  model: all-MiniLM-L6-v2
```

</td>
</tr>
<tr><td><strong>OpenAI</strong></td><td><strong>Any OpenAI-compatible server</strong></td></tr>
<tr>
<td>

```yaml
embedding:
  provider: openai
  model: text-embedding-3-small
  api_key: sk-...
```

</td>
<td>

```yaml
# vLLM, llama.cpp, LiteLLM, Together,
# Anyscale, OpenRouter, etc.
embedding:
  provider: openai
  model: your-model
  base_url: http://localhost:8000/v1
```

</td>
</tr>
</table>

### Start the server

```bash
python -m cli.main serve        # API on :8420
open http://localhost:8420/dashboard  # Admin UI
```

---

## Use with Local Models

amem runs **100% locally** with zero cloud dependencies. Three options:

### Option A: Ollama (easiest)

```bash
# Install Ollama (https://ollama.com)
ollama pull nomic-embed-text        # 274MB, good quality

# Or use a smaller model
ollama pull all-minilm               # 46MB, fastest
```

```yaml
# config.yaml
embedding:
  provider: ollama
  model: nomic-embed-text            # or all-minilm
```

Works with any model Ollama supports. The same model handles embeddings, entity extraction, deduplication, and entity resolution — no separate LLM needed.

### Option B: sentence-transformers (no server)

```bash
pip install sentence-transformers
```

```yaml
# config.yaml
embedding:
  provider: local
  model: all-MiniLM-L6-v2           # 384 dims, fast
  # model: all-mpnet-base-v2        # 768 dims, better quality
  # model: BAAI/bge-small-en-v1.5   # 384 dims, very good
```

Downloads the model on first use (~80MB), then runs entirely from cache. **No server, no network, no API key.**

### Option C: llama.cpp / vLLM / LiteLLM

Any server that exposes `/v1/embeddings` works:

```bash
# Example: llama.cpp with an embedding model
./llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --port 8000 --embedding
```

```yaml
embedding:
  provider: openai
  model: nomic-embed-text
  base_url: http://localhost:8000/v1
```

### Option D: Auto-detect

```yaml
embedding:
  provider: auto    # tries Ollama → local sentence-transformers → OPENAI_API_KEY env
```

---

## How It Works

### The Five Memory Layers

| Layer | What it stores | How it works |
|---|---|---|
| **Episodic** | Raw conversation chunks | Temporal Associative Index — fused vectorized scoring across time-tiered shards (hot/warm/cold) |
| **Semantic** | Entities and relationships | Knowledge graph with entity resolution, contradiction detection, adaptive decay |
| **Behavioral** | How to interact with this user | 4-dimension EMA profile (depth, formality, expertise, verbosity) |
| **Working** | Current session context | Goals, established facts, open threads — flushed to episodic on session end |
| **Explicit** | User-controlled facts | Typed key-value store with priority — never decays, always injected |

### What makes this different from Mem0 / Zep / Letta

| Capability | Mem0 | Zep | Letta | amem |
|---|---|---|---|---|
| Memory layers | 3 scopes | 1 (graph) | 3 tiers | **5 layers** |
| Extraction method | LLM call | LLM call | LLM call | **Embedding only (no LLM)** |
| Temporal decay | LRU only | Not impl. | No | **4 adaptive tiers** |
| Intent-aware scoring | No | No | No | **Yes** |
| Relevance feedback | No | No | No | **Yes** |
| Behavioral profiling | No | No | Partial | **Yes** |
| Works offline | No | No | No | **Yes** |
| Zero config | No | Neo4j req. | No | **SQLite, auto-detect** |

---

## Integrations

### MCP Server — Claude Desktop, Cursor, OpenClaw

```json
{
  "mcpServers": {
    "amem": {
      "command": "python3",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/amem"
    }
  }
}
```

9 tools: `memory_ingest`, `memory_query`, `memory_remember`, `memory_forget`, `memory_list`, `memory_graph`, `memory_retract`, `memory_merge_entities`, `memory_stats`

### OpenAI-Compatible Proxy — Drop-in memory for any app

```bash
python -m api.openai_compat --target https://api.openai.com/v1 --port 8421
```

```python
# Existing code — just change the base URL
client = OpenAI(base_url="http://localhost:8421/v1")
# Memory is injected automatically into every request
```

### REST API — 26 endpoints

```bash
curl -X POST http://localhost:8420/ingest \
  -d '{"text": "Alice works on ML pipelines.", "speaker": "user"}'

curl -X POST http://localhost:8420/query \
  -d '{"query": "What does Alice work on?"}'
```

### CLI — 15 commands

```bash
amem ingest "Alice works on ML pipelines."
amem query "What does Alice work on?"
amem remember name Will --type fact --priority 10
amem graph Alice
amem serve
```

### Admin Dashboard

```
http://localhost:8420/dashboard
```

Live web UI with: system stats, knowledge graph visualization (force-directed, interactive), memory timeline, query explorer with budget allocation breakdown, behavioral profile bars.

---

## Key Features

<details>
<summary><strong>Temporal Associative Index (TAI)</strong> — novel vector index for memory</summary>

Not HNSW. Not FAISS. A purpose-built index where time is structural:

- **Time-tiered shards**: Hot (brute-force, sub-ms) → Warm → Cold (archival)
- **Fused scoring**: `similarity × temporal × reinforcement × importance` in one numpy pass
- **O(n) top-k** via argpartition
- **Batch dedup + novelty** in one matmul
- **Smart dedup**: Checks distinctive tokens, not just cosine
- **Auto-compaction**: Hot → warm, cold eviction
</details>

<details>
<summary><strong>Entity Resolution</strong> — "k8s" = "Kubernetes" = "kube"</summary>

- Fuzzy matching (typos: "Kuberntes" → "Kubernetes")
- Alias tracking, possessive removal, acronym expansion
- Vector similarity for semantic matches
- Entity merging with relation consolidation
</details>

<details>
<summary><strong>Contradiction Detection</strong> — "left OCI" supersedes "works at OCI"</summary>

- Direct conflicts detected on exclusive predicates
- Temporal resolution: newer facts win
- Negation detection: "left X" negates "works at X"
- Fact retraction: user-driven corrections
</details>

<details>
<summary><strong>Intent-Aware Scoring</strong> — query analysis adjusts weights</summary>

- "What's my **current** status?" → boosts temporal weight
- "What's the **exact** config?" → boosts similarity weight
- "Tell me **everything**" → flattens all weights
- "**Team Beta** issue" → boosts context-anchor matching
</details>

<details>
<summary><strong>Relevance Feedback</strong> — retrieval improves with use</summary>

- Compares LLM response against retrieved chunks
- Used chunks reinforced, ignored chunks demoted
- Retrieval quality improves automatically over time
</details>

<details>
<summary><strong>Memory Consolidation</strong> — like human sleep</summary>

- Frequent entities promoted to semantic graph
- Low-confidence old chunks evicted
- Topic clusters detected from co-retrieval
</details>

---

## Benchmarks

Measured with live Ollama `nomic-embed-text` (768 dimensions).

### Performance

| Operation | Measured |
|---|---|
| Query (retrieve + rank + assemble) | **44ms avg** at 50 chunks |
| Ingest (embed + store + extract) | **~300ms** per message |
| Explicit memory set/get | **0.03ms** |

### Scenario Evaluations (42 tests, 11 real-life scenarios)

| Scenario | Result |
|---|---|
| Software engineer — 2-week daily workflow | 5/5 |
| Researcher — evolving hypotheses over 3 months | 3/3 |
| Manager — 3-team context switching | 4/4 |
| Career change — job/location/title contradictions | 6/6 |
| Learning journey — beginner→advanced profile shift | 2/2 |
| Multi-session continuity — survives restart | 3/3 |
| Relevance feedback — retrieval quality improves | 2/2 |
| Entity resolution — k8s/Kubernetes, Postgres/PG | 5/5 |
| Memory consolidation — pattern promotion | 2/2 |
| Working memory — session tracking + flush | 5/5 |
| Performance at scale — 100+ chunks | 5/5 |

### Detail Preservation

| Detail | Stored | Retrieved |
|---|---|---|
| Learning rate `3e-4` | ✅ | ✅ |
| Batch size `256` | ✅ | ✅ |
| GPU model `A100 80GB` | ✅ | ✅ |
| Precision `BF16` | ✅ | ✅ |

Current systems summarize these away. amem preserves them verbatim.

---

## Production Features

| Feature | Details |
|---|---|
| **Persistence** | SQLite WAL — crash-safe, concurrent reads, incremental writes |
| **Authentication** | API key auth via `X-API-Key` or `AMEM_API_KEYS` env var |
| **Rate Limiting** | Token bucket per API key / IP |
| **Multi-User** | `user_id` scoping on every table, GDPR delete |
| **Observability** | JSON structured logging, Prometheus `/metrics`, request IDs |
| **Migrations** | Forward-only SQL schema versioning |
| **Backup** | `VACUUM INTO` for safe SQLite snapshots |

---

## Project Structure

```
amem/
├── amem/
│   ├── episodic/          # Layer 1: TAI vector index, chunking, dedup, importance
│   ├── semantic/          # Layer 2: Knowledge graph, entity resolution, contradictions
│   ├── behavioral/        # Layer 3: 4-dimension user profile
│   ├── working/           # Layer 4: Session scratchpad
│   ├── explicit/          # Layer 5: User-controlled key-value store
│   ├── retrieval/         # Query pipeline: orchestrator, intent analysis
│   ├── embeddings/        # Provider factory: Ollama, OpenAI, local, Anthropic
│   ├── feedback/          # Relevance feedback loop
│   ├── maintenance/       # Memory consolidation engine
│   ├── persistence/       # SQLite WAL backend + migrations
│   └── utils/             # Auth, logging, metrics, tokenizer, rate limiter
├── api/                   # FastAPI (26 endpoints) + OpenAI proxy
├── mcp/                   # MCP server (9 tools, stdio JSON-RPC)
├── cli/                   # Click CLI (15 commands)
├── dashboard/             # Admin web UI (single HTML file)
└── tests/                 # 212 unit tests + 6 integration + 42 scenario evals
```

---

## Contributing

Contributions welcome. Highest impact areas:

- **Embedding models** — test with different models, report retrieval quality
- **Storage backends** — PostgreSQL + pgvector, DuckDB
- **Integrations** — LangChain, LlamaIndex, OpenClaw plugins
- **Benchmarks** — compare against Mem0/Zep on LoCoMo, LongMemEval
- **Language support** — improve extraction for non-English text

---

## License

[MIT](LICENSE)

<p align="center">
  <br/>
  <strong>amem</strong> — The five-layer brain for AI agents.<br/>
  <sub>Query-conditioned · Self-improving · Runs locally · Your data stays yours.</sub>
</p>
