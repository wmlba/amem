<p align="center">
  <h1 align="center">Associative Memory</h1>
  <p align="center">
    <strong>Memory infrastructure for AI that actually works.</strong>
  </p>
  <p align="center">
    Five-layer associative memory system that replaces lossy blob injection<br/>
    with query-conditioned retrieval, temporal reasoning, and self-improving feedback loops.
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> &bull;
    <a href="#why-this-exists">Why This Exists</a> &bull;
    <a href="#architecture">Architecture</a> &bull;
    <a href="#features">Features</a> &bull;
    <a href="#api-reference">API Reference</a> &bull;
    <a href="#benchmarks">Benchmarks</a>
  </p>
</p>

---

> **The problem:** Every major LLM provider stores memory as lossy text blobs injected wholesale into every context window. It's the equivalent of photocopying your entire diary and paper-clipping it to every letter you write.
>
> **The solution:** An associative memory system modeled after how human brains actually work — episodic, semantic, behavioral, working, and explicit memory layers, each optimized for a different type of knowledge, with query-conditioned retrieval that only injects what's relevant.

---

## The Numbers

| Metric | Value |
|---|---|
| Context waste reduction | **~60% fewer irrelevant tokens injected** |
| Detail preservation | **100% — no lossy summarization** |
| Write latency | **< 50ms** (SQLite WAL, write-through) |
| Query latency | **< 1 second** at 100+ chunks |
| Python codebase | **12,000+ lines** across **72 files** |
| Test suite | **212 unit tests + 45 integration tests + 42 scenario evals** |

---

## Quickstart

### Option 1: With Ollama (recommended)

```bash
# Install Ollama and pull an embedding model
ollama pull nomic-embed-text

# Clone and install
git clone https://github.com/your-org/associative-memory.git
cd associative-memory
pip install -e .

# Start the API server
python -m cli.main serve

# Open the dashboard
open http://localhost:8420/dashboard
```

### Option 2: With OpenAI

```yaml
# config.yaml
embedding:
  provider: openai
  model: text-embedding-3-small
  api_key: sk-...
```

### Option 3: Completely offline

```yaml
# config.yaml — no server needed
embedding:
  provider: local
  model: all-MiniLM-L6-v2
```

```bash
pip install sentence-transformers
python -m cli.main serve
```

### Option 4: Any OpenAI-compatible server

Works with vLLM, llama.cpp, LiteLLM, Together, Anyscale, OpenRouter — anything that speaks `/v1/embeddings`.

```yaml
embedding:
  provider: openai
  model: your-model-name
  base_url: http://localhost:8000/v1
```

---

## Why This Exists

Current LLM memory systems have six fundamental problems:

| Problem | Current Systems | Associative Memory |
|---|---|---|
| **Context Waste** | Inject ALL memories into EVERY query | Query-conditioned — only relevant chunks |
| **Lossy Compression** | Summarize "learning rate 3e-4" → "trains ML models" | Raw chunks preserved verbatim |
| **Staleness** | Nightly batch updates | Write-through in < 50ms |
| **No Temporal Reasoning** | "Works at OCI" and "works at Anthropic" coexist | Contradiction detection + temporal resolution |
| **No Structure** | Flat text blob | 5 queryable layers + knowledge graph |
| **No Learning** | Same results every time | Relevance feedback — improves with use |

---

## Architecture

```
                         ┌─────────────┐
                         │   Query     │
                         └──────┬──────┘
                                │
                    ┌───────────┼───────────┐
                    │    Intent Analysis     │  ← Novel: dynamic scoring weights
                    └───────────┬───────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                       │
    ┌─────┴─────┐       ┌──────┴──────┐       ┌───────┴───────┐
    │ Episodic  │       │  Semantic   │       │   Explicit    │
    │  Store    │       │   Graph     │       │    Store      │
    │           │       │             │       │               │
    │ TAI Index │       │  NetworkX   │       │  Key-Value    │
    │ (vectors) │       │  (entities) │       │  (user CRUD)  │
    └─────┬─────┘       └──────┬──────┘       └───────┬───────┘
          │                     │                       │
          │              ┌──────┴──────┐                │
          │              │ Behavioral  │                │
          │              │  Profile    │                │
          │              └──────┬──────┘                │
          │                     │                       │
          └─────────────────────┼───────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │  Budget Allocation    │  ← Dynamic per-layer token budget
                    └───────────┬───────────┘
                                │
                    ┌───────────┼───────────┐
                    │  Context Assembly     │  ← Behavioral modulation
                    └───────────┬───────────┘
                                │
                         ┌──────┴──────┐
                         │  Injection  │  → 245 tokens instead of 3000
                         └─────────────┘
```

### The Five Memory Layers

| Layer | Analogy | What It Stores | How It Works |
|---|---|---|---|
| **Episodic** | "I remember that conversation" | Raw text chunks with embeddings | Temporal Associative Index — fused vectorized search with time-tiered shards |
| **Semantic** | "I know Alice leads the ML team" | Entities, relations, confidence scores | NetworkX knowledge graph with entity resolution and contradiction detection |
| **Behavioral** | "You prefer concise technical answers" | Communication preferences | 4-dimension EMA profile built from message signals |
| **Working** | "We're debugging the auth flow right now" | Session goals, established facts, open threads | In-session scratchpad, flushed to episodic on session end |
| **Explicit** | "You told me your name is Will" | User-controlled facts, preferences, instructions | Typed key-value store with priority, never decays, always injected |

---

## Features

### Temporal Associative Index (TAI)

A novel vector index designed for memory, not generic similarity search.

- **Time-tiered shards**: Hot (brute-force, sub-ms) → Warm → Cold (archival)
- **Fused scoring**: `similarity × temporal × reinforcement × importance` in a single numpy expression — zero Python loops
- **O(n) top-k** via `argpartition` instead of O(n log n) full sort
- **Batch dedup + novelty** in one matmul — not N separate searches
- **Smart dedup**: Checks distinctive tokens, not just cosine. "Working on Kubernetes" and "Working on React" aren't duplicates even if cosine > 0.95
- **Auto-compaction**: Hot → warm when threshold exceeded, cold chunks auto-evicted

### Entity Resolution

- Fuzzy string matching (typos: "Kuberntes" → "Kubernetes")
- Alias tracking ("k8s", "kube" → "Kubernetes")
- Possessive normalization ("Will's GB10" → "GB10")
- Acronym expansion ("ML" → "Machine Learning")
- Vector-based linking (embedding similarity for semantic matches)
- Entity merging with relation consolidation

### Contradiction Detection

- Direct conflicts: "works at OCI" vs "works at Anthropic" → `newer_wins`
- Temporal supersession: predicate-aware temporal reasoning
- Negation detection: "left OCI" negates "works at OCI"
- Confidence-based resolution when timestamps are tied
- Fact retraction: user-driven corrections

### Intent-Aware Dynamic Scoring

Queries are analyzed for intent before scoring:
- "What's my **current** status?" → boosts temporal weight
- "What's the **exact** config?" → boosts similarity weight
- "Tell me **everything** about X" → flattens all weights
- "Team **Beta** issue" → boosts context-anchor matching

### Relevance Feedback Loop

The system learns which retrievals are actually useful:
- Compares LLM response tokens against retrieved chunk tokens
- Used chunks get reinforced (access_count++)
- Ignored chunks get gently demoted (confidence × 0.95)
- Over hundreds of queries, retrieval quality improves automatically

### Memory Consolidation

Inspired by human memory consolidation during sleep:
- Frequently-mentioned entities promoted to semantic graph
- Low-confidence old chunks evicted from cold tier
- Topic clusters detected from co-retrieval patterns

### Adaptive Decay

Not all facts should decay at the same rate:

| Tier | Example | Decay Rate |
|---|---|---|
| Identity | "My name is Will" | 0.001/day |
| Professional | "I work at Anthropic" | 0.005/day |
| Project | "Using PyTorch for training" | 0.01/day |
| Ephemeral | "I prefer dark mode" | 0.05/day |

---

## API Reference

### REST API (26 endpoints)

```bash
# Ingest text
curl -X POST http://localhost:8420/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice works on ML pipelines using Python.", "speaker": "user"}'

# Query memory
curl -X POST http://localhost:8420/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Alice work on?", "top_k": 5}'

# Remember something explicitly
curl -X POST http://localhost:8420/explicit \
  -H "Content-Type: application/json" \
  -d '{"key": "name", "value": "Will", "entry_type": "fact", "priority": 10}'

# Query the knowledge graph
curl -X POST http://localhost:8420/graph/query \
  -H "Content-Type: application/json" \
  -d '{"entities": ["Alice"], "max_depth": 2}'

# Merge entities
curl -X POST http://localhost:8420/graph/merge \
  -H "Content-Type: application/json" \
  -d '{"name_a": "GB10", "name_b": "Blackwell workstation"}'

# View dashboard
open http://localhost:8420/dashboard
```

### CLI (15 commands)

```bash
amem ingest "Alice works on ML pipelines."
amem query "What does Alice work on?"
amem remember name Will --type fact --priority 10
amem forget name
amem memories
amem graph Alice --depth 2
amem merge GB10 "Blackwell workstation"
amem alias Kubernetes k8s
amem retract Will works_at OCI
amem contradictions
amem profile
amem feedback response_depth 0.9
amem decay
amem status
amem serve --port 8420
```

### MCP Server (9 tools)

Add to Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "associative-memory": {
      "command": "python3",
      "args": ["-m", "mcp.server"],
      "cwd": "/path/to/associative-memory"
    }
  }
}
```

Tools: `memory_ingest`, `memory_query`, `memory_remember`, `memory_forget`, `memory_list`, `memory_graph`, `memory_retract`, `memory_merge_entities`, `memory_stats`

### OpenAI-Compatible Proxy

Drop-in replacement — just change the base URL:

```bash
# Start the proxy
python -m api.openai_compat --target https://api.openai.com/v1 --port 8421

# Use with any OpenAI client — memory is injected automatically
export OPENAI_BASE_URL=http://localhost:8421/v1
```

Every `/v1/chat/completions` request gets memory context injected before forwarding, and the conversation is ingested into memory after the response.

---

## Embedding Providers

One config field. Any model. Same model powers all five uses.

| Provider | Config | Use Case |
|---|---|---|
| **Ollama** | `provider: ollama` | Local development, free |
| **OpenAI** | `provider: openai` | Highest quality embeddings |
| **Anthropic/Voyage** | `provider: anthropic` | Claude ecosystem |
| **Local** | `provider: local` | Completely offline, no server |
| **Any OpenAI-compat** | `provider: openai, base_url: ...` | vLLM, llama.cpp, LiteLLM, Together, etc. |
| **Auto** | `provider: auto` | Tries Ollama → local → OpenAI env key |

---

## Benchmarks

Tested against live Ollama with `nomic-embed-text` (768 dimensions):

### Semantic Retrieval Accuracy

| Scenario | Test | Result |
|---|---|---|
| 4 distinct topics (ML, kitchen, finance, yoga) | Correct topic ranked #1 | **4/4 pass** |
| Person-specific queries (Alice vs Bob) | Correct person ranked #1 | **pass** |
| Detail preservation (learning rate, batch size, GPU) | Exact values retrievable | **4/4 pass** |

### Real-Life Scenario Evaluations (42 tests, 11 scenarios)

| Scenario | Tests | Result |
|---|---|---|
| Software engineer — 2-week daily workflow | 5/5 | Pass |
| Researcher — evolving hypotheses over 3 months | 3/3 | Pass |
| Manager — 3-team context switching | 4/4 | Pass |
| Career change — job/location/title contradictions | 6/6 | Pass |
| Learning journey — beginner to advanced profile shift | 2/2 | Pass |
| Multi-session continuity — survives restart | 3/3 | Pass |
| Relevance feedback — retrieval quality improves | 2/2 | Pass |
| Entity resolution — k8s/Kubernetes, Postgres/PG | 5/5 | Pass |
| Memory consolidation — pattern promotion | 2/2 | Pass |
| Working memory — session tracking + flush | 5/5 | Pass |
| Performance at scale — 100+ chunks | 5/5 | Pass |

### Performance

| Operation | Latency |
|---|---|
| Ingest (embed + store + extract) | < 2s per message |
| Query (retrieve + rank + assemble) | < 1s |
| Explicit memory set/get | < 1ms |
| SQLite persistence | Write-through, crash-safe |
| Database size | ~116 KB for 11 chunks |

---

## Persistence

SQLite with WAL mode. Crash-safe, concurrent reads, incremental writes.

```
data/
├── amem.db           ← Single SQLite database (all 5 layers)
├── semantic/
│   ├── resolver.json  ← Entity resolver state
│   └── contradictions.json
└── episodic/
    └── tai/           ← Temporal Associative Index shards
```

Full GDPR compliance: `DELETE /user/{user_id}` wipes all data for a user.

---

## Production Features

- **Authentication**: API key auth via `X-API-Key` header or `AMEM_API_KEYS` env var
- **Rate Limiting**: Token bucket algorithm per API key / IP
- **Structured Logging**: JSON-formatted logs with request IDs
- **Prometheus Metrics**: `/metrics` endpoint for monitoring
- **Schema Migrations**: Forward-only SQL migrations on startup
- **Multi-User Isolation**: `user_id` scoping on every table
- **Backup**: `VACUUM INTO` for safe SQLite snapshots
- **Admin Dashboard**: Live web UI at `/dashboard`

---

## Project Structure

```
associative_memory/
├── amem/                          # Core library
│   ├── episodic/                  # Layer 1: Temporal Associative Index
│   │   ├── temporal_index.py      # Novel TAI with time-tiered shards
│   │   ├── vector_index.py        # Legacy index (backward compat)
│   │   ├── store.py               # Ingest pipeline with smart dedup
│   │   ├── importance.py          # Chunk importance scoring
│   │   ├── smart_dedup.py         # Distinctive-token deduplication
│   │   └── chunker.py             # Sentence-boundary chunking
│   ├── semantic/                  # Layer 2: Knowledge Graph
│   │   ├── graph.py               # NetworkX graph with entity resolution
│   │   ├── resolver.py            # Fuzzy/alias/vector entity linking
│   │   ├── contradictions.py      # Temporal contradiction detection
│   │   ├── embedding_extractor.py # Entity extraction via same embedding model
│   │   ├── adaptive_decay.py      # 4-tier decay rates
│   │   └── temporal.py            # Temporal expression parsing
│   ├── behavioral/                # Layer 3: User Profile
│   │   └── profile.py             # 4-dimension EMA behavioral tracking
│   ├── working/                   # Layer 4: Session Scratchpad
│   │   └── session.py             # Goals, facts, threads
│   ├── explicit/                  # Layer 5: User-Controlled Memory
│   │   └── store.py               # Typed CRUD with priority
│   ├── retrieval/                 # Query Pipeline
│   │   ├── orchestrator.py        # Cross-layer retrieval + budget allocation
│   │   └── intent.py              # Intent-aware dynamic scoring
│   ├── embeddings/                # Provider Abstraction
│   │   ├── factory.py             # Universal provider factory
│   │   ├── ollama.py              # Ollama (with circuit breaker)
│   │   ├── openai_embed.py        # OpenAI-compatible endpoints
│   │   ├── anthropic_embed.py     # Voyage models
│   │   └── local_embed.py         # sentence-transformers (offline)
│   ├── feedback/                  # Learning Loop
│   │   └── relevance.py           # Reinforcement from LLM usage
│   ├── maintenance/               # Background Jobs
│   │   └── consolidation.py       # Memory consolidation engine
│   ├── persistence/               # Storage
│   │   ├── sqlite.py              # SQLite WAL backend
│   │   └── migrations.py          # Schema versioning
│   └── utils/                     # Utilities
│       ├── tokenizer.py           # tiktoken integration
│       ├── auth.py                # API key authentication
│       ├── logging.py             # Structured JSON logging + metrics
│       └── ratelimit.py           # Token bucket rate limiter
├── api/                           # REST API
│   ├── app.py                     # FastAPI (26 endpoints)
│   ├── openai_compat.py           # OpenAI-compatible proxy
│   └── models.py                  # Pydantic schemas
├── mcp/                           # MCP Server
│   └── server.py                  # 9 tools over stdio JSON-RPC
├── cli/                           # CLI
│   └── main.py                    # 15 Click commands
├── dashboard/                     # Admin UI
│   └── index.html                 # Single-file web dashboard
├── tests/                         # Test Suite
│   ├── test_tai.py                # Temporal Associative Index
│   ├── test_vector_index.py       # Legacy vector index
│   ├── test_resolver.py           # Entity resolution
│   ├── test_contradictions.py     # Contradiction detection
│   ├── test_budget.py             # Dynamic budget allocation
│   ├── test_feedback_and_consolidation.py
│   ├── test_sqlite.py             # Persistence
│   ├── test_providers.py          # Embedding providers
│   ├── test_mcp_and_proxy.py      # MCP + OpenAI proxy
│   ├── test_integration.py        # End-to-end with mock embeddings
│   └── test_integration_real.py   # End-to-end with live Ollama
├── config.yaml                    # Configuration
└── pyproject.toml                 # Dependencies
```

---

## What Makes This Different

This isn't another RAG framework. It's a fundamentally different approach to LLM memory.

**Existing systems** (ChatGPT Memory, Claude Memory, LangChain, LlamaIndex):
- Summarize conversations into text blobs
- Inject everything into every context window
- No temporal reasoning — old and new facts coexist with equal weight
- No learning — retrieval quality never improves
- Vendor-locked, cloud-only

**This system**:
- Five specialized memory layers modeled after human cognition
- Query-conditioned retrieval — only relevant context injected
- Temporal Associative Index with time-tiered shards and fused scoring
- Contradiction detection with automatic resolution
- Relevance feedback loop — the system learns what's useful
- Memory consolidation — patterns crystallize into knowledge
- Runs locally, works with any embedding model, your data stays yours

---

## Contributing

Contributions welcome. The areas with highest impact:

1. **Better embedding models** — test with different models and report quality
2. **New extraction patterns** — improve the embedding-based entity extractor
3. **Storage backends** — PostgreSQL + pgvector, DuckDB
4. **Benchmarks** — compare against blob-injection baselines on standard datasets
5. **Language support** — test and improve extraction for non-English text

---

## License

MIT

---

<p align="center">
  <strong>Memory infrastructure for AI that actually works.</strong><br/>
  <em>Five layers. Query-conditioned. Self-improving. Runs locally.</em>
</p>
