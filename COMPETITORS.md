# Competitive Analysis: amem vs The Top 5

## The Landscape (April 2026)

| Framework | Stars | Core Approach | Primary Strength |
|---|---|---|---|
| **Mem0** | 48K | Hybrid vector + graph + KV | Largest ecosystem, production polish |
| **Zep/Graphiti** | 20K | Bi-temporal knowledge graph (Neo4j) | Best temporal validity windows |
| **Letta (MemGPT)** | 15K | LLM-as-OS, agent self-manages memory | Autonomous memory management |
| **Cognee** | 12K | ECL pipeline + self-improving graph | Document ingestion, 38+ sources |
| **SuperLocalMemory** | 2K | Math-first, 7 retrieval channels | Zero-LLM mode, formal guarantees |
| **amem** | New | 5-layer associative + TAI + intent scoring | Brain-like architecture, single model |

---

## Deep Comparison

### 1. Mem0

**Architecture:** Dual-store — Qdrant/FAISS vectors + optional Neo4j graph. LLM decides ADD/UPDATE/DELETE/NOOP for every fact. ThreadPoolExecutor runs vector + graph ops concurrently. 3 scopes: user, session, agent.

**Strengths:**
- Largest community, most integrations, SOC 2 + HIPAA cloud
- 26% accuracy uplift over OpenAI memory on LoCoMo
- 91% faster, ~90% fewer tokens vs full-context
- Graph mode adds multi-hop reasoning

**Weaknesses:**
- Every memory operation requires an LLM call (cost + latency)
- No principled decay — LRU policies only, no forgetting curve
- Contradiction detection is LLM-dependent, no mathematical guarantees
- No behavioral profiling — stores preferences as flat facts
- No working memory tier
- No intent-aware scoring — fixed retrieval weights
- No relevance feedback — retrieval quality never improves
- Benchmark claims contested by Zep founders
- Azure/HuggingFace integration bugs, Windows installation friction

**What amem does that Mem0 doesn't:**
- 5 layers vs 3 scopes — behavioral + working memory are entirely missing from Mem0
- Single embedding model for extraction (no LLM calls per memory operation)
- Mathematical temporal decay (exponential with adaptive tiers, not LRU)
- Intent-aware dynamic scoring (query analysis adjusts retrieval weights)
- Relevance feedback loop (chunks that get used are reinforced)
- Smart deduplication (distinctive tokens, not just cosine)
- Memory consolidation engine (patterns promote to semantic graph)

---

### 2. Zep / Graphiti

**Architecture:** Neo4j temporal knowledge graph. Episodes ingested → entities/relations extracted by LLM → bi-temporal edges (created_at, valid_at, invalid_at). Search fuses cosine + BM25 + BFS. P95 ~300ms.

**Strengths:**
- Best temporal reasoning — bi-temporal validity windows are genuinely novel
- Outperforms Mem0 on LongMemEval by 15 points
- Real-time incremental ingestion
- Three-pronged search (semantic + BM25 + graph BFS)
- Strong academic foundation (ICLR-level paper)

**Weaknesses:**
- Hard Neo4j dependency — no embedded/lightweight option
- Every entity resolution requires an LLM call — expensive at scale
- **No temporal decay** — open GitHub issue #1300, never implemented
- Docker image 12 versions behind, deployment confusion
- No SDK client library
- Zep Cloud went closed-source
- No behavioral profiling
- No working memory
- No relevance feedback
- Python-only

**What amem does that Zep doesn't:**
- No Neo4j dependency — SQLite + in-memory NetworkX
- Temporal decay IS implemented (4 adaptive tiers)
- Entity extraction uses embeddings, not LLM calls (10-50x cheaper)
- Behavioral profiling (4-dimension EMA tracking)
- Working memory (session scratchpad)
- Explicit user memory store (typed CRUD with priority)
- Relevance feedback loop
- Intent-aware scoring
- Memory consolidation
- Runs anywhere — no graph database server required

---

### 3. Letta (MemGPT)

**Architecture:** LLM-as-Operating-System. 3 tiers: Core Memory (RAM — always in context), Recall Memory (disk cache — searchable history), Archival Memory (cold storage). The LLM self-manages via function calls: core_memory_append, core_memory_replace, archival_memory_insert, etc.

**Strengths:**
- Unique self-managing design — agent decides what to remember
- 74.0% on LoCoMo with GPT-4o mini
- Memory blocks with labels break context into purposeful units
- Multi-agent via Conversations API

**Weaknesses:**
- Critically model-dependent — degrades badly with smaller/quantized models
- Every memory operation costs inference tokens
- If the model fails to save something, it's **gone permanently**
- No graph — relationship reasoning is pure LLM hallucination
- No temporal decay
- No contradiction detection (relies on LLM noticing conflicts)
- No entity resolution system
- No relevance feedback
- Past security vulnerability (CVE-2024-39025)
- Only works for conversational data

**What amem does that Letta doesn't:**
- Works with ANY model quality — doesn't require GPT-4 class reasoning
- Never loses data — every ingest is persisted regardless of LLM behavior
- Knowledge graph with entity resolution + contradiction detection
- Temporal decay (adaptive, 4 tiers)
- Behavioral profiling
- Relevance feedback
- Intent-aware scoring
- Single embedding model (no expensive LLM calls for memory management)
- Document + conversation + structured data ingestion

---

### 4. Cognee

**Architecture:** ECL pipeline (Extract, Cognify, Load). 6-stage ingestion → knowledge graph + vectors. "Memify" phase self-improves graph by pruning stale nodes, strengthening frequent connections, reweighting. 14 retrieval modes. LanceDB vectors + Neo4j/Memgraph graph.

**Strengths:**
- Self-improving graph (Memify) — genuinely novel
- 0.93 on HotPotQA (near human-level)
- 38+ data source connectors — broadest ingestion
- 14 retrieval modes
- Temporal cognification with event-based timestamps
- 70+ companies in production

**Weaknesses:**
- Python SDK only — blocks TypeScript/Go teams
- Weak at conversation personalization — designed for documents, not users
- No behavioral profiling
- No working memory
- No explicit user memory store
- No intent-aware scoring
- Contradiction detection exists but is implicit (graph integrity rules)
- Scaling to terabyte datasets unproven
- Cloud platform immature vs Mem0/Zep

**What amem does that Cognee doesn't:**
- Conversation-native — designed for user interactions, not just documents
- Behavioral profiling (adapts to communication style)
- Working memory (session-level scratchpad)
- Explicit user memory (typed CRUD, never decays, always injected)
- Intent-aware scoring (query analysis adjusts weights)
- Single embedding model (Cognee uses LLM calls for extraction)
- Smart dedup with distinctive token analysis
- OpenAI-compatible proxy for drop-in integration

---

### 5. SuperLocalMemory

**Architecture:** Math-first, CPU-only, single SQLite file. 3 modes: A (zero-LLM), B (local LLM), C (cloud LLM). 7 retrieval channels: semantic (Fisher-information-weighted), BM25, entity graph, temporal, spreading activation, consolidation, Hopfield associative. Fusion via weighted reciprocal rank. Contradiction detection via cellular sheaf cohomology.

**Strengths:**
- True zero-LLM mode (74.8% LoCoMo with no cloud)
- Mathematical guarantees for contradiction detection
- Ebbinghaus adaptive forgetting
- 7 retrieval channels — most diverse
- Works with 17+ AI tools
- Single SQLite file — zero infrastructure

**Weaknesses:**
- Research-stage — small community, limited production testing
- Mathematical complexity creates contribution barrier
- No managed cloud offering
- No multi-user scoping or access control
- No behavioral profiling
- No explicit user memory store
- No intent-aware scoring
- No OpenAI proxy or LangChain integration
- Claims unvalidated by independent researchers
- Documentation limited to papers + README

**What amem does that SuperLocalMemory doesn't:**
- Production-ready with SQLite WAL, auth, rate limiting, metrics
- Multi-user isolation (user_id scoping on every table)
- Behavioral profiling (4-dimension adaptation)
- Explicit user memory (typed CRUD with priority)
- Intent-aware dynamic scoring
- OpenAI-compatible proxy, MCP server, REST API, CLI, dashboard
- LangChain/LlamaIndex integration path
- Any embedding provider (Ollama, OpenAI, local, Anthropic)
- Simpler architecture that's auditable and extensible

---

## Feature Matrix: Do We Cover ALL Their Gaps?

| Feature | Mem0 | Zep | Letta | Cognee | SLM | **amem** |
|---|---|---|---|---|---|---|
| **Episodic memory (raw chunks)** | Partial | No | Partial | Yes | Yes | **Yes** |
| **Semantic graph** | Optional | Yes | No | Yes | Yes | **Yes** |
| **Behavioral profiling** | No | No | Partial | No | No | **Yes** |
| **Working memory** | No | No | Yes | No | No | **Yes** |
| **Explicit user memory** | Partial | Partial | Yes | No | No | **Yes** |
| **Temporal decay** | LRU only | No (#1300) | No | Partial | Yes | **Yes (4 tiers)** |
| **Contradiction detection** | LLM-based | LLM-based | LLM-implicit | Implicit | Math (sheaf) | **Yes (rule + temporal)** |
| **Entity resolution** | Embedding | 3-tier+LLM | No | Yes | Yes | **Yes (fuzzy+alias+embed)** |
| **Query-conditioned retrieval** | Yes | Yes | Partial | Yes | Yes | **Yes** |
| **Intent-aware scoring** | No | No | No | No | No | **Yes (novel)** |
| **Relevance feedback** | No | No | No | Yes | No | **Yes** |
| **Smart dedup** | No | No | No | No | No | **Yes (novel)** |
| **Memory consolidation** | No | No | No | Yes (Memify) | Yes | **Yes** |
| **Sub-graphs** | Via graph | Natural | No | Natural | No | **Planned v2** |
| **Single model (no LLM calls)** | No (LLM) | No (LLM) | No (LLM) | No (LLM) | Yes (Mode A) | **Yes** |
| **OpenAI proxy** | No | No | No | No | No | **Yes** |
| **MCP server** | Yes | No | No | No | Yes | **Yes** |
| **Admin dashboard** | No | No | Yes | No | No | **Yes** |
| **Multi-user isolation** | Yes | Partial | Yes | No | No | **Yes** |
| **Offline capable** | No | No | No | No | Yes | **Yes** |

### The count

- Features where **only amem** has it: **Intent-aware scoring, Smart dedup, OpenAI proxy**
- Features where amem **uniquely combines**: 5-layer architecture (nobody else has all 5 in one system)
- Features where **amem is the only one without LLM dependency**: Single embedding model for extraction (only SLM Mode A also does this, but with different approach)

### Gaps amem still has

1. **Bi-temporal validity windows** (Zep has this, we don't — our contradiction detection handles the same problem differently but less elegantly)
2. **38+ data source connectors** (Cognee's ingestion breadth — we only do text)
3. **Mathematical contradiction guarantees** (SLM's sheaf cohomology — our rule-based approach works but lacks formal proof)
4. **Self-improving graph** (Cognee's Memify — our consolidation engine is similar but less mature)
5. **Agent self-management** (Letta's LLM-as-OS — we're transparent, not agent-driven)

### Honest assessment

amem covers **more features than any single competitor** but is **less mature than Mem0/Zep in production polish** and **less theoretically grounded than SuperLocalMemory**. The unique value is the combination: five layers + intent-aware scoring + single-model extraction + relevance feedback, all in one system that runs locally without LLM calls for memory operations.

No competitor has all five memory layers. No competitor has intent-aware dynamic scoring. No competitor uses a single embedding model for extraction, dedup, resolution, and retrieval. That's the moat.
