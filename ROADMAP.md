# Roadmap: From Prototype to Platform

## Naming

### Why not "Engram"

"Engram" is saturated. DeepSeek, multiple PyPI packages (engram-core, engram-cloud, engram-dev), npm (engram-sdk), and even an OpenClaw plugin all use it. "Mnemo," "Synapse," and "Cortex" are similarly taken.

### Name candidates (all available on PyPI + npm as of April 2026)

| Name | Meaning | CLI feel | Why it works |
|---|---|---|---|
| **amem** | Associative Memory (our module name) | `amem query` | Short, already our internal namespace, sounds like "I am" + "memory" |
| **stratamem** | Strata (layers) + memory | `strata query` | Directly communicates multi-layer architecture, geological metaphor |
| **pentamem** | Penta (five) + memory | `pentamem query` | References the five layers explicitly, scientific feel |
| **cogmeld** | Cognition + meld | `cogmeld query` | Cognition fused together, unique |
| **neuromeld** | Neural + meld | `neuromeld query` | Neural connections melded, evocative |

**Recommendation: `amem`**

Reasons:
- Already our module name (`from amem import ...`) — zero migration cost
- 4 letters, one syllable to say ("ay-mem"), easy to type
- `pip install amem`, `amem serve`, `amem query` — clean
- Carries meaning: **A**ssociative **Mem**ory
- No competition on PyPI or npm
- Domain options: amem.dev, amem.ai, getamem.com

---

## Competitive Landscape (April 2026)

| Framework | Stars | Approach | What we do differently |
|---|---|---|---|
| **Mem0** | 48K | Hybrid store (vectors + graph + KV), 3 scopes | We have 5 layers not 3, intent-aware scoring, relevance feedback |
| **Zep/Graphiti** | ~12K | Temporal knowledge graph, validity windows | We have temporal + episodic + behavioral + working, not just graph |
| **Letta/MemGPT** | ~15K | OS-like paging (RAM/disk), agent-controlled | We don't require agent cooperation, works transparently |
| **SuperLocalMemory** | ~2K | Bio-inspired, Hopfield networks, 7 channels | Academic, heavy, complex. We're simpler and more practical |
| **Memori** | ~1K | SQL-native, production-focused | No graph, no behavioral profile, no contradiction detection |
| **MemOS** | ~3K | Skill memory, task reuse | Different focus (skills vs personal memory) |

**Our moat:** The five-layer separation with the Temporal Associative Index, intent-aware scoring, and the single-model architecture. Nobody else fuses all five memory types into one query-conditioned pipeline.

---

## Architecture Evolution: Rich Graph Properties

### Current Graph (v1)

Nodes: `{name, type, mentions, timestamps}`
Edges: `{predicate, confidence, timestamps, status}`
Structure: Flat — all entities and relations in one NetworkX graph.

### Target Graph (v2)

#### Rich Node Properties

```python
@dataclass
class EntityNode:
    # Identity
    canonical_id: str        # Stable UUID across renames
    name: str
    entity_type: str         # person, tool, org, project, concept, location

    # Rich schemaless properties (vary by entity type)
    properties: dict         # person: {"role": "CTO", "expertise": ["ML", "Go"]}
                             # tool: {"version": "3.12", "category": "language"}
                             # org: {"size": "startup", "industry": "fintech"}

    # Importance signals
    mention_count: int
    access_count: int        # Times retrieved in queries
    centrality: float        # Graph centrality score (PageRank)

    # Embedding (for resolution)
    embedding: np.ndarray    # Name + context embedded

    # Provenance
    source_chunks: list[str] # Which episodic chunks mention this
    confidence: float        # Extraction confidence
    extraction_method: str   # "embedding", "heuristic", "user_explicit"
```

#### Rich Edge Properties

```python
@dataclass
class RelationEdge:
    predicate: str

    # Rich properties
    properties: dict         # {"role": "senior", "department": "engineering"}

    # Temporal lifecycle
    valid_from: datetime     # When this relation became true
    valid_until: datetime    # When it stopped (null = still active)
    status: FactStatus       # active, superseded, retracted

    # Strength
    confidence: float
    weight: float            # Derived: confidence × recency × mentions
    mention_count: int

    # Provenance
    source_chunks: list[str]
    extraction_method: str
```

#### Sub-Graphs

```python
@dataclass
class SubGraph:
    id: str
    name: str                # "Project Phoenix", "Team Alpha", "Q4 2024"
    description: str
    subgraph_type: str       # "project", "team", "topic", "time_period"

    # Membership
    entity_ids: set[str]
    relation_ids: set[str]

    # Hierarchy (sub-graphs can nest)
    parent_id: str | None    # Team → Company
    children_ids: list[str]  # Team → [Project A, Project B]

    # Metadata
    created: datetime
    active: bool
    summary_embedding: np.ndarray  # For sub-graph-level retrieval

    # Auto-detection
    auto_detected: bool      # True if created by consolidation engine
    detection_signal: str    # "co-retrieval_cluster" or "entity_overlap" or "user_created"
```

#### How sub-graphs change the query pipeline

```
Query: "What's happening on Project Phoenix?"
  │
  ├─ 1. Intent analysis → detects "Project Phoenix" as sub-graph reference
  ├─ 2. Resolve sub-graph: find SubGraph(name="Project Phoenix")
  ├─ 3. Scope episodic search to chunks tagged with this sub-graph
  ├─ 4. Scope graph traversal to sub-graph's entities/relations
  ├─ 5. If insufficient results, expand to parent sub-graph
  └─ 6. Return context scoped to project, not polluted by other projects
```

---

## Implementation Roadmap

### Phase 1: Foundation Polish (Weeks 1-4)

| Week | Task | Deliverable |
|---|---|---|
| 1 | PyPI packaging + `pip install amem` | Published package, tested on Python 3.9-3.12 |
| 1 | Docker image + docker-compose | `docker run amem` with Ollama included |
| 2 | Rich node/edge properties | `EntityNode.properties` dict on every node, `RelationEdge.properties` on every edge |
| 2 | Property-aware extraction | Embedding extractor fills properties (role, expertise, version) |
| 3 | Sub-graph data structure | `SubGraph` class, SQLite table, CRUD API endpoints |
| 3 | Sub-graph auto-detection | Consolidation engine creates sub-graphs from co-retrieval clusters |
| 4 | Sub-graph-scoped queries | Query pipeline scopes search when sub-graph is referenced |
| 4 | 5-minute quickstart guide + 30-second demo GIF | README, tutorial, video |

### Phase 2: Integration Ecosystem (Weeks 5-10)

| Week | Task | Deliverable |
|---|---|---|
| 5 | LangChain memory backend adapter | `from amem.integrations import LangChainMemory` |
| 5 | LlamaIndex retriever adapter | `from amem.integrations import LlamaIndexRetriever` |
| 6 | OpenClaw plugin | Published npm package `@amem/openclaw` |
| 6 | Claude MCP server polished + directory listing | Listed in MCP server directory |
| 7-8 | Python SDK + TypeScript SDK | `pip install amem-sdk`, `npm install @amem/sdk` |
| 9 | Benchmark suite v1 | precision@k, recall@k, latency vs Mem0/Zep/blob-injection |
| 10 | Blog posts: "6 problems with LLM memory" + benchmark results | Published on HN, r/MachineLearning, dev.to |

### Phase 3: Production Platform (Weeks 11-18)

| Week | Task | Deliverable |
|---|---|---|
| 11-12 | PostgreSQL + pgvector backend | Multi-instance deployment, horizontal scaling |
| 13 | Multi-tenant API | Per-user quotas, usage tracking, billing hooks |
| 14 | Admin dashboard v3 | User management, sub-graph explorer, memory diff viewer |
| 15 | Plugin system | Custom extractors, custom decay functions, custom scorers |
| 16 | Webhook + SSE events | Real-time notifications: contradiction detected, entity merged |
| 17 | Graph analytics | PageRank centrality, community detection, sub-graph recommendations |
| 18 | Open memory format spec | Export/import standard for cross-platform portability |

### Phase 4: Ecosystem & Cloud (Weeks 19-30)

| Week | Task | Deliverable |
|---|---|---|
| 19-22 | Hosted cloud offering | amem.dev — managed service, free tier |
| 23-24 | Memory sharing (cross-user sub-graphs) | Shared team knowledge bases |
| 25-26 | Domain-specific memory packs | Pre-built sub-graphs: medical, legal, engineering, finance |
| 27-28 | Agent-to-agent memory sharing | Trust-based knowledge exchange between agents |
| 29-30 | Memory marketplace | Community-contributed extractors, decay models, sub-graph templates |

---

## Marketing Strategy

### Positioning

**Category:** AI Memory Infrastructure
**Tagline:** "The five-layer brain for AI agents"
**One-liner for developers:** "amem gives your LLM episodic, semantic, behavioral, working, and explicit memory — query-conditioned, self-improving, runs locally."

### Against each competitor

| Competitor | Our pitch |
|---|---|
| **Mem0** | "Mem0 has 3 scopes. amem has 5 layers. Your LLM's memory should work like a brain, not a database." |
| **Zep** | "Zep stores temporal facts. amem stores facts, episodes, behavior, session state, and user instructions — with intent-aware retrieval." |
| **Letta** | "Letta requires agent cooperation. amem works transparently — plug it in, memory just works." |
| **Raw RAG** | "RAG retrieves chunks. amem retrieves understanding — the right chunks, at the right time, with the right context." |

### Launch sequence

1. **Week 1:** Ship `pip install amem` + Docker image + README with 30-second GIF
2. **Week 2:** Hacker News post: "Show HN: Five-layer associative memory for LLMs (open source)"
3. **Week 3:** Blog: "We tested 6 LLM memory systems on 11 real-life scenarios. Here's what happened."
4. **Week 4:** r/MachineLearning, r/LocalLLaMA, Twitter/X thread
5. **Week 6:** Benchmark results published: amem vs Mem0 vs Zep vs blob-injection
6. **Week 8:** LangChain + LlamaIndex integration PRs merged
7. **Week 10:** OpenClaw plugin published
8. **Week 12:** "Introducing amem Cloud" (free tier)

### Content calendar (first 90 days)

| Week | Content | Channel |
|---|---|---|
| 1 | "amem: Memory infrastructure for AI that actually works" | GitHub README, HN |
| 2 | "The 6 problems with LLM memory (and how we solved them)" | Blog, dev.to |
| 3 | "How we built a vector index designed for memory, not search" | Blog (TAI deep dive) |
| 4 | "Intent-aware retrieval: why fixed scoring is wrong" | Blog, Twitter thread |
| 6 | Benchmark: "amem vs Mem0 vs Zep on 42 real-world scenarios" | Blog, r/MachineLearning |
| 8 | "Add memory to your LangChain agent in 3 lines" | Tutorial, LangChain docs PR |
| 10 | "Memory that forgets: adaptive decay for AI agents" | Blog (academic crossover) |
| 12 | "Introducing sub-graphs: how to organize 10,000 memories" | Blog, demo video |

---

## Success Metrics

### Phase 1 (Month 1)
- 500+ GitHub stars
- 100+ PyPI installs/week
- 10+ community issues/PRs

### Phase 2 (Month 3)
- 2,000+ GitHub stars
- 500+ PyPI installs/week
- 3+ integration adapters merged
- Benchmark results cited by 2+ blog posts

### Phase 3 (Month 6)
- 5,000+ GitHub stars
- 2,000+ PyPI installs/week
- 50+ production users
- 1+ enterprise pilot

### Phase 4 (Month 12)
- 15,000+ GitHub stars
- amem Cloud: 500+ free tier users
- Listed in "Top AI Memory Frameworks" articles
- Conference talk at NeurIPS/ICML workshop or AI Engineer Summit
