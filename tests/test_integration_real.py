"""Integration tests with real Ollama embeddings.

Validates the CORE PROPOSITION: semantically relevant retrieval.
Requires a running Ollama instance. Skips gracefully if unavailable.
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from amem.config import Config, OllamaConfig
from amem.embeddings.ollama import OllamaEmbedding
from amem.retrieval.orchestrator import MemoryOrchestrator


def _ollama_available() -> bool:
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _get_model() -> str:
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        for m in resp.json().get("models", []):
            name = m.get("name", "").split(":")[0]
            if name in ("nomic-embed-text", "all-minilm", "mxbai-embed-large"):
                return m.get("name", name)
        models = resp.json().get("models", [])
        if models:
            return models[0].get("name", "nomic-embed-text")
    except Exception:
        pass
    return "nomic-embed-text"


OLLAMA_AVAILABLE = _ollama_available()
skip_no_ollama = pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not available")


async def _make_orch() -> tuple:
    """Create an orchestrator with real embeddings inside the async context."""
    model = _get_model()
    config = Config()
    config.ollama.model = model
    embedder = OllamaEmbedding(config.ollama)
    # Probe real dimension
    test_vec = await embedder.embed("dimension probe")
    config.ollama.embedding_dim = len(test_vec)
    # Recreate embedder with correct dim
    orch = MemoryOrchestrator(embedder, config)
    return orch, embedder


@skip_no_ollama
class TestRealSemanticRetrieval:

    @pytest.mark.asyncio
    async def test_semantic_similarity_basic(self):
        """ML query should rank ML content above kitchen content."""
        orch, embedder = await _make_orch()

        await orch.ingest(
            text="Machine learning models require careful hyperparameter tuning. "
                 "Grid search and Bayesian optimization are common approaches.",
            conversation_id="ml-conv",
        )
        await orch.ingest(
            text="The kitchen renovation includes new granite countertops "
                 "and custom cabinets with a farmhouse sink.",
            conversation_id="kitchen-conv",
        )

        ctx = await orch.query("How do you tune ML hyperparameters?", top_k=5)
        assert len(ctx.episodic_chunks) > 0
        top = ctx.episodic_chunks[0]["text"].lower()
        assert "hyperparameter" in top or "learning" in top or "tuning" in top
        await embedder.close()

    @pytest.mark.asyncio
    async def test_semantic_similarity_person_context(self):
        """Query about Alice should rank Alice content highest."""
        orch, embedder = await _make_orch()

        await orch.ingest(
            text="Alice is a senior ML engineer at Google leading recommendation systems.",
            conversation_id="alice",
        )
        await orch.ingest(
            text="Bob manages cloud infrastructure at AWS focusing on Kubernetes.",
            conversation_id="bob",
        )

        ctx = await orch.query("What does Alice work on?", top_k=5)
        assert len(ctx.episodic_chunks) > 0
        top = ctx.episodic_chunks[0]["text"].lower()
        assert "alice" in top or "recommendation" in top
        await embedder.close()

    @pytest.mark.asyncio
    async def test_three_topics_discrimination(self):
        """Three distinct topics — query should retrieve the right one."""
        orch, embedder = await _make_orch()

        await orch.ingest(text="Python 3.12 includes performance improvements and better type hints.", conversation_id="python")
        await orch.ingest(text="Mediterranean diet emphasizes olive oil and fresh vegetables for heart health.", conversation_id="diet")
        await orch.ingest(text="James Webb telescope discovered exoplanets in habitable zones.", conversation_id="space")

        ctx = await orch.query("Tell me about space discoveries and telescopes", top_k=3)
        assert len(ctx.episodic_chunks) > 0
        top_cid = ctx.episodic_chunks[0].get("conversation_id", "")
        assert top_cid == "space"
        await embedder.close()

    @pytest.mark.asyncio
    async def test_similar_texts_have_high_cosine(self):
        """Verify embeddings are semantically meaningful."""
        model = _get_model()
        config = Config()
        config.ollama.model = model
        embedder = OllamaEmbedding(config.ollama)

        a = await embedder.embed("Machine learning is a subset of AI")
        b = await embedder.embed("ML is a branch of artificial intelligence")
        c = await embedder.embed("The recipe needs two cups of flour")

        a, b, c = a / np.linalg.norm(a), b / np.linalg.norm(b), c / np.linalg.norm(c)
        sim_ab = float(a @ b)
        sim_ac = float(a @ c)

        assert sim_ab > sim_ac
        assert sim_ab > 0.5
        await embedder.close()

    @pytest.mark.asyncio
    async def test_full_pipeline_real(self):
        """End-to-end with real embeddings: ingest + explicit + query."""
        orch, embedder = await _make_orch()

        await orch.ingest(
            text="I'm a distributed systems engineer building microservices with Go and Kubernetes.",
            speaker="user",
        )
        orch.explicit.set("role", "distributed systems engineer", entry_type="fact", priority=10)

        ctx = await orch.query("What do I work on?")
        assert len(ctx.episodic_chunks) > 0
        assert len(ctx.explicit_entries) > 0

        text = ctx.to_injection_text(profile=orch.behavioral)
        assert "distributed systems engineer" in text
        await embedder.close()

    @pytest.mark.asyncio
    async def test_embedding_extractor_works(self):
        """Verify the embedding extractor (same model) extracts entities."""
        orch, embedder = await _make_orch()

        result = await orch.ingest(
            text="Alice works at Google using Python and PyTorch for deep learning research.",
            speaker="user",
        )
        assert result["entities_extracted"] > 0
        # Should have extracted at least some known tools
        entities = orch.semantic.get_entities()
        names = [e.get("name", "").lower() for e in entities]
        # At least Python or PyTorch should be detected (they're in the known tools list)
        assert any(n in names for n in ["python", "pytorch", "alice", "google"])
        await embedder.close()
