"""Tests for the REST API."""

from __future__ import annotations

from unittest.mock import patch, AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from amem.config import Config
from amem.embeddings.base import EmbeddingProvider


class MockEmbedder(EmbeddingProvider):
    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> np.ndarray:
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [await self.embed(t) for t in texts]

    async def close(self):
        pass


@pytest.fixture
def client():
    """Create test client with mock embedder."""
    from api.app import create_app
    from amem.embeddings.ollama import OllamaEmbedding
    from amem.retrieval.orchestrator import MemoryOrchestrator

    # Patch create_embedder to use MockEmbedder
    with patch("api.app.create_embedder", return_value=MockEmbedder()):
        app = create_app()
        with TestClient(app) as c:
            yield c


class TestAPI:
    def test_ingest(self, client):
        resp = client.post("/ingest", json={
            "text": "Alice works on ML pipelines using Python.",
            "conversation_id": "test-conv",
            "speaker": "user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunks_stored"] > 0

    def test_query(self, client):
        # Ingest first
        client.post("/ingest", json={"text": "Bob leads the infrastructure team."})

        resp = client.post("/query", json={"query": "Who leads infrastructure?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "context_text" in data

    def test_explicit_crud(self, client):
        # Create
        resp = client.post("/explicit", json={
            "key": "role", "value": "engineer", "entry_type": "fact",
        })
        assert resp.status_code == 200

        # List
        resp = client.get("/explicit")
        assert resp.status_code == 200
        assert any(e["key"] == "role" for e in resp.json())

        # Update
        resp = client.put("/explicit/role", json={"value": "senior engineer"})
        assert resp.status_code == 200

        # Delete
        resp = client.delete("/explicit/role")
        assert resp.status_code == 200

        # Verify deleted
        resp = client.get("/explicit")
        assert not any(e["key"] == "role" for e in resp.json())

    def test_explicit_not_found(self, client):
        resp = client.delete("/explicit/nonexistent")
        assert resp.status_code == 404

    def test_session_lifecycle(self, client):
        resp = client.post("/session/start", json={"session_id": "test"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test"

        resp = client.post("/session/add", json={"entry_type": "goal", "content": "Fix bugs"})
        assert resp.status_code == 200

        resp = client.get("/session/context")
        assert resp.status_code == 200
        assert "Fix bugs" in str(resp.json())

        resp = client.post("/session/end")
        assert resp.status_code == 200

    def test_graph_query(self, client):
        client.post("/ingest", json={"text": "Alice works on ML Pipeline."})
        resp = client.post("/graph/query", json={"entities": ["Alice"]})
        assert resp.status_code == 200

    def test_profile(self, client):
        resp = client.get("/profile")
        assert resp.status_code == 200
        assert "priors" in resp.json()

    def test_stats(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert "episodic" in resp.json()

    def test_decay(self, client):
        resp = client.post("/admin/decay")
        assert resp.status_code == 200
