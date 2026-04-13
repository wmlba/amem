"""Tests for MCP server and OpenAI-compatible proxy."""

from __future__ import annotations

import json
from unittest.mock import patch, AsyncMock

import numpy as np
import pytest

from amem.config import Config
from amem.embeddings.base import EmbeddingProvider


class MockEmbedder(EmbeddingProvider):
    def __init__(self, dim=64):
        self._dim = dim
    @property
    def dimension(self):
        return self._dim
    async def embed(self, text):
        seed = hash(text) % (2**31)
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)
    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]
    async def close(self):
        pass


# ─── MCP Server Tests ──────────────────────────────────────────────

class TestMCPServer:
    @pytest.fixture
    def server(self):
        from mcp.server import MCPServer
        s = MCPServer()
        return s

    @pytest.mark.asyncio
    async def test_handle_initialize(self, server):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

        with patch("mcp.server.OllamaEmbedding", return_value=MockEmbedder()):
            resp = await server.handle_message(msg)

        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "associative-memory"

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server):
        # Initialize first
        with patch("mcp.server.OllamaEmbedding", return_value=MockEmbedder()):
            await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})

        msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = await server.handle_message(msg)

        assert "result" in resp
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "memory_ingest" in tool_names
        assert "memory_query" in tool_names
        assert "memory_remember" in tool_names
        assert "memory_forget" in tool_names
        assert "memory_stats" in tool_names
        assert len(tools) == 9

    @pytest.mark.asyncio
    async def test_tool_memory_ingest(self, server):
        with patch("mcp.server.OllamaEmbedding", return_value=MockEmbedder()):
            await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})

        msg = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "memory_ingest",
                "arguments": {"text": "Alice works on ML pipelines.", "speaker": "user"},
            },
        }
        resp = await server.handle_message(msg)
        assert "result" in resp
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["chunks_stored"] > 0

    @pytest.mark.asyncio
    async def test_tool_memory_remember_and_list(self, server):
        with patch("mcp.server.OllamaEmbedding", return_value=MockEmbedder()):
            await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})

        # Remember
        msg = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "memory_remember",
                "arguments": {"key": "name", "value": "Alice"},
            },
        }
        resp = await server.handle_message(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["remembered"] is True

        # List
        msg = {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "memory_list", "arguments": {}},
        }
        resp = await server.handle_message(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        assert content["count"] == 1

    @pytest.mark.asyncio
    async def test_tool_memory_query(self, server):
        with patch("mcp.server.OllamaEmbedding", return_value=MockEmbedder()):
            await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})

        # Ingest first
        await server.handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "memory_ingest", "arguments": {"text": "Test data for query."}},
        })

        # Query
        msg = {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "memory_query", "arguments": {"query": "test"}},
        }
        resp = await server.handle_message(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "context_text" in content

    @pytest.mark.asyncio
    async def test_tool_memory_stats(self, server):
        with patch("mcp.server.OllamaEmbedding", return_value=MockEmbedder()):
            await server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})

        msg = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "memory_stats", "arguments": {}},
        }
        resp = await server.handle_message(msg)
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "episodic" in content

    @pytest.mark.asyncio
    async def test_unknown_method(self, server):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "nonexistent", "params": {}}
        resp = await server.handle_message(msg)
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_notification_no_response(self, server):
        msg = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        resp = await server.handle_message(msg)
        assert resp is None

    @pytest.mark.asyncio
    async def test_ping(self, server):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
        resp = await server.handle_message(msg)
        assert resp["result"] == {}


# ─── OpenAI Proxy Tests ────────────────────────────────────────────

class TestOpenAIProxy:
    @pytest.fixture
    def client(self):
        from api.openai_compat import create_proxy_app
        from fastapi.testclient import TestClient

        with patch("api.openai_compat.OllamaEmbedding", return_value=MockEmbedder()):
            app = create_proxy_app(target_url="http://localhost:11434/v1")
            with TestClient(app) as c:
                yield c

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "memory_stats" in data


# ─── Auth Tests ────────────────────────────────────────────────────

class TestAuth:
    def test_auth_disabled_by_default(self):
        from amem.utils.auth import AuthManager
        auth = AuthManager()
        assert not auth.enabled
        assert auth.validate(None) is True

    def test_auth_enabled_with_keys(self):
        from amem.utils.auth import AuthManager
        auth = AuthManager(api_keys=["key1", "key2"])
        assert auth.enabled
        assert auth.validate("key1") is True
        assert auth.validate("wrong") is False
        assert auth.validate(None) is False

    def test_generate_key(self):
        from amem.utils.auth import AuthManager
        key = AuthManager.generate_key()
        assert key.startswith("amem_")
        assert len(key) > 20


# ─── Rate Limiter Tests ────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_within_limit(self):
        from amem.utils.ratelimit import RateLimiter
        from unittest.mock import MagicMock

        limiter = RateLimiter(rate=100.0, capacity=10)
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        # Should not raise for 10 requests
        for _ in range(10):
            limiter.check(request)

    def test_rejects_over_limit(self):
        from amem.utils.ratelimit import RateLimiter
        from fastapi import HTTPException
        from unittest.mock import MagicMock

        limiter = RateLimiter(rate=0.001, capacity=2)
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"

        limiter.check(request)
        limiter.check(request)
        with pytest.raises(HTTPException) as exc_info:
            limiter.check(request)
        assert exc_info.value.status_code == 429


# ─── Metrics Tests ─────────────────────────────────────────────────

class TestMetrics:
    def test_counter(self):
        from amem.utils.logging import MetricsCollector
        m = MetricsCollector()
        m.increment("test_counter")
        m.increment("test_counter")
        assert m.get_all()["counters"]["test_counter"] == 2

    def test_gauge(self):
        from amem.utils.logging import MetricsCollector
        m = MetricsCollector()
        m.gauge("test_gauge", 42.5)
        assert m.get_all()["gauges"]["test_gauge"] == 42.5

    def test_histogram(self):
        from amem.utils.logging import MetricsCollector
        m = MetricsCollector()
        for v in [1.0, 2.0, 3.0]:
            m.observe("test_hist", v)
        h = m.get_all()["histograms"]["test_hist"]
        assert h["count"] == 3
        assert h["mean"] == 2.0

    def test_prometheus_format(self):
        from amem.utils.logging import MetricsCollector
        m = MetricsCollector()
        m.increment("requests")
        m.gauge("chunks", 100)
        output = m.to_prometheus()
        assert "amem_requests_total 1" in output
        assert "amem_chunks 100" in output


# ─── Schema Migrations Tests ──────────────────────────────────────

class TestMigrations:
    def test_get_current_version(self):
        import sqlite3
        import tempfile
        from amem.persistence.migrations import get_current_version
        from amem.persistence.sqlite import SQLiteStore

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = SQLiteStore(f.name)
            conn = store._get_conn()
            v = get_current_version(conn)
            assert v == 1  # initial schema
            store.close()

    def test_no_pending_migrations(self):
        import sqlite3
        import tempfile
        from amem.persistence.migrations import needs_migration
        from amem.persistence.sqlite import SQLiteStore

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            store = SQLiteStore(f.name)
            conn = store._get_conn()
            assert needs_migration(conn) is False  # no migrations defined yet
            store.close()
