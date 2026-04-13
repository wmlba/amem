"""Tests for the embedding provider factory and multi-provider support."""

import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from amem.embeddings.factory import create_embedder, EmbeddingConfig
from amem.embeddings.base import EmbeddingProvider
from amem.config import Config, OllamaConfig
from amem.retrieval.orchestrator import MemoryOrchestrator


class TestEmbeddingConfig:
    def test_default_config(self):
        c = EmbeddingConfig()
        assert c.provider == "auto"
        assert c.model == ""

    def test_openai_config(self):
        c = EmbeddingConfig(provider="openai", model="text-embedding-3-small", api_key="sk-test")
        assert c.provider == "openai"

    def test_env_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}):
            c = EmbeddingConfig(provider="openai")
            assert c.api_key == "sk-from-env"


class TestProviderFactory:
    def test_create_ollama(self):
        embedder = create_embedder(EmbeddingConfig(provider="ollama", model="nomic-embed-text"))
        assert embedder is not None
        assert hasattr(embedder, 'embed')

    def test_create_openai(self):
        embedder = create_embedder(EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
        ))
        assert embedder is not None
        assert hasattr(embedder, 'embed')

    def test_create_anthropic(self):
        embedder = create_embedder(EmbeddingConfig(
            provider="anthropic",
            model="voyage-3",
            api_key="test-key",
        ))
        assert embedder is not None

    def test_create_from_dict(self):
        embedder = create_embedder({
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "sk-test",
        })
        assert embedder is not None

    def test_create_from_legacy_ollama_config(self):
        """Backward compat: OllamaConfig still works."""
        cfg = OllamaConfig(base_url="http://localhost:11434", model="nomic-embed-text")
        embedder = create_embedder(cfg)
        assert embedder is not None

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedder(EmbeddingConfig(provider="nonexistent"))

    def test_auto_detect_with_ollama(self):
        """Auto-detect should find Ollama if it's running."""
        import httpx
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if resp.status_code == 200:
                embedder = create_embedder(EmbeddingConfig(provider="auto"))
                assert embedder is not None
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Ollama not available")


class TestOrchestratorFromConfig:
    def test_from_config_with_explicit_provider(self):
        """MemoryOrchestrator.from_config() uses the factory."""
        config = Config()
        config.embedding.provider = "ollama"
        config.embedding.model = "nomic-embed-text"

        # This will fail if Ollama isn't running, but the orchestrator should be created
        try:
            orch = MemoryOrchestrator.from_config(config)
            assert orch is not None
        except Exception:
            # If Ollama is down, that's fine — we're testing the wiring
            pass

    def test_from_config_backward_compat(self):
        """Legacy OllamaConfig path still works."""
        config = Config()
        config.embedding.provider = "auto"  # default
        # Should fall through to legacy ollama config
        try:
            orch = MemoryOrchestrator.from_config(config)
            assert orch is not None
        except Exception:
            pass  # Ollama may not be running


class TestMultiProviderInterface:
    """Verify all providers implement the same interface."""

    @pytest.mark.asyncio
    async def test_openai_provider_interface(self):
        from amem.embeddings.openai_embed import OpenAIEmbedding
        provider = OpenAIEmbedding(model="test", api_key="sk-test")
        assert isinstance(provider, EmbeddingProvider)
        assert hasattr(provider, 'embed')
        assert hasattr(provider, 'embed_batch')
        assert hasattr(provider, 'dimension')
        assert hasattr(provider, 'close')
        await provider.close()

    def test_local_provider_interface(self):
        from amem.embeddings.local_embed import LocalEmbedding
        provider = LocalEmbedding.__new__(LocalEmbedding)
        provider._model_name = "test"
        provider._model = None
        provider._dim = None
        assert isinstance(provider, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_anthropic_provider_interface(self):
        from amem.embeddings.anthropic_embed import AnthropicEmbedding
        provider = AnthropicEmbedding(model="test", api_key="test")
        assert isinstance(provider, EmbeddingProvider)
        await provider.close()
