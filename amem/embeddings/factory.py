"""Embedding provider factory.

One config field. One function. Any model.

Usage:
    embedder = create_embedder(config)  # reads config.embedding section
    # Everything else (episodic store, entity extraction, dedup, resolution)
    # uses this same embedder instance. One model for all five uses.

Config examples:

    # Ollama (local, free)
    embedding:
      provider: ollama
      model: nomic-embed-text
      base_url: http://localhost:11434

    # OpenAI
    embedding:
      provider: openai
      model: text-embedding-3-small
      api_key: sk-...

    # Any OpenAI-compatible server (vLLM, llama.cpp, LiteLLM, Together, etc.)
    embedding:
      provider: openai
      model: your-model-name
      base_url: http://localhost:8000/v1
      api_key: optional

    # Claude/Anthropic (Voyage)
    embedding:
      provider: anthropic
      model: voyage-3
      api_key: ...

    # Local sentence-transformers (completely offline)
    embedding:
      provider: local
      model: all-MiniLM-L6-v2

    # Auto-detect: tries Ollama → local → fails with helpful error
    embedding:
      provider: auto
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from amem.embeddings.base import EmbeddingProvider


@dataclass
class EmbeddingConfig:
    """Universal embedding configuration."""
    provider: str = "auto"           # ollama, openai, anthropic, local, auto
    model: str = ""                  # model name (provider-specific)
    base_url: str = ""               # API base URL (for ollama, openai-compat)
    api_key: str = ""                # API key (for openai, anthropic)
    dimension: int = 0               # Override dimension (0 = auto-detect)

    def __post_init__(self):
        # Read API key from environment if not set
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY", "")
            elif self.provider == "anthropic":
                self.api_key = os.environ.get("VOYAGE_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")


def create_embedder(config: Any = None) -> EmbeddingProvider:
    """Create an embedding provider from config.

    Accepts:
    - EmbeddingConfig dataclass
    - dict with provider/model/base_url/api_key keys
    - None (auto-detect)
    - The old OllamaConfig (backward compat)

    Returns an initialized EmbeddingProvider ready for use.
    """
    # Normalize config
    if config is None:
        ec = EmbeddingConfig(provider="auto")
    elif isinstance(config, EmbeddingConfig):
        ec = config
    elif isinstance(config, dict):
        ec = EmbeddingConfig(**{k: v for k, v in config.items() if k in EmbeddingConfig.__dataclass_fields__})
    elif hasattr(config, "base_url") and hasattr(config, "model"):
        # Backward compat: OllamaConfig
        return _create_ollama(config.base_url, config.model, getattr(config, "embedding_dim", 0))
    else:
        ec = EmbeddingConfig(provider="auto")

    provider = ec.provider.lower()

    if provider == "ollama":
        base = ec.base_url or "http://localhost:11434"
        model = ec.model or "nomic-embed-text"
        return _create_ollama(base, model, ec.dimension)

    elif provider == "openai":
        base = ec.base_url or "https://api.openai.com/v1"
        model = ec.model or "text-embedding-3-small"
        return _create_openai(base, model, ec.api_key, ec.dimension)

    elif provider == "anthropic":
        model = ec.model or "voyage-3"
        return _create_anthropic(model, ec.api_key, ec.base_url)

    elif provider == "local":
        model = ec.model or "all-MiniLM-L6-v2"
        return _create_local(model)

    elif provider == "auto":
        return _auto_detect(ec)

    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Supported: ollama, openai, anthropic, local, auto"
        )


def _create_ollama(base_url: str, model: str, dimension: int = 0) -> EmbeddingProvider:
    from amem.embeddings.ollama import OllamaEmbedding
    from amem.config import OllamaConfig
    cfg = OllamaConfig(base_url=base_url, model=model, embedding_dim=dimension or 768)
    return OllamaEmbedding(cfg)


def _create_openai(base_url: str, model: str, api_key: str, dimension: int = 0) -> EmbeddingProvider:
    from amem.embeddings.openai_embed import OpenAIEmbedding
    return OpenAIEmbedding(
        model=model,
        base_url=base_url,
        api_key=api_key,
        dimension=dimension or None,
    )


def _create_anthropic(model: str, api_key: str, base_url: str = "") -> EmbeddingProvider:
    from amem.embeddings.anthropic_embed import AnthropicEmbedding
    return AnthropicEmbedding(
        model=model,
        api_key=api_key,
        base_url=base_url or "https://api.voyageai.com/v1",
    )


def _create_local(model: str) -> EmbeddingProvider:
    from amem.embeddings.local_embed import LocalEmbedding
    return LocalEmbedding(model_name=model)


def _auto_detect(ec: EmbeddingConfig) -> EmbeddingProvider:
    """Try providers in order: Ollama → local → fail with helpful error."""
    import httpx

    # Try Ollama first (most common for local development)
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            # Prefer embedding-specific models
            for m in models:
                name = m.get("name", "").split(":")[0]
                if name in ("nomic-embed-text", "all-minilm", "mxbai-embed-large"):
                    return _create_ollama("http://localhost:11434", m.get("name", name))
            # Fall back to any available model
            if models:
                return _create_ollama("http://localhost:11434", models[0].get("name", ""))
    except (httpx.ConnectError, httpx.TimeoutException):
        pass

    # Try local sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        return _create_local(ec.model or "all-MiniLM-L6-v2")
    except ImportError:
        pass

    # Check for OpenAI key in environment
    if os.environ.get("OPENAI_API_KEY"):
        return _create_openai(
            "https://api.openai.com/v1",
            ec.model or "text-embedding-3-small",
            os.environ["OPENAI_API_KEY"],
            0,
        )

    raise RuntimeError(
        "No embedding provider available. Options:\n"
        "  1. Start Ollama: ollama serve && ollama pull nomic-embed-text\n"
        "  2. Install sentence-transformers: pip install sentence-transformers\n"
        "  3. Set OPENAI_API_KEY environment variable\n"
        "  4. Configure explicitly in config.yaml:\n"
        "     embedding:\n"
        "       provider: openai\n"
        "       model: text-embedding-3-small\n"
        "       api_key: sk-...\n"
    )
