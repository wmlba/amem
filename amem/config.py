"""Configuration loading and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class OllamaConfig:
    """Legacy Ollama-specific config. Kept for backward compat."""
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    embedding_dim: int = 768


@dataclass
class EmbeddingProviderConfig:
    """Universal embedding provider config — works with any model."""
    provider: str = "auto"           # ollama, openai, anthropic, local, auto
    model: str = ""                  # model name (provider-specific)
    base_url: str = ""               # API base URL
    api_key: str = ""                # API key
    dimension: int = 0               # Override dimension (0 = auto-detect)


@dataclass
class EpisodicConfig:
    chunk_sentences: int = 3
    chunk_overlap: int = 1
    default_top_k: int = 10


@dataclass
class VectorIndexConfig:
    temporal_decay_lambda: float = 0.01
    reinforcement_weight: float = 0.1
    confidence_default: float = 1.0
    ivf_threshold: int = 10000
    n_partitions: int = 64


@dataclass
class SemanticConfig:
    max_traversal_depth: int = 2
    decay_lambda: float = 0.005
    min_confidence: float = 0.05
    extraction_mode: str = "embedding"  # "embedding", "heuristic", "llm"
    extraction_model: str = ""          # only for LLM mode


@dataclass
class BehavioralConfig:
    signal_window: int = 50
    dimensions: list[str] = field(default_factory=lambda: [
        "response_depth", "formality", "domain_expertise", "verbosity_preference",
    ])


@dataclass
class RetrievalConfig:
    token_budget: int = 4000
    episodic_weight: float = 0.4
    semantic_weight: float = 0.3
    explicit_weight: float = 0.2
    behavioral_weight: float = 0.1


@dataclass
class StorageConfig:
    data_dir: str = "./data"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingProviderConfig = field(default_factory=EmbeddingProviderConfig)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)
    vector_index: VectorIndexConfig = field(default_factory=VectorIndexConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    behavioral: BehavioralConfig = field(default_factory=BehavioralConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            ollama=OllamaConfig(**raw.get("ollama", {})),
            embedding=EmbeddingProviderConfig(**raw.get("embedding", {})),
            episodic=EpisodicConfig(**raw.get("episodic", {})),
            vector_index=VectorIndexConfig(**raw.get("vector_index", {})),
            semantic=SemanticConfig(**raw.get("semantic", {})),
            behavioral=BehavioralConfig(**raw.get("behavioral", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            storage=StorageConfig(**raw.get("storage", {})),
        )


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = Path("config.yaml")
    return Config.from_yaml(path)
