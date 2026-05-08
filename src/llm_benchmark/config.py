"""Central configuration dataclass, loaded from env vars or constructor kwargs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Ollama
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "120")))

    # Benchmark defaults
    default_runs: int = 3
    default_warmup: int = 1
    default_max_tokens: int = 256
    default_temperature: float = 0.0   # deterministic for reproducibility
    default_prompt_set: str = "general"

    # Storage
    results_dir: Path = field(default_factory=lambda: Path(os.getenv("LLM_BENCH_RESULTS", "results")))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("LLM_BENCH_DB", "results/bench.db")))

    # Prompts
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).parents[2] / "prompts")

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.db_path = Path(self.db_path)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "charts").mkdir(exist_ok=True)


_default: Config | None = None


def get_config() -> Config:
    global _default
    if _default is None:
        _default = Config()
    return _default
