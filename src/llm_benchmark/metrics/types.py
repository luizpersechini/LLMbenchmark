"""Shared dataclasses for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class GenerationResult:
    """Raw result from a single generation call."""

    model: str
    backend: str
    prompt: str
    output: str
    prompt_tokens: int
    output_tokens: int
    time_to_first_token_s: float  # seconds
    total_time_s: float  # seconds
    tokens_per_second: float
    peak_rss_mb: float  # process RSS at peak
    peak_metal_mb: float  # Metal GPU memory (unified, macOS only)
    timestamp: datetime = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return self.time_to_first_token_s * 1000


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass
class BenchmarkResult:
    """Aggregated result for a full benchmark run (multiple repeats)."""

    model: str
    backend: str
    benchmark_type: str  # "speed", "latency", "memory", "quality"
    prompt_set: str

    # Speed
    mean_tps: float = 0.0
    std_tps: float = 0.0
    min_tps: float = 0.0
    max_tps: float = 0.0

    # Latency
    latency: LatencyStats | None = None

    # Memory
    mean_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    mean_metal_mb: float = 0.0
    peak_metal_mb: float = 0.0

    # Quality
    quality_score: float | None = None  # 0.0 – 1.0
    quality_details: dict[str, Any] = field(default_factory=dict)

    runs: int = 0
    timestamp: datetime = field(default_factory=_now)

    raw: list[GenerationResult] = field(default_factory=list, repr=False)
