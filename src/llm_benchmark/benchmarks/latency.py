"""Latency benchmark: measures time-to-first-token (TTFT) with percentiles."""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np

from ..backends.base import BaseBackend
from ..config import Config, get_config
from ..metrics.types import BenchmarkResult, GenerationResult, LatencyStats
from .base import peak_rss_mb


_LATENCY_PROMPT = "Hello, how are you?"   # short prompt → TTFT dominated by model, not input


class LatencyBenchmark:
    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or get_config()

    def run(
        self,
        backend: BaseBackend,
        model: str,
        prompt: str = _LATENCY_PROMPT,
        max_tokens: int = 1,            # just first token
        runs: int | None = None,
        warmup: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        runs = runs or self._cfg.default_runs
        warmup = warmup if warmup is not None else self._cfg.default_warmup
        total = warmup + runs
        raw_results: list[GenerationResult] = []

        for i in range(total):
            if on_progress:
                on_progress(i + 1, total)

            rss_before = peak_rss_mb()
            t0 = time.perf_counter()

            chunks = []
            first_token_time: float | None = None

            for chunk in backend.stream(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self._cfg.default_temperature,
            ):
                if first_token_time is None and chunk.text:
                    first_token_time = time.perf_counter() - t0
                chunks.append(chunk)
                if chunk.is_final:
                    break

            elapsed = time.perf_counter() - t0
            rss_after = peak_rss_mb()
            final = next((c for c in reversed(chunks) if c.is_final), None)

            raw_results.append(
                GenerationResult(
                    model=model,
                    backend=backend.name,
                    prompt=prompt,
                    output="".join(c.text for c in chunks),
                    prompt_tokens=final.prompt_tokens if final else 0,
                    output_tokens=final.output_tokens if final else 1,
                    time_to_first_token_s=first_token_time or elapsed,
                    total_time_s=elapsed,
                    tokens_per_second=1.0 / elapsed if elapsed > 0 else 0.0,
                    peak_rss_mb=max(rss_before, rss_after),
                    peak_metal_mb=0.0,
                )
            )

        valid = raw_results[warmup:]
        ttft_ms = [r.time_to_first_token_s * 1000 for r in valid]
        arr = np.array(ttft_ms)

        lat = LatencyStats(
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(arr.mean()),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
        )

        return BenchmarkResult(
            model=model,
            backend=backend.name,
            benchmark_type="latency",
            prompt_set="custom",
            latency=lat,
            mean_rss_mb=statistics.mean(r.peak_rss_mb for r in valid),
            peak_rss_mb=max(r.peak_rss_mb for r in valid),
            runs=runs,
            raw=valid,
        )
