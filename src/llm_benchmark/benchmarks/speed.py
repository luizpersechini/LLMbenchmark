"""Throughput benchmark: measures sustained tokens/second."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

from ..backends.base import BaseBackend
from ..config import Config, get_config
from ..metrics.types import BenchmarkResult, GenerationResult
from .base import peak_rss_mb

_DEFAULT_PROMPT = (
    "Write a detailed explanation of how transformer neural networks work, "
    "covering attention mechanisms, positional encoding, and training."
)


class SpeedBenchmark:
    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or get_config()

    def run(
        self,
        backend: BaseBackend,
        model: str,
        prompt: str = _DEFAULT_PROMPT,
        max_tokens: int | None = None,
        runs: int | None = None,
        warmup: int | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        max_tokens = max_tokens or self._cfg.default_max_tokens
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

            elapsed = time.perf_counter() - t0
            rss_after = peak_rss_mb()

            final = next((c for c in reversed(chunks) if c.is_final), None)
            output_tokens = final.output_tokens if final else sum(1 for c in chunks if c.text)
            prompt_tokens = final.prompt_tokens if final else 0
            output_text = "".join(c.text for c in chunks)
            tps = output_tokens / elapsed if elapsed > 0 else 0.0

            raw_results.append(
                GenerationResult(
                    model=model,
                    backend=backend.name,
                    prompt=prompt,
                    output=output_text,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    time_to_first_token_s=first_token_time or 0.0,
                    total_time_s=elapsed,
                    tokens_per_second=tps,
                    peak_rss_mb=max(rss_before, rss_after),
                    peak_metal_mb=0.0,
                )
            )

        valid = raw_results[warmup:]
        tps_values = [r.tokens_per_second for r in valid]

        return BenchmarkResult(
            model=model,
            backend=backend.name,
            benchmark_type="speed",
            prompt_set="custom",
            mean_tps=statistics.mean(tps_values),
            std_tps=statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0,
            min_tps=min(tps_values),
            max_tps=max(tps_values),
            mean_rss_mb=statistics.mean(r.peak_rss_mb for r in valid),
            peak_rss_mb=max(r.peak_rss_mb for r in valid),
            runs=runs,
            raw=valid,
        )
