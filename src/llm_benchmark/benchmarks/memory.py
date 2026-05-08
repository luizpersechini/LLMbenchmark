"""Memory benchmark: tracks RSS and Metal GPU memory during generation."""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable

import psutil

from ..backends.base import BaseBackend
from ..config import Config, get_config
from ..metrics.collector import _metal_memory_mb
from ..metrics.types import BenchmarkResult, GenerationResult

_MEM_PROMPT = (
    "Explain in detail the history of computing from vacuum tubes to modern "
    "silicon chips, covering key milestones and innovations across each decade."
)


class _MemoryPoller:
    """Background thread that polls RSS and Metal memory during a generation."""

    def __init__(self, interval: float = 0.1) -> None:
        self._interval = interval
        self._rss: list[float] = []
        self._metal: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        peak_rss = max(self._rss, default=0.0)
        peak_metal = max(self._metal, default=0.0)
        return peak_rss, peak_metal

    def _poll(self) -> None:
        while self._running:
            try:
                self._rss.append(self._proc.memory_info().rss / (1024 * 1024))
                self._metal.append(_metal_memory_mb())
            except Exception:
                pass
            time.sleep(self._interval)


class MemoryBenchmark:
    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or get_config()

    def run(
        self,
        backend: BaseBackend,
        model: str,
        prompt: str = _MEM_PROMPT,
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

            poller = _MemoryPoller()
            poller.start()
            t0 = time.perf_counter()
            first_token_time: float | None = None
            chunks = []

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
            peak_rss, peak_metal = poller.stop()

            final = next((c for c in reversed(chunks) if c.is_final), None)
            output_tokens = final.output_tokens if final else sum(1 for c in chunks if c.text)

            raw_results.append(
                GenerationResult(
                    model=model,
                    backend=backend.name,
                    prompt=prompt,
                    output="".join(c.text for c in chunks),
                    prompt_tokens=final.prompt_tokens if final else 0,
                    output_tokens=output_tokens,
                    time_to_first_token_s=first_token_time or 0.0,
                    total_time_s=elapsed,
                    tokens_per_second=output_tokens / elapsed if elapsed > 0 else 0.0,
                    peak_rss_mb=peak_rss,
                    peak_metal_mb=peak_metal,
                )
            )

        valid = raw_results[warmup:]
        return BenchmarkResult(
            model=model,
            backend=backend.name,
            benchmark_type="memory",
            prompt_set="custom",
            mean_rss_mb=statistics.mean(r.peak_rss_mb for r in valid),
            peak_rss_mb=max(r.peak_rss_mb for r in valid),
            mean_metal_mb=statistics.mean(r.peak_metal_mb for r in valid),
            peak_metal_mb=max(r.peak_metal_mb for r in valid),
            runs=runs,
            raw=valid,
        )
