"""Quality benchmark: scored tasks from JSONL prompt files."""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from ..backends.base import BaseBackend
from ..config import Config, get_config
from ..metrics.types import BenchmarkResult, GenerationResult
from .base import peak_rss_mb


def _score_mcq(output: str, expected: str) -> float:
    """Score a multiple-choice answer. Expected is 'A', 'B', 'C', or 'D'."""
    text = output.strip().upper()
    # Try to find the answer letter at the start of the response
    if text.startswith(expected):
        return 1.0
    # Look for patterns like "Answer: A", "(A)", "A)"
    patterns = [
        rf"\b{expected}\b",
        rf"\({expected}\)",
        rf"answer[:\s]+{expected}",
        rf"option[:\s]+{expected}",
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return 1.0
    return 0.0


def _score_keywords(output: str, keywords: list[str]) -> float:
    """Fraction of required keywords present in output (case-insensitive)."""
    if not keywords:
        return 1.0
    text = output.lower()
    found = sum(1 for kw in keywords if kw.lower() in text)
    return found / len(keywords)


def _score_task(output: str, task: dict[str, Any]) -> float:
    task_type = task.get("type", "keywords")
    if task_type == "mcq":
        return _score_mcq(output, task["answer"])
    if task_type == "keywords":
        return _score_keywords(output, task.get("keywords", []))
    if task_type == "exact":
        return 1.0 if output.strip().lower() == task["answer"].lower() else 0.0
    return 0.0


class QualityBenchmark:
    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or get_config()

    def _load_prompts(self, prompt_set: str) -> list[dict[str, Any]]:
        path = self._cfg.prompts_dir / f"{prompt_set}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Prompt set not found: {path}")
        tasks = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    def run(
        self,
        backend: BaseBackend,
        model: str,
        prompt_set: str | None = None,
        max_tokens: int = 128,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> BenchmarkResult:
        prompt_set = prompt_set or self._cfg.default_prompt_set
        tasks = self._load_prompts(prompt_set)
        raw_results: list[GenerationResult] = []
        scores: list[float] = []
        details: dict[str, Any] = {"tasks": []}

        for i, task in enumerate(tasks):
            if on_progress:
                on_progress(i + 1, len(tasks))

            prompt = task["prompt"]
            rss_before = peak_rss_mb()
            t0 = time.perf_counter()

            output, prompt_tokens, output_tokens = backend.generate(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            elapsed = time.perf_counter() - t0
            rss_after = peak_rss_mb()
            score = _score_task(output, task)
            scores.append(score)

            raw_results.append(
                GenerationResult(
                    model=model,
                    backend=backend.name,
                    prompt=prompt,
                    output=output,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    time_to_first_token_s=0.0,
                    total_time_s=elapsed,
                    tokens_per_second=output_tokens / elapsed if elapsed > 0 else 0.0,
                    peak_rss_mb=max(rss_before, rss_after),
                    peak_metal_mb=0.0,
                    metadata={"task_id": task.get("id", str(i)), "score": score},
                )
            )
            details["tasks"].append(
                {
                    "id": task.get("id", str(i)),
                    "category": task.get("category", "general"),
                    "score": score,
                    "output_snippet": output[:120],
                }
            )

        overall = statistics.mean(scores) if scores else 0.0
        details["per_category"] = _aggregate_by_category(details["tasks"])

        return BenchmarkResult(
            model=model,
            backend=backend.name,
            benchmark_type="quality",
            prompt_set=prompt_set,
            quality_score=overall,
            quality_details=details,
            mean_rss_mb=statistics.mean(r.peak_rss_mb for r in raw_results),
            peak_rss_mb=max(r.peak_rss_mb for r in raw_results),
            runs=len(tasks),
            raw=raw_results,
        )


def _aggregate_by_category(tasks: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for t in tasks:
        cat = t.get("category", "general")
        buckets.setdefault(cat, []).append(t["score"])
    return {cat: statistics.mean(scores) for cat, scores in buckets.items()}
