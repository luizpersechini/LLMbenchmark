"""Unit tests for benchmark runners using a fake backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from llm_benchmark.backends.base import BaseBackend, StreamChunk
from llm_benchmark.benchmarks.speed import SpeedBenchmark
from llm_benchmark.benchmarks.latency import LatencyBenchmark
from llm_benchmark.benchmarks.memory import MemoryBenchmark
from llm_benchmark.benchmarks.quality import QualityBenchmark, _score_mcq, _score_keywords
from llm_benchmark.config import Config
from llm_benchmark.metrics.types import BenchmarkResult


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------

class FakeBackend(BaseBackend):
    name = "fake"

    def __init__(self, tokens: list[str] | None = None, latency: float = 0.0) -> None:
        self._tokens = tokens or ["Hello", " ", "world", "!"]
        self._latency = latency

    def generate(self, model, prompt, max_tokens=256, temperature=0.0):
        output = "".join(self._tokens)
        return output, len(prompt.split()), len(self._tokens)

    def stream(self, model, prompt, max_tokens=256, temperature=0.0) -> Iterator[StreamChunk]:
        for i, tok in enumerate(self._tokens):
            if self._latency:
                time.sleep(self._latency / len(self._tokens))
            is_last = i == len(self._tokens) - 1
            yield StreamChunk(
                text=tok,
                is_final=is_last,
                prompt_tokens=len(prompt.split()) if is_last else 0,
                output_tokens=len(self._tokens) if is_last else 0,
            )

    def list_models(self):
        return ["fake-model"]

    def is_available(self):
        return True


# ---------------------------------------------------------------------------
# SpeedBenchmark
# ---------------------------------------------------------------------------

class TestSpeedBenchmark:
    def test_returns_benchmark_result(self, tmp_path):
        cfg = Config()
        cfg.default_runs = 2
        cfg.default_warmup = 1
        bench = SpeedBenchmark(cfg)
        result = bench.run(FakeBackend(), "fake-model", runs=2, warmup=1, max_tokens=10)
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_type == "speed"
        assert result.runs == 2

    def test_warmup_excluded(self):
        bench = SpeedBenchmark()
        result = bench.run(FakeBackend(), "fake-model", runs=3, warmup=2, max_tokens=5)
        assert result.runs == 3
        assert len(result.raw) == 3

    def test_tps_positive(self):
        bench = SpeedBenchmark()
        result = bench.run(FakeBackend(latency=0.0), "fake-model", runs=2, warmup=0, max_tokens=5)
        assert result.mean_tps > 0

    def test_progress_callback(self):
        calls = []
        bench = SpeedBenchmark()
        bench.run(FakeBackend(), "fake-model", runs=2, warmup=1, on_progress=lambda c, t: calls.append((c, t)))
        assert len(calls) == 3  # warmup + runs


# ---------------------------------------------------------------------------
# LatencyBenchmark
# ---------------------------------------------------------------------------

class TestLatencyBenchmark:
    def test_latency_stats_populated(self):
        bench = LatencyBenchmark()
        result = bench.run(FakeBackend(latency=0.01), "fake-model", runs=5, warmup=1)
        assert result.latency is not None
        assert result.latency.p50_ms >= 0
        assert result.latency.p95_ms >= result.latency.p50_ms

    def test_runs_count(self):
        bench = LatencyBenchmark()
        result = bench.run(FakeBackend(), "fake-model", runs=4, warmup=0)
        assert result.runs == 4


# ---------------------------------------------------------------------------
# MemoryBenchmark
# ---------------------------------------------------------------------------

class TestMemoryBenchmark:
    def test_memory_metrics_present(self):
        bench = MemoryBenchmark()
        result = bench.run(FakeBackend(), "fake-model", runs=1, warmup=0, max_tokens=5)
        assert result.mean_rss_mb > 0
        assert result.peak_rss_mb >= result.mean_rss_mb


# ---------------------------------------------------------------------------
# QualityBenchmark scoring functions
# ---------------------------------------------------------------------------

class TestScoringFunctions:
    def test_mcq_exact_match(self):
        assert _score_mcq("A", "A") == 1.0
        assert _score_mcq("B", "A") == 0.0

    def test_mcq_with_prefix(self):
        assert _score_mcq("Answer: A", "A") == 1.0
        assert _score_mcq("The answer is (B)", "B") == 1.0

    def test_mcq_case_insensitive(self):
        assert _score_mcq("answer: a", "A") == 1.0

    def test_keywords_all_present(self):
        assert _score_keywords("Python is a dynamic language", ["python", "dynamic"]) == 1.0

    def test_keywords_partial(self):
        score = _score_keywords("Python is cool", ["python", "java", "ruby"])
        assert score == pytest.approx(1 / 3)

    def test_keywords_empty(self):
        assert _score_keywords("anything", []) == 1.0


# ---------------------------------------------------------------------------
# QualityBenchmark with real prompt file
# ---------------------------------------------------------------------------

class TestQualityBenchmark:
    @pytest.fixture
    def prompts_dir(self, tmp_path):
        """Write a minimal JSONL prompt file."""
        d = tmp_path / "prompts"
        d.mkdir()
        tasks = [
            {"id": "q1", "prompt": "What is 2+2?", "type": "mcq", "answer": "A", "category": "math"},
            {"id": "q2", "prompt": "Name a programming language.", "type": "keywords", "keywords": ["python"], "category": "coding"},
        ]
        (d / "general.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
        return d

    def test_perfect_score(self, prompts_dir):
        class PerfectBackend(FakeBackend):
            def generate(self, model, prompt, **kwargs):
                if "2+2" in prompt:
                    return "A", 4, 1
                return "Python is a language", 5, 4

        cfg = Config()
        cfg.prompts_dir = prompts_dir
        bench = QualityBenchmark(cfg)
        result = bench.run(PerfectBackend(), "perfect", prompt_set="general")
        assert result.quality_score == pytest.approx(1.0)

    def test_zero_score(self, prompts_dir):
        class WrongBackend(FakeBackend):
            def generate(self, model, prompt, **kwargs):
                return "I don't know", 3, 4

        cfg = Config()
        cfg.prompts_dir = prompts_dir
        bench = QualityBenchmark(cfg)
        result = bench.run(WrongBackend(), "wrong", prompt_set="general")
        assert result.quality_score == pytest.approx(0.0)

    def test_missing_prompt_file_raises(self, tmp_path):
        cfg = Config()
        cfg.prompts_dir = tmp_path
        bench = QualityBenchmark(cfg)
        with pytest.raises(FileNotFoundError):
            bench.run(FakeBackend(), "m", prompt_set="nonexistent")

    def test_per_category_in_details(self, prompts_dir):
        class MixedBackend(FakeBackend):
            def generate(self, model, prompt, **kwargs):
                if "2+2" in prompt:
                    return "A", 4, 1          # math: correct
                return "no idea", 3, 2        # coding: wrong

        cfg = Config()
        cfg.prompts_dir = prompts_dir
        bench = QualityBenchmark(cfg)
        result = bench.run(MixedBackend(), "mixed", prompt_set="general")
        per_cat = result.quality_details.get("per_category", {})
        assert "math" in per_cat
        assert "coding" in per_cat
        assert per_cat["math"] == pytest.approx(1.0)
        assert per_cat["coding"] == pytest.approx(0.0)
