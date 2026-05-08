"""Tests for report formatter and chart generation."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pytest

from llm_benchmark.config import Config
from llm_benchmark.metrics.types import BenchmarkResult, LatencyStats
from llm_benchmark.report.formatter import print_result, print_comparison_table
from llm_benchmark.report.charts import generate_charts, generate_html_report


def _speed_result(model="m1") -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        backend="ollama",
        benchmark_type="speed",
        prompt_set="general",
        mean_tps=55.0,
        std_tps=2.0,
        min_tps=50.0,
        max_tps=60.0,
        mean_rss_mb=512.0,
        peak_rss_mb=600.0,
        runs=3,
    )


def _latency_result(model="m1") -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        backend="ollama",
        benchmark_type="latency",
        prompt_set="general",
        latency=LatencyStats(p50_ms=100, p95_ms=200, p99_ms=250, mean_ms=120, min_ms=90, max_ms=300),
        runs=5,
    )


def _quality_result(model="m1") -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        backend="ollama",
        benchmark_type="quality",
        prompt_set="general",
        quality_score=0.75,
        quality_details={"per_category": {"math": 0.8, "coding": 0.7}, "tasks": []},
        runs=10,
    )


class TestPrintResult:
    """Smoke-test that print_result doesn't raise."""

    def test_speed(self, capsys):
        print_result(_speed_result())

    def test_latency(self, capsys):
        print_result(_latency_result())

    def test_quality(self, capsys):
        print_result(_quality_result())

    def test_memory(self, capsys):
        r = BenchmarkResult(
            model="m1", backend="b", benchmark_type="memory", prompt_set="general",
            mean_rss_mb=400, peak_rss_mb=450, mean_metal_mb=200, peak_metal_mb=250, runs=2,
        )
        print_result(r)


class TestComparisonTable:
    def test_speed_comparison(self, capsys):
        results = [_speed_result("a"), _speed_result("b")]
        print_comparison_table(results)

    def test_latency_comparison(self, capsys):
        results = [_latency_result("a"), _latency_result("b")]
        print_comparison_table(results)

    def test_empty_list(self, capsys):
        print_comparison_table([])


class TestCharts:
    def _rows(self):
        ts = datetime.now(timezone.utc).isoformat()
        return [
            {"model": "a", "backend": "ollama", "bench_type": "speed",
             "mean_tps": 50.0, "std_tps": 1.0, "ttft_p50_ms": None,
             "ttft_p95_ms": None, "peak_rss_mb": 500.0,
             "mean_metal_mb": 0.0, "peak_metal_mb": 0.0, "quality_score": None, "timestamp": ts},
            {"model": "b", "backend": "ollama", "bench_type": "speed",
             "mean_tps": 80.0, "std_tps": 2.0, "ttft_p50_ms": None,
             "ttft_p95_ms": None, "peak_rss_mb": 700.0,
             "mean_metal_mb": 0.0, "peak_metal_mb": 0.0, "quality_score": None, "timestamp": ts},
        ]

    def test_generates_speed_chart(self, tmp_path):
        cfg = Config()
        cfg.results_dir = tmp_path
        (tmp_path / "charts").mkdir()
        paths = generate_charts(self._rows(), cfg)
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)
        assert any("speed" in p.name for p in paths)

    def test_html_report_created(self, tmp_path):
        cfg = Config()
        cfg.results_dir = tmp_path
        (tmp_path / "charts").mkdir()
        charts = generate_charts(self._rows(), cfg)
        html = generate_html_report(self._rows(), charts, cfg)
        assert html.exists()
        content = html.read_text()
        assert "<html" in content
        assert "LLM Benchmark" in content
