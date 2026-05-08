"""Tests for metrics.types dataclasses."""

from datetime import datetime

from llm_benchmark.metrics.types import GenerationResult, BenchmarkResult, LatencyStats


def test_generation_result_latency_ms():
    r = GenerationResult(
        model="test",
        backend="ollama",
        prompt="hi",
        output="hello",
        prompt_tokens=1,
        output_tokens=1,
        time_to_first_token_s=0.250,
        total_time_s=1.0,
        tokens_per_second=1.0,
        peak_rss_mb=100.0,
        peak_metal_mb=0.0,
    )
    assert r.latency_ms == pytest.approx(250.0)


def test_benchmark_result_defaults():
    r = BenchmarkResult(model="m", backend="b", benchmark_type="speed", prompt_set="general")
    assert r.mean_tps == 0.0
    assert r.quality_score is None
    assert r.latency is None
    assert r.raw == []


def test_latency_stats():
    lat = LatencyStats(p50_ms=100, p95_ms=200, p99_ms=250, mean_ms=120, min_ms=80, max_ms=300)
    assert lat.p50_ms == 100
    assert lat.p99_ms == 250


import pytest
