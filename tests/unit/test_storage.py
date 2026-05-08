"""Tests for SQLite storage."""

from datetime import datetime, timezone

import pytest

from llm_benchmark.config import Config
from llm_benchmark.metrics.types import BenchmarkResult, LatencyStats
from llm_benchmark.storage import Database


@pytest.fixture
def tmp_db(tmp_path):
    cfg = Config()
    cfg.db_path = tmp_path / "test.db"
    cfg.results_dir = tmp_path
    (tmp_path / "charts").mkdir()
    return Database(cfg)


def _speed_result(model="llama3:8b") -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        backend="ollama",
        benchmark_type="speed",
        prompt_set="general",
        mean_tps=42.5,
        std_tps=1.2,
        min_tps=40.0,
        max_tps=45.0,
        mean_rss_mb=800.0,
        peak_rss_mb=850.0,
        runs=3,
        timestamp=datetime.now(timezone.utc),
    )


def _latency_result(model="llama3:8b") -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        backend="ollama",
        benchmark_type="latency",
        prompt_set="general",
        latency=LatencyStats(
            p50_ms=120, p95_ms=200, p99_ms=250, mean_ms=130, min_ms=100, max_ms=300
        ),
        runs=5,
        timestamp=datetime.now(timezone.utc),
    )


def _quality_result(model="llama3:8b") -> BenchmarkResult:
    return BenchmarkResult(
        model=model,
        backend="ollama",
        benchmark_type="quality",
        prompt_set="general",
        quality_score=0.85,
        quality_details={"tasks": [], "per_category": {"reasoning": 0.9}},
        runs=10,
        timestamp=datetime.now(timezone.utc),
    )


class TestDatabase:
    def test_save_and_query_speed(self, tmp_db):
        row_id = tmp_db.save(_speed_result())
        assert row_id > 0
        rows = tmp_db.query()
        assert len(rows) == 1
        assert rows[0]["mean_tps"] == pytest.approx(42.5)
        assert rows[0]["bench_type"] == "speed"

    def test_save_and_query_latency(self, tmp_db):
        tmp_db.save(_latency_result())
        rows = tmp_db.query(bench_type="latency")
        assert rows[0]["ttft_p50_ms"] == pytest.approx(120)
        assert rows[0]["ttft_p95_ms"] == pytest.approx(200)

    def test_save_and_query_quality(self, tmp_db):
        tmp_db.save(_quality_result())
        rows = tmp_db.query(bench_type="quality")
        assert rows[0]["quality_score"] == pytest.approx(0.85)

    def test_query_filter_by_model(self, tmp_db):
        tmp_db.save(_speed_result("model-a"))
        tmp_db.save(_speed_result("model-b"))
        rows = tmp_db.query(model="model-a")
        assert len(rows) == 1
        assert rows[0]["model"] == "model-a"

    def test_query_limit(self, tmp_db):
        for _ in range(10):
            tmp_db.save(_speed_result())
        rows = tmp_db.query(limit=3)
        assert len(rows) == 3

    def test_all_models(self, tmp_db):
        tmp_db.save(_speed_result("alpha"))
        tmp_db.save(_speed_result("beta"))
        tmp_db.save(_latency_result("alpha"))
        models = tmp_db.all_models()
        assert set(models) == {"alpha", "beta"}

    def test_empty_query(self, tmp_db):
        assert tmp_db.query() == []
