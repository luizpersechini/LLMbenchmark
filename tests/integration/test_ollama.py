"""Integration tests against a live Ollama instance.

Run with: pytest --run-integration tests/integration/
Requires Ollama running locally with at least one model pulled.
"""

from __future__ import annotations

import pytest

from llm_benchmark.backends.ollama import OllamaBackend
from llm_benchmark.benchmarks.speed import SpeedBenchmark
from llm_benchmark.benchmarks.latency import LatencyBenchmark
from llm_benchmark.benchmarks.memory import MemoryBenchmark


@pytest.fixture(scope="module")
def backend():
    b = OllamaBackend()
    if not b.is_available():
        pytest.skip("Ollama is not running — skipping integration tests")
    return b


@pytest.fixture(scope="module")
def model(backend):
    models = backend.list_models()
    if not models:
        pytest.skip("No models available in Ollama")
    # prefer a small model if available
    for preferred in ["llama3.2:1b", "phi4-mini", "gemma:2b", "llama3:8b"]:
        if preferred in models:
            return preferred
    return models[0]


class TestOllamaBackendLive:
    def test_list_models_nonempty(self, backend):
        models = backend.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_generate_returns_text(self, backend, model):
        output, pt, ot = backend.generate(model, "Say 'hello' and nothing else.", max_tokens=10)
        assert isinstance(output, str)
        assert len(output) > 0
        assert pt > 0
        assert ot > 0

    def test_stream_yields_chunks(self, backend, model):
        chunks = list(backend.stream(model, "Say 'hi'.", max_tokens=5))
        assert len(chunks) > 0
        assert any(c.is_final for c in chunks)
        text = "".join(c.text for c in chunks)
        assert len(text) > 0

    def test_stream_final_chunk_has_token_counts(self, backend, model):
        chunks = list(backend.stream(model, "Hello.", max_tokens=5))
        final = next(c for c in reversed(chunks) if c.is_final)
        assert final.output_tokens > 0


class TestSpeedBenchmarkLive:
    @pytest.mark.slow
    def test_speed_tps_positive(self, backend, model):
        bench = SpeedBenchmark()
        result = bench.run(backend, model, runs=2, warmup=1, max_tokens=50)
        assert result.mean_tps > 0
        assert result.model == model


class TestLatencyBenchmarkLive:
    def test_latency_populated(self, backend, model):
        bench = LatencyBenchmark()
        result = bench.run(backend, model, runs=3, warmup=1)
        assert result.latency is not None
        assert result.latency.p50_ms > 0
        assert result.latency.p95_ms >= result.latency.p50_ms


class TestMemoryBenchmarkLive:
    def test_memory_rss_positive(self, backend, model):
        bench = MemoryBenchmark()
        result = bench.run(backend, model, runs=1, warmup=0, max_tokens=30)
        assert result.peak_rss_mb > 0
