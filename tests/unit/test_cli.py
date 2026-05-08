"""CLI smoke tests using Click test runner."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_benchmark.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestListCommand:
    def test_lists_models(self, runner):
        with patch("llm_benchmark.cli.get_backend") as mock_get:
            backend = MagicMock()
            backend.is_available.return_value = True
            backend.list_models.return_value = ["llama3:8b", "phi3:mini"]
            mock_get.return_value = backend
            result = runner.invoke(main, ["list", "--backend", "ollama"])
        assert result.exit_code == 0
        assert "llama3:8b" in result.output

    def test_backend_unavailable(self, runner):
        with patch("llm_benchmark.cli.get_backend") as mock_get:
            backend = MagicMock()
            backend.is_available.return_value = False
            mock_get.return_value = backend
            result = runner.invoke(main, ["list"])
        assert result.exit_code == 1

    def test_no_models(self, runner):
        with patch("llm_benchmark.cli.get_backend") as mock_get:
            backend = MagicMock()
            backend.is_available.return_value = True
            backend.list_models.return_value = []
            mock_get.return_value = backend
            result = runner.invoke(main, ["list"])
        assert "No models" in result.output


class TestResultsCommand:
    def test_empty_results(self, runner, tmp_path):
        with patch("llm_benchmark.cli.Database") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query.return_value = []
            mock_db_cls.return_value = mock_db
            result = runner.invoke(main, ["results"])
        assert "No results" in result.output

    def test_json_output(self, runner):
        row = {
            "model": "llama3",
            "backend": "ollama",
            "bench_type": "speed",
            "mean_tps": 42.0,
            "std_tps": 1.0,
            "min_tps": 40.0,
            "max_tps": 44.0,
            "ttft_p50_ms": None,
            "ttft_p95_ms": None,
            "ttft_p99_ms": None,
            "mean_rss_mb": 500.0,
            "peak_rss_mb": 550.0,
            "mean_metal_mb": 0.0,
            "peak_metal_mb": 0.0,
            "quality_score": None,
            "quality_details": None,
            "run_count": 3,
            "timestamp": "2024-01-01T00:00:00",
            "id": 1,
        }
        with patch("llm_benchmark.cli.Database") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query.return_value = [row]
            mock_db_cls.return_value = mock_db
            result = runner.invoke(main, ["results", "--output", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["model"] == "llama3"


class TestReportCommand:
    def test_no_data(self, runner):
        with patch("llm_benchmark.cli.Database") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query.return_value = []
            mock_db_cls.return_value = mock_db
            result = runner.invoke(main, ["report"])
        assert "No results" in result.output

    def test_generates_report(self, runner, tmp_path):
        rows = [
            {
                "model": "a",
                "backend": "ollama",
                "bench_type": "speed",
                "mean_tps": 50.0,
                "std_tps": 1.0,
                "ttft_p50_ms": None,
                "ttft_p95_ms": None,
                "peak_rss_mb": 500.0,
                "mean_metal_mb": 0.0,
                "peak_metal_mb": 0.0,
                "quality_score": None,
                "timestamp": "2024-01-01T00:00:00",
            },
        ]
        with (
            patch("llm_benchmark.cli.Database") as mock_db_cls,
            patch("llm_benchmark.cli.generate_charts") as mock_charts,
            patch("llm_benchmark.cli.generate_html_report") as mock_html,
        ):
            mock_db = MagicMock()
            mock_db.query.return_value = rows
            mock_db_cls.return_value = mock_db
            mock_charts.return_value = []
            mock_html.return_value = tmp_path / "report.html"
            (tmp_path / "report.html").write_text("<html></html>")
            result = runner.invoke(main, ["report"])
        assert result.exit_code == 0
