"""Tests for config module."""

import os
from pathlib import Path

import pytest

from llm_benchmark.config import Config, get_config


def test_defaults():
    cfg = Config()
    assert cfg.ollama_host == "http://localhost:11434"
    assert cfg.default_runs == 3
    assert cfg.default_warmup == 1
    assert cfg.default_max_tokens == 256
    assert cfg.default_temperature == 0.0


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_HOST", "http://custom:9999")
    monkeypatch.setenv("LLM_BENCH_RESULTS", str(tmp_path / "results"))
    cfg = Config()
    assert cfg.ollama_host == "http://custom:9999"
    assert cfg.results_dir == tmp_path / "results"


def test_results_dir_created(tmp_path):
    cfg = Config()
    cfg.results_dir = tmp_path / "new_results"
    cfg.__post_init__()
    assert cfg.results_dir.exists()
    assert (cfg.results_dir / "charts").exists()


def test_get_config_singleton():
    import llm_benchmark.config as c
    c._default = None
    a = get_config()
    b = get_config()
    assert a is b
    c._default = None  # reset
