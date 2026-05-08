# llm-benchmark

Local LLM benchmarking tool built for Apple Silicon (M-series). Measures inference speed, latency, memory usage, and output quality across models served via [Ollama](https://ollama.com) or [MLX](https://github.com/ml-explore/mlx).

## Hardware target

Designed for Apple Silicon (M1–M4) with unified memory. Tested on MacBook Air M4 24 GB.

## Features

- **Speed** — tokens/second sustained generation, warmup-aware
- **Latency** — time-to-first-token (TTFT) with P50/P95/P99 percentiles
- **Memory** — process RSS and Metal GPU memory via `ioreg`
- **Quality** — scored reasoning, coding, and math tasks from local prompt sets
- **Compare** — side-by-side table for multiple models
- **Report** — HTML report with matplotlib charts saved to `results/`
- **Backends** — Ollama (HTTP) and MLX-LM (native)
- **SQLite storage** — all runs persisted; re-queryable with `llm-bench results`

## Quick start

```bash
# 1. Install (requires Python 3.10+)
pip install -e ".[dev]"

# 2. Pull some models via Ollama
ollama pull llama3.2:3b
ollama pull llama3.2:1b
ollama pull gemma3:4b
ollama pull phi4-mini

# 3. Run full benchmark on one model
llm-bench run llama3.2:3b

# 4. Compare several models
llm-bench compare llama3.2:1b llama3.2:3b gemma3:4b

# 5. Generate HTML report
llm-bench report --open

# 6. Show stored results
llm-bench results --last 10
```

## CLI reference

```
llm-bench list                         List available models from all backends
llm-bench run  MODEL [OPTIONS]         Benchmark a single model
llm-bench compare MODEL... [OPTIONS]   Benchmark and compare multiple models
llm-bench results [OPTIONS]            Query stored benchmark history
llm-bench report [OPTIONS]             Generate HTML/chart report from stored runs
```

### `run` options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `ollama` | Backend: `ollama` or `mlx` |
| `--runs` | `3` | Repetitions per benchmark |
| `--warmup` | `1` | Warmup runs (excluded from stats) |
| `--max-tokens` | `256` | Max tokens to generate per run |
| `--prompt-set` | `general` | Prompt set: `general`, `coding`, `reasoning`, `math` |
| `--skip-quality` | off | Skip quality benchmarks (faster) |
| `--output` | `table` | Output format: `table`, `json`, `csv` |

## Project layout

```
src/llm_benchmark/
├── backends/       # Ollama and MLX backends
├── benchmarks/     # Speed, latency, memory, quality runners
├── metrics/        # System metric collection (CPU, RAM, Metal)
├── storage/        # SQLite persistence
├── report/         # Rich tables + matplotlib chart generation
└── cli.py          # Click entry points
prompts/            # JSONL prompt datasets
tests/
├── unit/           # Fast, no network, full mocks
└── integration/    # Requires Ollama running (marked @pytest.mark.integration)
```

## Running tests

```bash
# Unit tests only (no Ollama needed)
pytest tests/unit/

# All tests including integration (requires Ollama)
pytest --run-integration

# With coverage
pytest tests/unit/ --cov=src/llm_benchmark --cov-report=html
```

## Adding a custom backend

Subclass `llm_benchmark.backends.base.BaseBackend` and implement `generate()`, `list_models()`, and `is_available()`. Register it in `backends/__init__.py`.

## License

MIT
