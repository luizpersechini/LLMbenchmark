# llm-benchmark — Local LLM benchmarking on Apple Silicon

## Stack
Python 3.10+ · Click CLI · Ollama (HTTP) + MLX (native) backends · matplotlib · pandas · pydantic · SQLite

## Run
```
pip install -e ".[dev]"     # one-time setup
llm-bench run               # benchmark all configured models
llm-bench results           # query past runs from SQLite
llm-bench report            # generate HTML report → results/
```

## Test
```
pytest                      # unit + integration tests
pytest tests/unit -q        # unit only (fast)
```

## Structure
- `src/llm_benchmark/` — package source (CLI, backends, metrics, storage)
- `prompts/` — task prompt sets (coding, math, reasoning, general) as JSONL
- `tests/unit/` — pure-logic tests
- `tests/integration/` — backend-dependent tests
- `results/` — generated HTML reports + SQLite DB, gitignored

## What it measures
Tokens/sec sustained, time-to-first-token (P50/P95/P99), process RSS + Metal GPU memory, scored quality on reasoning/coding/math prompts.

## Don't touch
- `results/*.db` — persisted run history; don't truncate
- `prompts/*.jsonl` — scoring depends on exact prompts; propose plan before editing

## Hardware
Designed for Apple Silicon (M1–M4) with unified memory. MLX backend requires `mlx-lm` (install via `pip install -e ".[mlx]"`).
