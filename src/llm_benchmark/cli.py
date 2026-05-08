"""Click CLI entry point for llm-bench."""

from __future__ import annotations

import json
import subprocess
from typing import Any

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from .backends import get_backend
from .benchmarks import LatencyBenchmark, MemoryBenchmark, QualityBenchmark, SpeedBenchmark
from .config import get_config
from .metrics.types import BenchmarkResult
from .report import (
    generate_charts,
    generate_html_report,
    print_comparison_table,
    print_result,
)
from .storage import Database

console = Console()
cfg = get_config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_all_benchmarks(
    backend_name: str,
    model: str,
    runs: int,
    warmup: int,
    max_tokens: int,
    prompt_set: str,
    skip_quality: bool,
) -> list[BenchmarkResult]:
    backend = get_backend(backend_name)
    if not backend.is_available():
        console.print(f"[red]Backend '{backend_name}' is not available.[/]")
        raise SystemExit(1)

    db = Database()
    results: list[BenchmarkResult] = []

    benchmarks: list[tuple[str, Any]] = [
        ("Speed", SpeedBenchmark()),
        ("Latency", LatencyBenchmark()),
        ("Memory", MemoryBenchmark()),
    ]
    if not skip_quality:
        benchmarks.append(("Quality", QualityBenchmark()))

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for label, bench in benchmarks:
            task = progress.add_task(f"[cyan]{label}[/] {model} ...", total=None)

            def _on_progress(current: int, total: int, _task=task) -> None:
                progress.update(_task, description=f"[cyan]{label}[/] {model} ({current}/{total})")

            kwargs: dict[str, Any] = dict(
                backend=backend,
                model=model,
                runs=runs,
                warmup=warmup,
                on_progress=_on_progress,
            )
            if label == "Speed":
                kwargs["max_tokens"] = max_tokens
            elif label == "Latency":
                pass
            elif label == "Memory":
                kwargs["max_tokens"] = max_tokens
            elif label == "Quality":
                kwargs = dict(
                    backend=backend, model=model, prompt_set=prompt_set, on_progress=_on_progress
                )
                del kwargs["runs"]
                del kwargs["warmup"]

            result = bench.run(**kwargs)
            progress.update(task, completed=True)
            results.append(result)
            db.save(result)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
def main() -> None:
    """llm-bench — local LLM benchmarking for Apple Silicon."""


@main.command()
@click.option("--backend", default="ollama", show_default=True, help="Backend: ollama or mlx")
def list(backend: str) -> None:
    """List available models from a backend."""
    b = get_backend(backend)
    if not b.is_available():
        console.print(f"[red]{backend} is not available. Is it running?[/]")
        raise SystemExit(1)
    models = b.list_models()
    if not models:
        console.print("[yellow]No models found.[/]")
        return
    console.print(f"\n[bold]{backend}[/] — {len(models)} model(s):\n")
    for m in models:
        console.print(f"  {m}")
    console.print()


@main.command()
@click.argument("model")
@click.option("--backend", default="ollama", show_default=True)
@click.option("--runs", default=cfg.default_runs, show_default=True, type=int)
@click.option("--warmup", default=cfg.default_warmup, show_default=True, type=int)
@click.option("--max-tokens", default=cfg.default_max_tokens, show_default=True, type=int)
@click.option("--prompt-set", default=cfg.default_prompt_set, show_default=True)
@click.option("--skip-quality", is_flag=True, default=False)
@click.option("--output", default="table", type=click.Choice(["table", "json"]), show_default=True)
def run(
    model: str,
    backend: str,
    runs: int,
    warmup: int,
    max_tokens: int,
    prompt_set: str,
    skip_quality: bool,
    output: str,
) -> None:
    """Benchmark a single MODEL."""
    results = _run_all_benchmarks(
        backend, model, runs, warmup, max_tokens, prompt_set, skip_quality
    )

    if output == "json":
        data = [_result_to_dict(r) for r in results]
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        for r in results:
            print_result(r)


@main.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--backend", default="ollama", show_default=True)
@click.option("--runs", default=cfg.default_runs, show_default=True, type=int)
@click.option("--warmup", default=cfg.default_warmup, show_default=True, type=int)
@click.option("--max-tokens", default=cfg.default_max_tokens, show_default=True, type=int)
@click.option("--prompt-set", default=cfg.default_prompt_set, show_default=True)
@click.option("--skip-quality", is_flag=True, default=False)
def compare(
    models: tuple[str, ...],
    backend: str,
    runs: int,
    warmup: int,
    max_tokens: int,
    prompt_set: str,
    skip_quality: bool,
) -> None:
    """Benchmark and compare multiple MODELS side-by-side."""
    all_results: list[BenchmarkResult] = []
    for model in models:
        console.rule(f"[bold]{model}[/]")
        results = _run_all_benchmarks(
            backend, model, runs, warmup, max_tokens, prompt_set, skip_quality
        )
        all_results.extend(results)

    # Group by bench_type for comparison tables
    by_type: dict[str, list[BenchmarkResult]] = {}
    for r in all_results:
        by_type.setdefault(r.benchmark_type, []).append(r)

    for bench_type, group in by_type.items():
        print_comparison_table(group)


@main.command()
@click.option("--model", default=None)
@click.option("--type", "bench_type", default=None, help="Filter by bench type")
@click.option("--last", default=20, show_default=True, type=int)
@click.option("--output", default="table", type=click.Choice(["table", "json", "csv"]))
def results(model: str | None, bench_type: str | None, last: int, output: str) -> None:
    """Query stored benchmark results."""
    db = Database()
    rows = db.query(model=model, bench_type=bench_type, limit=last)
    if not rows:
        console.print("[yellow]No results found.[/]")
        return

    if output == "json":
        click.echo(json.dumps(rows, indent=2, default=str))
        return

    if output == "csv":
        import csv
        import io

        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
        click.echo(buf.getvalue())
        return

    # table
    from rich import box
    from rich.table import Table

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    for col in [
        "model",
        "backend",
        "bench_type",
        "mean_tps",
        "ttft_p50_ms",
        "peak_rss_mb",
        "quality_score",
        "timestamp",
    ]:
        t.add_column(col, overflow="fold")
    for r in rows:
        t.add_row(
            r.get("model", ""),
            r.get("backend", ""),
            r.get("bench_type", ""),
            f"{r['mean_tps']:.1f}" if r.get("mean_tps") else "-",
            f"{r['ttft_p50_ms']:.0f} ms" if r.get("ttft_p50_ms") else "-",
            f"{r['peak_rss_mb']:.0f} MB" if r.get("peak_rss_mb") else "-",
            f"{r['quality_score'] * 100:.1f}%" if r.get("quality_score") else "-",
            str(r.get("timestamp", ""))[:16],
        )
    console.print(t)


@main.command()
@click.option("--last", default=100, show_default=True, type=int, help="Rows to include")
@click.option("--open", "open_report", is_flag=True, default=False, help="Open in browser")
def report(last: int, open_report: bool) -> None:
    """Generate an HTML report with charts from stored results."""
    db = Database()
    rows = db.query(limit=last)
    if not rows:
        console.print("[yellow]No results to report on.[/]")
        return

    console.print("Generating charts...")
    chart_paths = generate_charts(rows)
    html_path = generate_html_report(rows, chart_paths)
    console.print(f"[green]Report saved:[/] {html_path}")
    for cp in chart_paths:
        console.print(f"  chart: {cp}")

    if open_report:
        subprocess.run(["open", str(html_path)], check=False)


# ---------------------------------------------------------------------------


def _result_to_dict(r: BenchmarkResult) -> dict:
    return {
        "model": r.model,
        "backend": r.backend,
        "benchmark_type": r.benchmark_type,
        "mean_tps": r.mean_tps,
        "std_tps": r.std_tps,
        "latency_p50_ms": r.latency.p50_ms if r.latency else None,
        "latency_p95_ms": r.latency.p95_ms if r.latency else None,
        "peak_rss_mb": r.peak_rss_mb,
        "peak_metal_mb": r.peak_metal_mb,
        "quality_score": r.quality_score,
        "runs": r.runs,
        "timestamp": r.timestamp.isoformat(),
    }
