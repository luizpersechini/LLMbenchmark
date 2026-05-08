"""Rich terminal tables for benchmark results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table
from rich import box

from ..metrics.types import BenchmarkResult

console = Console()


def _fmt(val: float | None, decimals: int = 1, unit: str = "") -> str:
    if val is None or val == 0.0:
        return "-"
    return f"{val:.{decimals}f}{unit}"


def _best(values: list[float | None], higher_is_better: bool = True) -> int | None:
    """Return index of the best non-None value, or None if all are missing."""
    filtered = [(i, v) for i, v in enumerate(values) if v is not None and v != 0.0]
    if not filtered:
        return None
    return max(filtered, key=lambda x: x[1] if higher_is_better else -x[1])[0]


def _highlight(s: str, is_best: bool) -> str:
    return f"[bold green]{s}[/]" if is_best else s


def _print_quality_tasks(tasks: list[dict]) -> None:
    """Print a per-task quality breakdown table."""
    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    t.add_column("ID", style="dim")
    t.add_column("Category", style="dim")
    t.add_column("Score", justify="right")
    t.add_column("Output snippet")
    for task in tasks:
        score = task.get("score", 0.0)
        if score >= 1.0:
            score_str = "[green]PASS[/]"
        elif score > 0:
            score_str = f"[yellow]{score*100:.0f}%[/]"
        else:
            score_str = "[red]FAIL[/]"
        t.add_row(
            str(task.get("id", "")),
            task.get("category", ""),
            score_str,
            str(task.get("output_snippet", ""))[:70],
        )
    console.print(t)


def print_result(result: BenchmarkResult, verbose: bool = False) -> None:
    ts = (
        result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(result.timestamp, datetime)
        else str(result.timestamp)
    )
    console.print(
        f"\n[bold cyan]{result.model}[/] ({result.backend})"
        f" — {result.benchmark_type.upper()}  [dim]{ts}[/]"
    )

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    t.add_column("Metric", style="dim")
    t.add_column("Value", justify="right")

    if result.benchmark_type == "speed":
        t.add_row("Mean TPS", _fmt(result.mean_tps, 1, " tok/s"))
        t.add_row("Std TPS", _fmt(result.std_tps, 2, " tok/s"))
        t.add_row("Min / Max TPS", f"{_fmt(result.min_tps, 1)} / {_fmt(result.max_tps, 1)}")
        t.add_row("Peak RSS", _fmt(result.peak_rss_mb, 0, " MB"))
        if result.peak_metal_mb:
            t.add_row("Peak Metal", _fmt(result.peak_metal_mb, 0, " MB"))

    elif result.benchmark_type == "latency":
        lat = result.latency
        if lat:
            t.add_row("TTFT P50", _fmt(lat.p50_ms, 0, " ms"))
            t.add_row("TTFT P95", _fmt(lat.p95_ms, 0, " ms"))
            t.add_row("TTFT P99", _fmt(lat.p99_ms, 0, " ms"))
            t.add_row("Mean TTFT", _fmt(lat.mean_ms, 0, " ms"))
            t.add_row("Min / Max", f"{_fmt(lat.min_ms, 0)} / {_fmt(lat.max_ms, 0)} ms")
        t.add_row("Peak RSS", _fmt(result.peak_rss_mb, 0, " MB"))

    elif result.benchmark_type == "memory":
        t.add_row("Mean RSS", _fmt(result.mean_rss_mb, 0, " MB"))
        t.add_row("Peak RSS", _fmt(result.peak_rss_mb, 0, " MB"))
        if result.peak_metal_mb:
            t.add_row("Mean Metal", _fmt(result.mean_metal_mb, 0, " MB"))
            t.add_row("Peak Metal", _fmt(result.peak_metal_mb, 0, " MB"))

    elif result.benchmark_type == "quality":
        t.add_row("Overall Score", _fmt((result.quality_score or 0) * 100, 1, "%"))
        per_cat = (result.quality_details.get("per_category") or {})
        for cat, score in sorted(per_cat.items()):
            t.add_row(f"  {cat}", _fmt(score * 100, 1, "%"))
        t.add_row("Tasks Run", str(result.runs))
        t.add_row("Prompt Set", result.prompt_set)
        console.print(t)
        if verbose:
            tasks = (result.quality_details or {}).get("tasks", [])
            if tasks:
                _print_quality_tasks(tasks)
        return

    console.print(t)


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    if not results:
        console.print("[yellow]No results to compare.[/]")
        return

    bench_type = results[0].benchmark_type
    console.print(f"\n[bold]Side-by-side — {bench_type.upper()}[/]  "
                  f"[dim]({len(results)} model(s))[/]\n")

    t = Table(box=box.MARKDOWN, show_header=True, header_style="bold cyan")
    t.add_column("Model")
    t.add_column("Backend")

    if bench_type == "speed":
        cols = ["Mean TPS", "Std", "Min", "Max", "Peak RSS", "Peak Metal"]
        t.add_column("Mean TPS", justify="right")
        t.add_column("Std", justify="right")
        t.add_column("Min", justify="right")
        t.add_column("Max", justify="right")
        t.add_column("Peak RSS", justify="right")
        t.add_column("Peak Metal", justify="right")

        mean_tps = [r.mean_tps for r in results]
        peak_rss = [r.peak_rss_mb for r in results]
        best_tps = _best(mean_tps, higher_is_better=True)
        best_rss = _best(peak_rss, higher_is_better=False)

        for i, r in enumerate(results):
            t.add_row(
                r.model, r.backend,
                _highlight(_fmt(r.mean_tps, 1), i == best_tps),
                _fmt(r.std_tps, 2),
                _fmt(r.min_tps, 1),
                _fmt(r.max_tps, 1),
                _highlight(_fmt(r.peak_rss_mb, 0, " MB"), i == best_rss),
                _fmt(r.peak_metal_mb, 0, " MB") if r.peak_metal_mb else "-",
            )

    elif bench_type == "latency":
        t.add_column("P50 TTFT", justify="right")
        t.add_column("P95 TTFT", justify="right")
        t.add_column("P99 TTFT", justify="right")
        t.add_column("Mean TTFT", justify="right")

        p50s = [r.latency.p50_ms if r.latency else None for r in results]
        best_p50 = _best(p50s, higher_is_better=False)

        for i, r in enumerate(results):
            lat = r.latency
            t.add_row(
                r.model, r.backend,
                _highlight(_fmt(lat.p50_ms if lat else None, 0, " ms"), i == best_p50),
                _fmt(lat.p95_ms if lat else None, 0, " ms"),
                _fmt(lat.p99_ms if lat else None, 0, " ms"),
                _fmt(lat.mean_ms if lat else None, 0, " ms"),
            )

    elif bench_type == "memory":
        t.add_column("Mean RSS", justify="right")
        t.add_column("Peak RSS", justify="right")
        t.add_column("Peak Metal", justify="right")

        peak_rss = [r.peak_rss_mb for r in results]
        best_rss = _best(peak_rss, higher_is_better=False)

        for i, r in enumerate(results):
            t.add_row(
                r.model, r.backend,
                _fmt(r.mean_rss_mb, 0, " MB"),
                _highlight(_fmt(r.peak_rss_mb, 0, " MB"), i == best_rss),
                _fmt(r.peak_metal_mb, 0, " MB") if r.peak_metal_mb else "-",
            )

    elif bench_type == "quality":
        t.add_column("Score", justify="right")
        t.add_column("Tasks", justify="right")
        t.add_column("Prompt Set")

        scores = [r.quality_score for r in results]
        best_score = _best(scores, higher_is_better=True)

        for i, r in enumerate(results):
            t.add_row(
                r.model, r.backend,
                _highlight(_fmt((r.quality_score or 0) * 100, 1, "%"), i == best_score),
                str(r.runs),
                r.prompt_set,
            )

    console.print(t)
