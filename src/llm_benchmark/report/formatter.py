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
    if val is None:
        return "-"
    return f"{val:.{decimals}f}{unit}"


def print_result(result: BenchmarkResult) -> None:
    ts = result.timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(result.timestamp, datetime) else str(result.timestamp)
    console.print(f"\n[bold cyan]{result.model}[/] ({result.backend}) — {result.benchmark_type.upper()}  [dim]{ts}[/]")

    t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    t.add_column("Metric", style="dim")
    t.add_column("Value", justify="right")

    if result.benchmark_type == "speed":
        t.add_row("Mean TPS", _fmt(result.mean_tps, 1, " tok/s"))
        t.add_row("Std TPS", _fmt(result.std_tps, 2, " tok/s"))
        t.add_row("Min TPS", _fmt(result.min_tps, 1, " tok/s"))
        t.add_row("Max TPS", _fmt(result.max_tps, 1, " tok/s"))
        t.add_row("Peak RSS", _fmt(result.peak_rss_mb, 0, " MB"))

    elif result.benchmark_type == "latency":
        lat = result.latency
        if lat:
            t.add_row("TTFT P50", _fmt(lat.p50_ms, 0, " ms"))
            t.add_row("TTFT P95", _fmt(lat.p95_ms, 0, " ms"))
            t.add_row("TTFT P99", _fmt(lat.p99_ms, 0, " ms"))
            t.add_row("Mean TTFT", _fmt(lat.mean_ms, 0, " ms"))
            t.add_row("Min TTFT", _fmt(lat.min_ms, 0, " ms"))
        t.add_row("Peak RSS", _fmt(result.peak_rss_mb, 0, " MB"))

    elif result.benchmark_type == "memory":
        t.add_row("Mean RSS", _fmt(result.mean_rss_mb, 0, " MB"))
        t.add_row("Peak RSS", _fmt(result.peak_rss_mb, 0, " MB"))
        t.add_row("Mean Metal", _fmt(result.mean_metal_mb, 0, " MB"))
        t.add_row("Peak Metal", _fmt(result.peak_metal_mb, 0, " MB"))

    elif result.benchmark_type == "quality":
        t.add_row("Overall Score", _fmt((result.quality_score or 0) * 100, 1, "%"))
        for cat, score in (result.quality_details.get("per_category") or {}).items():
            t.add_row(f"  {cat}", _fmt(score * 100, 1, "%"))
        t.add_row("Tasks Run", str(result.runs))

    console.print(t)


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    if not results:
        console.print("[yellow]No results to compare.[/]")
        return

    bench_type = results[0].benchmark_type
    console.print(f"\n[bold]Side-by-side comparison — {bench_type.upper()}[/]\n")

    t = Table(box=box.MARKDOWN, show_header=True, header_style="bold cyan")
    t.add_column("Model")
    t.add_column("Backend")

    if bench_type == "speed":
        t.add_column("Mean TPS", justify="right")
        t.add_column("Std", justify="right")
        t.add_column("Min", justify="right")
        t.add_column("Max", justify="right")
        t.add_column("Peak RSS", justify="right")
        for r in results:
            t.add_row(
                r.model,
                r.backend,
                _fmt(r.mean_tps, 1),
                _fmt(r.std_tps, 2),
                _fmt(r.min_tps, 1),
                _fmt(r.max_tps, 1),
                _fmt(r.peak_rss_mb, 0, " MB"),
            )

    elif bench_type == "latency":
        t.add_column("P50 TTFT", justify="right")
        t.add_column("P95 TTFT", justify="right")
        t.add_column("P99 TTFT", justify="right")
        t.add_column("Mean TTFT", justify="right")
        for r in results:
            lat = r.latency
            t.add_row(
                r.model,
                r.backend,
                _fmt(lat.p50_ms if lat else None, 0, " ms"),
                _fmt(lat.p95_ms if lat else None, 0, " ms"),
                _fmt(lat.p99_ms if lat else None, 0, " ms"),
                _fmt(lat.mean_ms if lat else None, 0, " ms"),
            )

    elif bench_type == "memory":
        t.add_column("Mean RSS", justify="right")
        t.add_column("Peak RSS", justify="right")
        t.add_column("Peak Metal", justify="right")
        for r in results:
            t.add_row(
                r.model,
                r.backend,
                _fmt(r.mean_rss_mb, 0, " MB"),
                _fmt(r.peak_rss_mb, 0, " MB"),
                _fmt(r.peak_metal_mb, 0, " MB"),
            )

    elif bench_type == "quality":
        t.add_column("Score", justify="right")
        t.add_column("Tasks", justify="right")
        for r in results:
            t.add_row(
                r.model,
                r.backend,
                _fmt((r.quality_score or 0) * 100, 1, "%"),
                str(r.runs),
            )

    console.print(t)
