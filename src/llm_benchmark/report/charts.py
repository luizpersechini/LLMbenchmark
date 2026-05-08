"""Matplotlib chart generation and HTML report builder."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from ..config import Config, get_config


def _bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    path: Path,
    color: str = "#4C9BE8",
    errors: list[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=color, width=0.5, yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(l, 14) for l in labels], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_charts(rows: list[dict[str, Any]], cfg: Config | None = None) -> list[Path]:
    """Generate PNG charts from raw DB rows. Returns list of generated paths."""
    cfg = cfg or get_config()
    out_dir = cfg.results_dir / "charts"
    out_dir.mkdir(exist_ok=True)
    generated: list[Path] = []

    # Separate by bench_type
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["bench_type"], []).append(r)

    speed_rows = by_type.get("speed", [])
    if speed_rows:
        labels = [r["model"] for r in speed_rows]
        values = [r["mean_tps"] or 0 for r in speed_rows]
        errors = [r["std_tps"] or 0 for r in speed_rows]
        p = out_dir / "speed_tps.png"
        _bar_chart("Throughput (tokens/sec)", labels, values, "tok/s", p, errors=errors)
        generated.append(p)

    latency_rows = by_type.get("latency", [])
    if latency_rows:
        labels = [r["model"] for r in latency_rows]
        p50 = [r["ttft_p50_ms"] or 0 for r in latency_rows]
        p95 = [r["ttft_p95_ms"] or 0 for r in latency_rows]
        p = out_dir / "latency_ttft.png"
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, p50, w, label="P50", color="#4C9BE8")
        ax.bar(x + w / 2, p95, w, label="P95", color="#F08030")
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(l, 14) for l in labels], fontsize=9)
        ax.set_ylabel("Time to first token (ms)")
        ax.set_title("TTFT Latency — P50 vs P95", fontsize=12, fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        generated.append(p)

    memory_rows = by_type.get("memory", [])
    if memory_rows:
        labels = [r["model"] for r in memory_rows]
        rss = [r["peak_rss_mb"] or 0 for r in memory_rows]
        p = out_dir / "memory_rss.png"
        _bar_chart("Peak Process Memory (RSS)", labels, rss, "MB", p, color="#68B068")
        generated.append(p)

    quality_rows = by_type.get("quality", [])
    if quality_rows:
        labels = [r["model"] for r in quality_rows]
        scores = [(r["quality_score"] or 0) * 100 for r in quality_rows]
        p = out_dir / "quality_score.png"
        _bar_chart("Quality Score", labels, scores, "%", p, color="#C060C0")
        generated.append(p)

    return generated


def generate_html_report(
    rows: list[dict[str, Any]],
    chart_paths: list[Path],
    cfg: Config | None = None,
) -> Path:
    cfg = cfg or get_config()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    charts_html = ""
    for cp in chart_paths:
        charts_html += f'<img src="charts/{cp.name}" style="max-width:720px;margin:16px 0;display:block">\n'

    rows_html = ""
    for r in rows:
        rows_html += f"""<tr>
            <td>{r['model']}</td>
            <td>{r['backend']}</td>
            <td>{r['bench_type']}</td>
            <td>{r['mean_tps']:.1f}</td>
            <td>{r['ttft_p50_ms'] or '-'}</td>
            <td>{r['peak_rss_mb']:.0f} MB</td>
            <td>{f"{(r['quality_score'] or 0)*100:.1f}%" if r['quality_score'] is not None else '-'}</td>
            <td>{r['timestamp'][:16]}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LLM Benchmark Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; color: #222; }}
  h1 {{ border-bottom: 2px solid #4C9BE8; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 24px 0; }}
  th {{ background: #4C9BE8; color: white; padding: 8px 12px; text-align: left; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f5f9ff; }}
  .section {{ margin-top: 40px; }}
  footer {{ color: #999; font-size: 12px; margin-top: 40px; }}
</style>
</head>
<body>
<h1>LLM Benchmark Report</h1>
<p>Generated: {ts} &mdash; {len(rows)} run(s)</p>
<div class="section">
<h2>Charts</h2>
{charts_html}
</div>
<div class="section">
<h2>Raw Results</h2>
<table>
<tr>
  <th>Model</th><th>Backend</th><th>Type</th>
  <th>Mean TPS</th><th>TTFT P50</th><th>Peak RSS</th>
  <th>Quality</th><th>Timestamp</th>
</tr>
{rows_html}
</table>
</div>
<footer>llm-benchmark &mdash; Apple Silicon M-series</footer>
</body>
</html>"""

    out = cfg.results_dir / "report.html"
    out.write_text(html)
    return out
