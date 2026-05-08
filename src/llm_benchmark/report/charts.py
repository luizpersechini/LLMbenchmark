"""Matplotlib chart generation and React dashboard HTML report builder."""

from __future__ import annotations

import base64
import io
import json
import re
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from ..config import Config, get_config

# ---------------------------------------------------------------------------
# Matplotlib PNG charts (still used by generate_charts / legacy tests)
# ---------------------------------------------------------------------------


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
    ax.set_xticklabels([textwrap.fill(lbl, 14) for lbl in labels], fontsize=9)
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


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_charts(rows: list[dict[str, Any]], cfg: Config | None = None) -> list[Path]:
    """Generate PNG chart files. Returns list of generated paths."""
    cfg = cfg or get_config()
    out_dir = cfg.results_dir / "charts"
    out_dir.mkdir(exist_ok=True)
    generated: list[Path] = []

    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["bench_type"], []).append(r)

    speed_rows = by_type.get("speed", [])
    if speed_rows:
        labels = [r["model"] for r in speed_rows]
        values = [r["mean_tps"] or 0.0 for r in speed_rows]
        errors = [r["std_tps"] or 0.0 for r in speed_rows]
        p = out_dir / "speed_tps.png"
        _bar_chart("Throughput (tokens/sec)", labels, values, "tok/s", p, errors=errors)
        generated.append(p)

    latency_rows = by_type.get("latency", [])
    if latency_rows:
        labels = [r["model"] for r in latency_rows]
        p50 = [r["ttft_p50_ms"] or 0.0 for r in latency_rows]
        p95 = [r["ttft_p95_ms"] or 0.0 for r in latency_rows]
        p = out_dir / "latency_ttft.png"
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, p50, w, label="P50", color="#4C9BE8")
        ax.bar(x + w / 2, p95, w, label="P95", color="#F08030")
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(lbl, 14) for lbl in labels], fontsize=9)
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
        rss = [r["peak_rss_mb"] or 0.0 for r in memory_rows]
        metal = [r["peak_metal_mb"] or 0.0 for r in memory_rows]
        p = out_dir / "memory_rss.png"
        if any(m > 0 for m in metal):
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w / 2, rss, w, label="RSS", color="#68B068")
            ax.bar(x + w / 2, metal, w, label="Metal", color="#E87040")
            ax.set_xticks(x)
            ax.set_xticklabels([textwrap.fill(lbl, 14) for lbl in labels], fontsize=9)
            ax.set_ylabel("MB")
            ax.set_title("Peak Memory — RSS vs Metal GPU", fontsize=12, fontweight="bold")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            plt.close(fig)
        else:
            _bar_chart("Peak Process Memory (RSS)", labels, rss, "MB", p, color="#68B068")
        generated.append(p)

    quality_rows = by_type.get("quality", [])
    if quality_rows:
        labels = [r["model"] for r in quality_rows]
        scores = [(r["quality_score"] or 0.0) * 100 for r in quality_rows]
        p = out_dir / "quality_score.png"
        _bar_chart("Quality Score", labels, scores, "%", p, color="#C060C0")
        generated.append(p)

    return generated


def _charts_as_base64(rows: list[dict[str, Any]]) -> list[str]:
    """Build charts in memory and return list of base64 PNG data URIs."""
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["bench_type"], []).append(r)

    b64_images: list[str] = []

    speed_rows = by_type.get("speed", [])
    if speed_rows:
        labels = [r["model"] for r in speed_rows]
        values = [r["mean_tps"] or 0.0 for r in speed_rows]
        errors = [r["std_tps"] or 0.0 for r in speed_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color="#4C9BE8", width=0.5, yerr=errors, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(lbl, 14) for lbl in labels], fontsize=9)
        ax.set_ylabel("tok/s", fontsize=10)
        ax.set_title("Throughput (tokens/sec)", fontsize=12, fontweight="bold")
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
        b64_images.append(_fig_to_base64(fig))

    latency_rows = by_type.get("latency", [])
    if latency_rows:
        labels = [r["model"] for r in latency_rows]
        p50 = [r["ttft_p50_ms"] or 0.0 for r in latency_rows]
        p95 = [r["ttft_p95_ms"] or 0.0 for r in latency_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, p50, w, label="P50", color="#4C9BE8")
        ax.bar(x + w / 2, p95, w, label="P95", color="#F08030")
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(lbl, 14) for lbl in labels], fontsize=9)
        ax.set_ylabel("Time to first token (ms)")
        ax.set_title("TTFT Latency — P50 vs P95", fontsize=12, fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        b64_images.append(_fig_to_base64(fig))

    memory_rows = by_type.get("memory", [])
    if memory_rows:
        labels = [r["model"] for r in memory_rows]
        rss = [r["peak_rss_mb"] or 0.0 for r in memory_rows]
        metal = [r["peak_metal_mb"] or 0.0 for r in memory_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        if any(m > 0 for m in metal):
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w / 2, rss, w, label="RSS", color="#68B068")
            ax.bar(x + w / 2, metal, w, label="Metal", color="#E87040")
            ax.set_xticks(x)
            ax.set_xticklabels([textwrap.fill(lbl, 14) for lbl in labels], fontsize=9)
            ax.set_ylabel("MB")
            ax.set_title("Peak Memory — RSS vs Metal GPU", fontsize=12, fontweight="bold")
            ax.legend()
        else:
            x = np.arange(len(labels))
            bars = ax.bar(x, rss, color="#68B068", width=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([textwrap.fill(lbl, 14) for lbl in labels], fontsize=9)
            ax.set_ylabel("MB")
            ax.set_title("Peak Process Memory (RSS)", fontsize=12, fontweight="bold")
            for bar, val in zip(bars, rss):
                ax.annotate(
                    f"{val:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        b64_images.append(_fig_to_base64(fig))

    quality_rows = by_type.get("quality", [])
    if quality_rows:
        labels = [r["model"] for r in quality_rows]
        scores = [(r["quality_score"] or 0.0) * 100 for r in quality_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, scores, color="#C060C0", width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(lbl, 14) for lbl in labels], fontsize=9)
        ax.set_ylabel("%", fontsize=10)
        ax.set_title("Quality Score", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, scores):
            ax.annotate(
                f"{val:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        fig.tight_layout()
        b64_images.append(_fig_to_base64(fig))

    return b64_images


# ---------------------------------------------------------------------------
# Dashboard data helpers
# ---------------------------------------------------------------------------


def _model_family(model_id: str) -> str:
    base = model_id.lower().split(":")[0].split("/")[-1]
    patterns = [
        (r"codellama", "Code Llama"),
        (r"llama(\d+)\.(\d+)", lambda m: f"Llama {m.group(1)}.{m.group(2)}"),
        (r"llama(\d+)", lambda m: f"Llama {m.group(1)}"),
        (r"gemma(\d+)", lambda m: f"Gemma {m.group(1)}"),
        (r"qwen(\d+)\.(\d+)", lambda m: f"Qwen {m.group(1)}.{m.group(2)}"),
        (r"qwen(\d+)", lambda m: f"Qwen {m.group(1)}"),
        (r"phi(\d+)", lambda m: f"Phi-{m.group(1)}"),
        (r"mistral", "Mistral"),
        (r"mixtral", "Mixtral"),
        (r"deepseek", "DeepSeek"),
        (r"starcoder", "StarCoder"),
        (r"falcon", "Falcon"),
        (r"vicuna", "Vicuna"),
        (r"wizard", "WizardLM"),
        (r"orca", "Orca"),
        (r"nous", "Nous"),
    ]
    for pattern, name in patterns:
        m = re.search(pattern, base, re.IGNORECASE)
        if m:
            return name(m) if callable(name) else name
    return re.sub(r"[._-]", " ", base).title()


def _model_params(model_id: str) -> str:
    tag = model_id.split(":")[-1] if ":" in model_id else ""
    m = re.match(r"^(\d+\.?\d*)(b|m)$", tag, re.IGNORECASE)
    if m:
        val, unit = float(m.group(1)), m.group(2).upper()
        return f"{val:g} {unit}"
    m = re.search(r"[:\-_](\d+\.?\d*)[bB]", model_id)
    if m:
        return f"{float(m.group(1)):g} B"
    return "—"


def _model_quant(model_id: str, backend: str) -> str:
    if backend == "mlx":
        return "4-bit"
    return "Q4_K_M"


def _build_models_data(rows: list[dict]) -> list[dict]:
    """Aggregate DB rows into per-model summary objects for the dashboard."""
    # Keep only the latest row per (model, bench_type)
    latest: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("model", ""), r.get("bench_type", ""))
        ts = str(r.get("timestamp", ""))
        if key not in latest or ts > str(latest[key].get("timestamp", "")):
            latest[key] = r

    by_model: dict[str, dict] = {}
    for (model, bench_type), r in latest.items():
        if not model:
            continue
        if model not in by_model:
            backend = r.get("backend", "ollama") or "ollama"
            by_model[model] = {
                "id": model,
                "fam": _model_family(model),
                "params": _model_params(model),
                "backend": backend,
                "quant": _model_quant(model, backend),
                "tok_s": 0.0,
                "ttft_p50": 0,
                "ttft_p95": 0,
                "ttft_p99": 0,
                "rss": 0,
                "metal": 0,
                "qual": {"coding": 0, "reasoning": 0, "math": 0, "general": 0},
                "status": "done",
                "_quality_score": None,
            }
        m = by_model[model]
        if bench_type == "speed" and r.get("mean_tps"):
            m["tok_s"] = round(float(r["mean_tps"]), 1)
        elif bench_type == "latency":
            if r.get("ttft_p50_ms"):
                m["ttft_p50"] = int(round(float(r["ttft_p50_ms"])))
            if r.get("ttft_p95_ms"):
                m["ttft_p95"] = int(round(float(r["ttft_p95_ms"])))
            if r.get("ttft_p99_ms"):
                m["ttft_p99"] = int(round(float(r["ttft_p99_ms"])))
        elif bench_type == "memory":
            if r.get("peak_rss_mb"):
                m["rss"] = int(round(float(r["peak_rss_mb"])))
            if r.get("peak_metal_mb"):
                m["metal"] = int(round(float(r["peak_metal_mb"])))
        elif bench_type == "quality":
            if r.get("quality_score") is not None:
                m["_quality_score"] = float(r["quality_score"])
            details = r.get("quality_details")
            if details:
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except Exception:
                        details = {}
                per_cat = (details or {}).get("per_category", {})
                for cat in ("coding", "reasoning", "math", "general"):
                    if cat in per_cat:
                        m["qual"][cat] = int(round(float(per_cat[cat]) * 100))

    # Fill qual from overall quality_score when per_category is absent
    result = []
    for m in by_model.values():
        if all(v == 0 for v in m["qual"].values()) and m.get("_quality_score") is not None:
            score_int = int(round(m["_quality_score"] * 100))
            m["qual"] = {k: score_int for k in m["qual"]}
        del m["_quality_score"]
        result.append(m)

    return result


def _get_hardware() -> dict:
    try:
        from ..metrics.collector import system_info

        info = system_info()
        chip = info.get("chip", "Apple Silicon")
        ram = info.get("ram_total_gb", 0)
        cpu = info.get("cpu_count", "—")
        import platform

        os_name = platform.mac_ver()[0]
        os_str = f"macOS {os_name}" if os_name else info.get("platform", "macOS")[:40]
        return {
            "chip": chip,
            "cores": f"{cpu}-core CPU",
            "unified": f"{ram:.0f} GB" if ram else "— GB",
            "unifiedMB": int(ram * 1024) if ram else 0,
            "os": os_str,
        }
    except Exception:
        return {
            "chip": "Apple Silicon",
            "cores": "—",
            "unified": "— GB",
            "unifiedMB": 0,
            "os": "macOS",
        }


def _build_run_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "id": "run_—",
            "started": "—",
            "duration": "—",
            "totalRuns": 0,
            "warmup": 1,
            "promptSet": "general",
        }
    timestamps = sorted(str(r.get("timestamp", ""))[:19] for r in rows if r.get("timestamp"))
    latest_ts = timestamps[-1] if timestamps else ""
    try:
        dt = datetime.fromisoformat(latest_ts.replace("Z", "+00:00"))
        run_id = "run_" + dt.strftime("%Y-%m-%dT%H-%M-%SZ")
        started_fmt = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        run_id, started_fmt = "run_—", latest_ts

    prompt_sets = {r.get("prompt_set") for r in rows if r.get("prompt_set")} - {None, ""}
    prompt_set = list(prompt_sets)[0] if len(prompt_sets) == 1 else "mixed"

    speed_rows = [r for r in rows if r.get("bench_type") == "speed"]
    total_runs = len(speed_rows) * 3 if speed_rows else len(rows)

    return {
        "id": run_id,
        "started": started_fmt,
        "duration": "—",
        "totalRuns": total_runs,
        "warmup": 1,
        "promptSet": prompt_set,
    }


def _build_run_history(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    date_groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        date_key = str(r.get("timestamp", ""))[:10]
        if date_key:
            date_groups[date_key].append(r)

    history = []
    for date_str in sorted(date_groups.keys(), reverse=True)[:10]:
        group = date_groups[date_str]
        models = {r.get("model") for r in group if r.get("model")}
        speed_rows = [r for r in group if r.get("bench_type") == "speed" and r.get("mean_tps")]
        avg_tok = (
            sum(float(r["mean_tps"]) for r in speed_rows) / len(speed_rows) if speed_rows else 0.0
        )
        top_model = (
            max(speed_rows, key=lambda r: float(r["mean_tps"]))["model"]
            if speed_rows
            else (sorted(models)[0] if models else "—")
        )

        try:
            dt = datetime.fromisoformat(date_str)
            date_fmt = dt.strftime("%b ") + str(dt.day)
        except Exception:
            date_fmt = date_str

        timestamps = sorted(str(r.get("timestamp", ""))[:19] for r in group if r.get("timestamp"))
        dur_fmt = "—"
        if len(timestamps) >= 2:
            try:
                t0 = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
                secs = int(abs((t1 - t0).total_seconds()))
                dur_fmt = f"{secs // 60}m {secs % 60:02d}s"
            except Exception:
                pass

        run_id = (timestamps[0][:16] + "Z") if timestamps else date_str
        entry: dict[str, Any] = {
            "id": run_id,
            "date": date_fmt,
            "models": len(models),
            "dur": dur_fmt,
            "avg_tok": round(avg_tok, 1),
            "top": top_model,
        }
        history.append(entry)

    if history:
        history[0]["tag"] = "current"

    return history


def _pick_spotlight(models: list[dict]) -> str:
    if not models:
        return ""

    def _score(m: dict) -> tuple:
        has_qual = sum(1 for v in m["qual"].values() if v > 0)
        has_speed = 1 if m["tok_s"] > 0 else 0
        has_latency = 1 if m["ttft_p50"] > 0 else 0
        avg_qual = sum(m["qual"].values()) / 4
        completeness = has_qual * 10 + has_speed + has_latency
        return (completeness, avg_qual)

    return max(models, key=_score)["id"]


def _build_prompt_breakdown(models: list[dict], spotlight_id: str) -> list[dict]:
    m = next((x for x in models if x["id"] == spotlight_id), models[0] if models else None)
    if not m:
        return [
            {"set": "coding", "n": 20, "score": 0, "tok_s": 0.0},
            {"set": "reasoning", "n": 20, "score": 0, "tok_s": 0.0},
            {"set": "math", "n": 20, "score": 0, "tok_s": 0.0},
            {"set": "general", "n": 20, "score": 0, "tok_s": 0.0},
        ]
    q = m.get("qual", {})
    tok_s = float(m.get("tok_s") or 0.0)
    return [
        {"set": cat, "n": 20, "score": q.get(cat, 0), "tok_s": round(tok_s, 1)}
        for cat in ("coding", "reasoning", "math", "general")
    ]


def _build_mem_timeline(models: list[dict], spotlight_id: str) -> list[int]:
    """Synthesise a memory ramp-to-peak timeline from stored peak/mean."""
    m = next((x for x in models if x["id"] == spotlight_id), models[0] if models else None)
    if not m or not m.get("rss"):
        return []
    peak = int(m["rss"])
    ramp = [int(peak * i / 10) for i in range(1, 11)]
    noise = max(1, peak // 200)
    stable = [peak + (i % 3 - 1) * noise for i in range(20)]
    return ramp + stable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _safe(val: float | None, fmt: str, suffix: str = "") -> str:
    if val is None:
        return "-"
    return f"{val:{fmt}}{suffix}"


def generate_html_report(
    rows: list[dict[str, Any]],
    chart_paths: list[Path],
    cfg: Config | None = None,
) -> Path:
    cfg = cfg or get_config()

    models_data = _build_models_data(rows)
    hardware = _get_hardware()
    run_summary = _build_run_summary(rows)
    run_history = _build_run_history(rows)
    spotlight_id = _pick_spotlight(models_data)
    prompt_breakdown = _build_prompt_breakdown(models_data, spotlight_id)
    mem_timeline = _build_mem_timeline(models_data, spotlight_id)

    if not models_data:
        models_data = [
            {
                "id": "—",
                "fam": "—",
                "params": "—",
                "backend": "—",
                "quant": "—",
                "tok_s": 0.0,
                "ttft_p50": 0,
                "ttft_p95": 0,
                "ttft_p99": 0,
                "rss": 0,
                "metal": 0,
                "qual": {"coding": 0, "reasoning": 0, "math": 0, "general": 0},
                "status": "done",
            }
        ]
        spotlight_id = "—"

    data_js = "\n".join(
        [
            "// ─── DATA (from llm-bench SQLite results) ──────────────────────────────────",
            f"const HARDWARE = {json.dumps(hardware, indent=2)};",
            "",
            f"const RUN = {json.dumps(run_summary, indent=2)};",
            "",
            f"const MODELS = {json.dumps(models_data, indent=2)};",
            "",
            f"const RUN_HISTORY = {json.dumps(run_history, indent=2)};",
            "",
            f"const SPOTLIGHT_ID = {json.dumps(spotlight_id)};",
            "",
            f"const PROMPT_BREAKDOWN = {json.dumps(prompt_breakdown, indent=2)};",
            "",
            f"const MEM_TIMELINE = {json.dumps(mem_timeline)};",
        ]
    )

    template_path = Path(__file__).parent / "dashboard_template.html"
    template = template_path.read_text(encoding="utf-8")
    html = template.replace("// @@BENCHMARK_DATA@@", data_js)

    out = cfg.results_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
