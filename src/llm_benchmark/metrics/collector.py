"""System metric collection — CPU, RAM, and Metal GPU memory on Apple Silicon."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import psutil


@dataclass
class SystemMetrics:
    rss_mb: float  # process resident set size
    vms_mb: float  # process virtual memory
    cpu_percent: float
    metal_mb: float  # Metal GPU allocated (0 on non-macOS)
    system_ram_used_mb: float
    system_ram_total_mb: float


def _metal_memory_mb() -> float:
    """Read Metal GPU allocated memory via ioreg (macOS only, no sudo needed)."""
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-n", "AGXA"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # ioreg output isn't JSON; parse the key we need
        for line in result.stdout.splitlines():
            if "PerformanceStatistics" in line:
                # skip — expensive nested dict
                continue
            if "Allocated System Memory" in line or "AllocatedSystemMemory" in line:
                parts = line.split("=")
                if len(parts) == 2:
                    val = parts[1].strip().rstrip("}")
                    try:
                        return int(val) / (1024 * 1024)
                    except ValueError:
                        pass
    except Exception:
        pass

    # Fallback: query via system_profiler (slower but more reliable)
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        displays = data.get("SPDisplaysDataType", [])
        for d in displays:
            vram = d.get("sppci_vram_shared", "") or d.get("sppci_vram", "")
            if vram:
                # value like "24576 MB"
                try:
                    return float(vram.split()[0])
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass

    return 0.0


def collect_system_metrics(pid: int | None = None) -> SystemMetrics:
    proc = psutil.Process(pid) if pid else psutil.Process()
    mi = proc.memory_info()
    vm = psutil.virtual_memory()
    return SystemMetrics(
        rss_mb=mi.rss / (1024 * 1024),
        vms_mb=mi.vms / (1024 * 1024),
        cpu_percent=proc.cpu_percent(interval=0.1),
        metal_mb=_metal_memory_mb(),
        system_ram_used_mb=vm.used / (1024 * 1024),
        system_ram_total_mb=vm.total / (1024 * 1024),
    )
