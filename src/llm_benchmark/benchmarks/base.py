"""Shared helpers used by all benchmark types."""

from __future__ import annotations

import time
from typing import Callable

import psutil


def timed_call(fn: Callable, *args, **kwargs):
    """Call fn and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def peak_rss_mb(pid: int | None = None) -> float:
    proc = psutil.Process(pid)
    return proc.memory_info().rss / (1024 * 1024)
