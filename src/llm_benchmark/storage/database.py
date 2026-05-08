"""SQLite persistence for benchmark runs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from ..config import Config, get_config
from ..metrics.types import BenchmarkResult, LatencyStats


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model       TEXT NOT NULL,
    backend     TEXT NOT NULL,
    bench_type  TEXT NOT NULL,
    prompt_set  TEXT NOT NULL,
    mean_tps    REAL,
    std_tps     REAL,
    min_tps     REAL,
    max_tps     REAL,
    ttft_p50_ms REAL,
    ttft_p95_ms REAL,
    ttft_p99_ms REAL,
    mean_rss_mb REAL,
    peak_rss_mb REAL,
    mean_metal_mb REAL,
    peak_metal_mb REAL,
    quality_score REAL,
    quality_details TEXT,
    run_count   INTEGER,
    timestamp   TEXT NOT NULL
);
"""


class Database:
    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg or get_config()
        self._path = self._cfg.db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, result: BenchmarkResult) -> int:
        lat = result.latency
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs
                    (model, backend, bench_type, prompt_set,
                     mean_tps, std_tps, min_tps, max_tps,
                     ttft_p50_ms, ttft_p95_ms, ttft_p99_ms,
                     mean_rss_mb, peak_rss_mb, mean_metal_mb, peak_metal_mb,
                     quality_score, quality_details, run_count, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    result.model,
                    result.backend,
                    result.benchmark_type,
                    result.prompt_set,
                    result.mean_tps,
                    result.std_tps,
                    result.min_tps,
                    result.max_tps,
                    lat.p50_ms if lat else None,
                    lat.p95_ms if lat else None,
                    lat.p99_ms if lat else None,
                    result.mean_rss_mb,
                    result.peak_rss_mb,
                    result.mean_metal_mb,
                    result.peak_metal_mb,
                    result.quality_score,
                    json.dumps(result.quality_details),
                    result.runs,
                    result.timestamp.isoformat(),
                ),
            )
            return int(cur.lastrowid)  # type: ignore[arg-type]

    def query(
        self,
        model: str | None = None,
        bench_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list = []
        if model:
            conditions.append("model = ?")
            params.append(model)
        if bench_type:
            conditions.append("bench_type = ?")
            params.append(bench_type)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM runs {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def all_models(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT model FROM runs ORDER BY model").fetchall()
        return [r["model"] for r in rows]
