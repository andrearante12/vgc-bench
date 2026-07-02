"""
Run browser: a cheap summary of every run directory under results/, for the
dashboard sidebar. Deliberately avoids parsing snapshots/team text — only the
small run_meta.json / live_status.json / psro_checkpoint.json files are read, so
scanning many runs stays fast even as a run's snapshot history grows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from vgc_bench.team_builder.viz.loader import (
    _max_mtime,
    _read_json_optional,
    compute_status,
    detect_run_type,
    list_runs,
)


@dataclass(frozen=True)
class RunSummary:
    """A lightweight, cheap-to-compute summary of one run directory."""

    name: str
    path: Path
    run_type: str  # "rl" | "evo"
    reg: str | None
    completed_iterations: int | None
    target_iterations: int | None
    status: str  # "live" | "done" | "interrupted" | "idle"
    best_win_rate: float | None
    last_activity: float


def _best_win_rate(live: dict | None, checkpoint: dict | None) -> float | None:
    """Cheapest available signal for 'how good is this run's best team so far'."""
    if live is not None and live.get("best_win_rate") is not None:
        return live["best_win_rate"]
    if checkpoint is not None:
        found = checkpoint.get("found_teams") or []
        rates = [t.get("win_rate") for t in found if t.get("win_rate") is not None]
        if rates:
            return max(rates)
    return None


def summarize_run(run_dir: Path) -> RunSummary:
    """Build a RunSummary for one run directory without parsing any snapshots."""
    run_type = detect_run_type(run_dir)
    meta = _read_json_optional(run_dir / "run_meta.json")
    live = _read_json_optional(run_dir / "live_status.json")
    checkpoint = _read_json_optional(run_dir / "psro_checkpoint.json")
    last_activity = live["updated_at"] if live else _max_mtime(run_dir)
    status = compute_status(run_dir, meta, live, checkpoint, last_activity, time.time())

    return RunSummary(
        name=run_dir.name,
        path=run_dir,
        run_type=run_type,
        reg=meta.get("reg") if meta else None,
        completed_iterations=checkpoint.get("completed_iterations")
        if checkpoint
        else None,
        target_iterations=meta.get("target_iterations") if meta else None,
        status=status,
        best_win_rate=_best_win_rate(live, checkpoint),
        last_activity=last_activity,
    )


def scan_runs(results_dir: str | Path = "results") -> list[RunSummary]:
    """
    Summarize every run directory under results_dir, most recently active first.

    Live runs always sort to the top regardless of clock skew between the run's
    last_activity timestamp and any other run's.
    """
    summaries = [summarize_run(d) for d in list_runs(results_dir)]
    summaries.sort(key=lambda s: (s.status != "live", -s.last_activity))
    return summaries
