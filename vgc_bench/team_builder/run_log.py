"""
Shared helpers for the run_meta.json / live_status.json files a team-builder run
writes to its output directory.

These are the only files that let the viz dashboard tell a live run from a
finished one and follow it step-by-step, independent of which oracle (RL or
evolutionary) is producing it:

* ``run_meta.json`` — written once at run start (run type, regulation, target
  iterations, cadence). Never overwritten on resume, so ``started_at`` sticks.
* ``live_status.json`` — overwritten atomically every step/round with the run's
  current point-in-time state (phase, iteration, step, win rates, the team
  currently/most-recently generated).

Both are plain JSON, written via the same atomic tmp-file + replace idiom
already used for ``psro_checkpoint.json`` and ``policy.pt``, so a concurrent
reader never observes a partial file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON to path via a temp file + atomic replace."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_run_meta(run_dir: Path, **fields) -> None:
    """
    Write run_meta.json once, at run start.

    A no-op if the file already exists, so resuming a run preserves the
    original ``started_at`` and launch configuration.
    """
    path = run_dir / "run_meta.json"
    if path.exists():
        return
    payload = {"started_at": time.time(), **fields}
    write_json_atomic(path, payload)


def write_live_status(run_dir: Path, **fields) -> None:
    """Overwrite live_status.json with the run's current state."""
    path = run_dir / "live_status.json"
    payload = {"updated_at": time.time(), **fields}
    write_json_atomic(path, payload)
