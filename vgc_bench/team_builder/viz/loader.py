"""
Discover and parse a team-builder run directory into a shared display model.

Handles both run types with one loader:

* **RL SP-PSRO** (``type == "sp_psro"``): per-iteration ``beta_iter{NNN}/`` and
  ``nu_iter{NNN}/`` phase dirs, each with ``snapshots/step_*.txt`` + ``metrics.jsonl``;
  the checkpoint carries aligned ``population_texts`` + ``nash_w``.
* **Evolutionary** (``type == "psro"``): per-iteration ``search_iter{NNN}/`` dirs with
  ``round_*.txt`` + ``rounds.jsonl``; Nash weights are recomputed from the payoff.

Pure Python (no Streamlit) so it can be unit-tested without the UI stack. Caching is
applied at the app layer.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vgc_bench.team_builder.viz.parse import TeamView, parse_header, parse_team

_ITER_RE = re.compile(r"iter(\d+)")
_STEP_FILE_RE = re.compile(r"step_(\d+)\.txt")
_ROUND_FILE_RE = re.compile(r"round_(\d+)\.txt")

# A run counts as "live" if its most recent activity (live_status.json's updated_at,
# or failing that the newest file mtime under the run dir) is within this window.
LIVE_FRESHNESS_SECONDS = 90.0


@dataclass(frozen=True)
class Frame:
    """One point on the best-team-evolution timeline."""

    order: int  # global position in the run timeline
    phase: str  # e.g. "iter0 · β" or "iter0 search"
    step: int  # step (RL) or round (evo) within the phase
    win_rate: float | None
    team: TeamView
    species: tuple[str, ...]  # the frame team's species, for churn tracking


@dataclass(frozen=True)
class PhaseTimeline:
    """A training phase (β / ν / a search iteration) and its ordered frames."""

    name: str
    kind: str  # "beta" | "nu" | "search"
    iteration: int
    frames: tuple[Frame, ...]
    metrics: tuple[dict, ...] = ()  # raw rows from metrics.jsonl / rounds.jsonl


@dataclass(frozen=True)
class IterationView:
    """A PSRO iteration's payoff matrix and derived Nash weights."""

    index: int
    payoff: np.ndarray
    nash_weights: list[float]


@dataclass(frozen=True)
class RunData:
    """Everything the dashboard needs about one run."""

    run_dir: Path
    run_type: str  # "rl" | "evo"
    phases: tuple[PhaseTimeline, ...]
    iterations: tuple[IterationView, ...]
    final_teams: tuple[
        tuple[TeamView, float | None], ...
    ]  # (team, nash weight), ranked
    meta: dict | None = None  # run_meta.json, if present
    live: dict | None = None  # live_status.json, if present
    status: str = "idle"  # "live" | "done" | "interrupted" | "idle"
    last_activity: float = 0.0  # epoch seconds of the run's most recent write

    @property
    def frames(self) -> list[Frame]:
        """All frames across phases in run order."""
        return [f for p in self.phases for f in p.frames]


def _nash_weights(payoff: np.ndarray) -> list[float]:
    """
    Nash equilibrium distribution over payoff rows. Mirrors
    ``vgc_bench.team_builder.psro._nash_weights`` but imported locally to avoid pulling
    in the battle/torch stack. Falls back to uniform on a singular game.
    """
    try:
        import nashpy

        return nashpy.Game(payoff).linear_program()[0].tolist()
    except Exception:
        n = payoff.shape[0]
        return [1.0 / n] * n if n else []


def _read_json_optional(path: Path) -> dict | None:
    """Read a JSON file, returning None if missing or unparsable (e.g. mid-write)."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _max_mtime(run_dir: Path) -> float:
    """Newest mtime under the run dir; 0.0 if the dir is empty or unreadable."""
    latest = 0.0
    for p in run_dir.rglob("*"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            continue
    return latest


def _any_interrupted(run_dir: Path) -> bool:
    return next(run_dir.rglob("interrupted_best_team.txt"), None) is not None


def compute_status(
    run_dir: Path,
    meta: dict | None,
    live: dict | None,
    checkpoint: dict | None,
    last_activity: float,
    now: float,
    freshness_seconds: float = LIVE_FRESHNESS_SECONDS,
) -> str:
    """
    Resolve a run's status: live > done > interrupted > idle.

    There is no persisted "training process is still running" flag, so liveness is
    inferred from recent activity. "done" is inferred from either the replays/ dir
    (only written at the very end of a successful run) or completed_iterations
    reaching the target recorded in run_meta.json.
    """
    if now - last_activity <= freshness_seconds:
        return "live"
    if (run_dir / "replays").exists():
        return "done"
    if meta is not None and checkpoint is not None:
        target = meta.get("target_iterations")
        completed = checkpoint.get("completed_iterations")
        if target is not None and completed is not None and completed >= target:
            return "done"
    if _any_interrupted(run_dir):
        return "interrupted"
    return "idle"


def detect_run_type(run_dir: Path) -> str:
    """Return 'rl' or 'evo' from the checkpoint type, falling back to dir globs."""
    ckpt = run_dir / "psro_checkpoint.json"
    if ckpt.exists():
        try:
            t = json.loads(ckpt.read_text(encoding="utf-8")).get("type", "")
            if t == "sp_psro":
                return "rl"
            if t == "psro":
                return "evo"
        except Exception:
            pass
    if any(run_dir.glob("beta_iter*")) or any(run_dir.glob("nu_iter*")):
        return "rl"
    if any(run_dir.glob("search_iter*")):
        return "evo"
    # Default to RL — its layout is the primary pipeline.
    return "rl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _frames_from_snapshots(
    snap_dir: Path, file_re: re.Pattern, phase: str, start_order: int
) -> list[Frame]:
    """Build ordered Frames from a phase's snapshot ``.txt`` files."""
    files: list[tuple[int, Path]] = []
    if snap_dir.is_dir():
        for p in snap_dir.iterdir():
            m = file_re.match(p.name)
            if m:
                files.append((int(m.group(1)), p))
    files.sort(key=lambda t: t[0])

    frames: list[Frame] = []
    for i, (step, path) in enumerate(files):
        text = path.read_text(encoding="utf-8")
        header = parse_header(text)
        team = parse_team(text)
        win_rate = header.get("win_rate", team.win_rate)
        species = tuple(m.species for m in team.members)
        frames.append(
            Frame(
                order=start_order + i,
                phase=phase,
                step=header.get("step", header.get("round", step)),
                win_rate=win_rate,
                team=team,
                species=species,
            )
        )
    return frames


def _load_rl_phases(run_dir: Path) -> list[PhaseTimeline]:
    phase_dirs: list[tuple[int, int, str, Path]] = []  # (iter, kind_order, kind, dir)
    for d in run_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("beta_iter"):
            kind, ko = "beta", 0
        elif d.name.startswith("nu_iter"):
            kind, ko = "nu", 1
        else:
            continue
        m = _ITER_RE.search(d.name)
        if m:
            phase_dirs.append((int(m.group(1)), ko, kind, d))
    phase_dirs.sort(key=lambda t: (t[0], t[1]))

    phases: list[PhaseTimeline] = []
    order = 0
    for it, _ko, kind, d in phase_dirs:
        label = "β" if kind == "beta" else "ν"
        name = f"iter{it} · {label}"
        frames = _frames_from_snapshots(d / "snapshots", _STEP_FILE_RE, name, order)
        order += len(frames)
        metrics = _read_jsonl(d / "metrics.jsonl")
        phases.append(
            PhaseTimeline(
                name=name,
                kind=kind,
                iteration=it,
                frames=tuple(frames),
                metrics=tuple(metrics),
            )
        )
    return phases


def _load_evo_phases(run_dir: Path) -> list[PhaseTimeline]:
    search_dirs: list[tuple[int, Path]] = []
    for d in run_dir.iterdir():
        if d.is_dir() and d.name.startswith("search_iter"):
            m = _ITER_RE.search(d.name)
            if m:
                search_dirs.append((int(m.group(1)), d))
    search_dirs.sort(key=lambda t: t[0])

    phases: list[PhaseTimeline] = []
    order = 0
    for it, d in search_dirs:
        name = f"iter{it} search"
        # Evo snapshots are round_*.txt directly in the search dir.
        frames = _frames_from_snapshots(d, _ROUND_FILE_RE, name, order)
        order += len(frames)
        metrics = _read_jsonl(d / "rounds.jsonl")
        phases.append(
            PhaseTimeline(
                name=name,
                kind="search",
                iteration=it,
                frames=tuple(frames),
                metrics=tuple(metrics),
            )
        )
    return phases


def _load_iterations(run_dir: Path) -> list[IterationView]:
    iters: list[IterationView] = []
    for path in sorted(run_dir.glob("payoff_iter*.json")):
        m = re.search(r"payoff_iter(\d+)\.json", path.name)
        if not m:
            continue
        try:
            payoff = np.array(json.loads(path.read_text(encoding="utf-8")), dtype=float)
        except Exception:
            continue
        if payoff.ndim != 2 or payoff.size == 0:
            continue
        iters.append(
            IterationView(
                index=int(m.group(1)), payoff=payoff, nash_weights=_nash_weights(payoff)
            )
        )
    return iters


def _load_final_teams(
    run_dir: Path, run_type: str, iterations: list[IterationView]
) -> list[tuple[TeamView, float | None]]:
    """
    Rank the final population by Nash weight.

    RL checkpoints store ``population_texts`` aligned with ``nash_w`` (both match the
    payoff dimension) — the exact source. For evo, fall back to ``found_teams`` sorted
    by their own win rate (evo checkpoints don't persist population texts/weights).
    """
    ckpt_path = run_dir / "psro_checkpoint.json"
    if not ckpt_path.exists():
        return []
    try:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    texts = ckpt.get("population_texts")
    weights = ckpt.get("nash_w")
    if texts and weights and len(texts) == len(weights):
        pairs: list[tuple[TeamView, float | None]] = [
            (parse_team(t), float(w)) for t, w in zip(texts, weights)
        ]
        pairs = [(tv, w) for tv, w in pairs if tv.members]
        pairs.sort(key=lambda p: -(p[1] or 0.0))
        return pairs

    # Evo (or older checkpoints): render found_teams, rank by win rate.
    found = ckpt.get("found_teams") or []
    if found:
        from vgc_bench.team_builder.team import team_from_dict

        out: list[tuple[TeamView, float | None]] = []
        for d in found:
            try:
                team = team_from_dict(d)
                out.append((parse_team(team.to_showdown_text()), team.win_rate))
            except Exception:
                continue
        out.sort(key=lambda p: -(p[1] or 0.0))
        return out
    return []


def load_run(run_dir: str | Path) -> RunData:
    """Load a full run directory into a RunData model."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    run_type = detect_run_type(run_dir)
    phases = _load_rl_phases(run_dir) if run_type == "rl" else _load_evo_phases(run_dir)
    iterations = _load_iterations(run_dir)
    final_teams = _load_final_teams(run_dir, run_type, iterations)

    meta = _read_json_optional(run_dir / "run_meta.json")
    live = _read_json_optional(run_dir / "live_status.json")
    checkpoint = _read_json_optional(run_dir / "psro_checkpoint.json")
    last_activity = live["updated_at"] if live else _max_mtime(run_dir)
    status = compute_status(run_dir, meta, live, checkpoint, last_activity, time.time())

    return RunData(
        run_dir=run_dir,
        run_type=run_type,
        phases=tuple(phases),
        iterations=tuple(iterations),
        final_teams=tuple(final_teams),
        meta=meta,
        live=live,
        status=status,
        last_activity=last_activity,
    )


def list_runs(results_dir: str | Path = "results") -> list[Path]:
    """List candidate run directories under results/ (those with a checkpoint)."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    runs = [
        d
        for d in sorted(results_dir.iterdir())
        if d.is_dir() and (d / "psro_checkpoint.json").exists()
    ]
    return runs
