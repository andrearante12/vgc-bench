"""
Streamlit dashboard for the team builder: a Live view for watching an
in-progress run update in near-real-time, and Playback for replaying (or
sampling fresh teams from) any past run — both driven purely off the logs a
run already writes to disk.

Run it with::

    streamlit run vgc_bench/team_builder/viz/app.py -- results/<run>

An optional run-dir path may be passed after ``--`` to seed the sidebar's run
browser (useful for a run outside ``results/``, e.g. a custom ``--output``);
otherwise pick any run from the browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from vgc_bench.team_builder.viz.live import render_live
from vgc_bench.team_builder.viz.playback import render_playback
from vgc_bench.team_builder.viz.runs import RunSummary, scan_runs, summarize_run

_STATUS_CHIP = {
    "live": "🔴 LIVE",
    "done": "✅ done",
    "interrupted": "⏸ interrupted",
    "idle": "⚪ idle",
}


def _arg_run_dir() -> str | None:
    # Streamlit forwards args after `--` into sys.argv.
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            return a
    return None


def _format_summary(s: RunSummary) -> str:
    chip = _STATUS_CHIP.get(s.status, s.status)
    bits = [s.run_type]
    if s.reg:
        bits.append(f"reg={s.reg}")
    if s.completed_iterations is not None:
        it = f"iter {s.completed_iterations}"
        if s.target_iterations:
            it += f"/{s.target_iterations}"
        bits.append(it)
    if s.best_win_rate is not None:
        bits.append(f"best={s.best_win_rate:.0%}")
    return f"{chip}  {s.name}  ({', '.join(bits)})"


def _pick_run(default: str | None) -> tuple[str, RunSummary | None]:
    """Render the sidebar run browser; return the selected run dir + its summary."""
    summaries = scan_runs("results")
    options = [str(s.path) for s in summaries]
    by_path = {str(s.path): s for s in summaries}

    if default and default not in options:
        options.insert(0, default)
        # Not under results/ (e.g. a custom --output) — summarize it directly so its
        # status still drives sensible defaults (live/done) instead of falling back.
        by_path[default] = summarize_run(Path(default))

    if not options:
        st.error("No runs found under results/. Pass a run dir after `--`.")
        st.stop()

    idx = options.index(default) if default in options else 0
    run_dir = st.selectbox(
        "Run",
        options,
        index=idx,
        format_func=lambda p: _format_summary(by_path[p]) if p in by_path else p,
    )
    return run_dir, by_path.get(run_dir)


def _pick_mode(run_dir: str, summary: RunSummary | None) -> str:
    """
    Radio for Live vs Playback. Defaults to whichever mode fits the run's
    status, but only when the run selection actually changes — otherwise an
    explicit user choice would get silently reset on every rerun.
    """
    default_mode = "Live" if summary and summary.status == "live" else "Playback"
    if st.session_state.get("_mode_run_dir") != run_dir:
        st.session_state["_mode_run_dir"] = run_dir
        st.session_state["_mode_choice"] = default_mode
    return st.radio("Mode", ["Live", "Playback"], key="_mode_choice", horizontal=True)


def main() -> None:
    st.set_page_config(page_title="VGC Team Builder", layout="wide")
    st.title("🧬 Team Builder — evolution & evaluation")

    with st.sidebar:
        st.header("Run")
        run_dir, summary = _pick_run(_arg_run_dir())
        mode = _pick_mode(run_dir, summary)

    if mode == "Live":
        render_live(run_dir)
    else:
        render_playback(run_dir)


if __name__ == "__main__":
    main()
