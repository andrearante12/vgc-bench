"""
Live training view: watches an in-progress run and updates on its own, without a
full-page reload, via a Streamlit fragment that re-reads the run dir every few
seconds (cheap — cached by file mtime, so a quiet run costs nothing extra).

Shows the team currently being generated (from live_status.json, which the
trainer overwrites every step/round — far more granular than the snapshot
cadence) alongside the best team found so far, the win-rate curve, the latest
PSRO payoff/Nash, and species churn.
"""

from __future__ import annotations

import streamlit as st

from vgc_bench.team_builder.viz.components import (
    load_run_cached,
    payoff_heatmap,
    render_team,
    species_timeline,
    winrate_curve,
)
from vgc_bench.team_builder.viz.parse import parse_team

_REFRESH_INTERVAL = "3s"


def _status_badge(status: str) -> str:
    return {
        "live": "🔴 LIVE",
        "done": "✅ done",
        "interrupted": "⏸ interrupted",
        "idle": "⚪ idle",
    }.get(status, status)


def render_live(run_dir: str) -> None:
    """Top-level entry point for the Live mode page."""
    st.subheader("Live training")
    _live_fragment(run_dir)


@st.fragment(run_every=_REFRESH_INTERVAL)
def _live_fragment(run_dir: str) -> None:
    run = load_run_cached(run_dir)
    live = run.live or {}

    header = f"{_status_badge(run.status)}  ·  `{run_dir}`  ·  type: `{run.run_type}`"
    if run.live:
        iteration = live.get("iteration")
        phase = live.get("phase")
        step = live.get("step")
        total = live.get("total")
        target = run.meta.get("target_iterations") if run.meta else None
        iter_txt = (
            f"iteration {iteration + 1}/{target}"
            if target and iteration is not None
            else f"iteration {iteration}"
        )
        header += f"  ·  {iter_txt}  ·  phase `{phase}`  ·  step {step}/{total}"
    st.caption(header)

    if run.status != "live":
        st.info(
            "This run doesn't look live right now (no recent activity). "
            "Switch to **Playback** to replay it, or leave this open — it'll "
            "pick back up automatically if training resumes."
        )

    if run.live is None:
        st.warning(
            "No live_status.json yet — the run may not have started its first "
            "step, or predates live logging. Showing the latest available snapshot."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Current win rate", _pct(live.get("current_win_rate")))
    c2.metric("Recent win rate", _pct(live.get("recent_win_rate")))
    c3.metric("Best win rate", _pct(live.get("best_win_rate")))

    if live.get("current_team_showdown"):
        st.subheader("Currently exploring")
        render_team(parse_team(live["current_team_showdown"]), None)

    if live.get("best_team_showdown"):
        st.subheader("Best so far")
        render_team(parse_team(live["best_team_showdown"]), None)

    frames = run.frames
    if frames:
        st.subheader("Win rate over the run")
        st.pyplot(winrate_curve(frames, len(frames) - 1))

    if run.iterations:
        st.subheader("Latest matchup payoff & Nash weights")
        it = run.iterations[-1]
        hcol, ncol = st.columns([3, 2])
        with hcol:
            st.pyplot(payoff_heatmap(it.payoff, it.nash_weights))
        with ncol:
            st.caption("Nash equilibrium weight per population team")
            st.bar_chart(
                {"nash weight": it.nash_weights},
                height=max(220, 26 * len(it.nash_weights)),
            )

    if frames:
        st.subheader("Species evolution")
        fig = species_timeline(frames)
        if fig is not None:
            st.pyplot(fig)


def _pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "—"
