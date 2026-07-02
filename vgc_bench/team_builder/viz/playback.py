"""
Playback: replay any run's team-evolution timeline like a video — play/pause,
speed, step buttons, and a manual scrubber — plus a panel to sample a fresh team
from a saved policy.pt checkpoint (a pure CPU forward pass, no Showdown server).

Autoplay is driven by a single Streamlit fragment ticking on a fixed interval;
the frame position lives in ``st.session_state`` under one key that both the
autoplay tick and the manual slider write to, so dragging the slider naturally
overrides autoplay (Streamlit's normal widget-state semantics — no extra
plumbing needed).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from vgc_bench.team_builder.viz.components import (
    load_run_cached,
    payoff_heatmap,
    render_team,
    species_timeline,
    winrate_curve,
)
from vgc_bench.team_builder.viz.loader import RunData
from vgc_bench.team_builder.viz.model import generate_team, load_network_cached

_TICK_INTERVAL = "0.6s"
_SPEEDS = (1, 2, 4)


def advance(playhead: int, n_frames: int, speed: int, loop: bool = True) -> int:
    """
    Compute the next playhead position, stepping forward by `speed` frames.

    Wraps to 0 past the last frame when loop=True; otherwise clamps at the end.
    A pure function (no Streamlit) so it's unit-testable on its own.
    """
    if n_frames <= 0:
        return 0
    nxt = playhead + speed
    if nxt >= n_frames:
        return nxt % n_frames if loop else n_frames - 1
    return playhead if speed <= 0 else nxt


def _find_policy_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/policy.pt"))


def _load_conditioning(run_dir: Path) -> tuple[list[str], list[float]] | None:
    """
    The opponent population + Nash weights a saved network was trained to best-
    respond to, straight from the run's own checkpoint — the exact conditioning
    it saw during training.
    """
    import json

    ckpt_path = run_dir / "psro_checkpoint.json"
    if not ckpt_path.exists():
        return None
    try:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    texts = ckpt.get("population_texts")
    weights = ckpt.get("nash_w")
    if texts and weights and len(texts) == len(weights):
        return texts, weights
    return None


def _reset_if_run_changed(run_dir: str, n_frames: int) -> None:
    if st.session_state.get("_playback_run_dir") != run_dir:
        st.session_state["_playback_run_dir"] = run_dir
        st.session_state["playback_head"] = max(n_frames - 1, 0)
        st.session_state["playback_playing"] = False
        st.session_state["playback_speed"] = 1


def render_playback(run_dir: str) -> None:
    """Top-level entry point for the Playback mode page."""
    st.subheader("Playback")
    run = load_run_cached(run_dir)
    frames = run.frames
    st.caption(
        f"`{run_dir}` · type: `{run.run_type}` · {len(run.phases)} phases · "
        f"{len(frames)} snapshots · {len(run.iterations)} PSRO iterations"
    )
    if not frames:
        st.warning("No team snapshots found in this run yet.")
        return

    _reset_if_run_changed(run_dir, len(frames))
    _transport_and_frames(run_dir, run, len(frames))
    _generate_panel(run_dir, run)


@st.fragment(run_every=_TICK_INTERVAL)
def _transport_and_frames(run_dir: str, run: RunData, n_frames: int) -> None:
    frames = run.frames
    labels = [f"{f.phase} · step {f.step}" for f in frames]

    if st.session_state.get("playback_playing", False):
        st.session_state["playback_head"] = advance(
            st.session_state["playback_head"],
            n_frames,
            st.session_state["playback_speed"],
        )

    tcol1, tcol2, tcol3, tcol4 = st.columns([1, 1, 1, 3])
    playing = st.session_state["playback_playing"]
    if tcol1.button("⏸ Pause" if playing else "▶ Play", use_container_width=True):
        st.session_state["playback_playing"] = not playing
    if tcol2.button("⏮ Step back", use_container_width=True):
        st.session_state["playback_head"] = max(
            0, st.session_state["playback_head"] - 1
        )
        st.session_state["playback_playing"] = False
    if tcol3.button("⏭ Step fwd", use_container_width=True):
        st.session_state["playback_head"] = min(
            n_frames - 1, st.session_state["playback_head"] + 1
        )
        st.session_state["playback_playing"] = False
    tcol4.radio(
        "Speed",
        _SPEEDS,
        key="playback_speed",
        horizontal=True,
        format_func=lambda s: f"{s}×",
    )

    st.select_slider(
        "Timeline",
        options=list(range(n_frames)),
        key="playback_head",
        format_func=lambda i: labels[i],
    )
    sel = st.session_state["playback_head"]
    frame = frames[sel]
    prev = frames[sel - 1] if sel > 0 else None

    wr = frame.win_rate
    c1, c2, c3 = st.columns(3)
    c1.metric("Phase", frame.phase)
    c2.metric("Step / round", frame.step)
    c3.metric("Win rate", f"{wr:.0%}" if wr is not None else "—")

    st.subheader("Best team at this point")
    st.caption(
        "Green **NEW** = species entered the team since the previous frame; "
        "green ● = move changed on a kept species."
    )
    render_team(frame.team, prev.team if prev else None)

    st.subheader("Win rate over the run")
    st.pyplot(winrate_curve(frames, sel))

    if run.iterations:
        st.subheader("Matchup payoff & Nash weights")
        it_idx = st.select_slider(
            "PSRO iteration",
            options=[it.index for it in run.iterations],
            value=run.iterations[-1].index,
        )
        it = next(it for it in run.iterations if it.index == it_idx)
        hcol, ncol = st.columns([3, 2])
        with hcol:
            st.pyplot(payoff_heatmap(it.payoff, it.nash_weights))
        with ncol:
            st.caption("Nash equilibrium weight per population team")
            st.bar_chart(
                {"nash weight": it.nash_weights},
                height=max(220, 26 * len(it.nash_weights)),
            )

    st.subheader("Species evolution")
    fig = species_timeline(frames)
    if fig is not None:
        st.pyplot(fig)

    if run.final_teams:
        st.subheader("Final population — ranked by Nash weight")
        for rank, (team, weight) in enumerate(run.final_teams, 1):
            species = ", ".join(m.species for m in team.members)
            wtxt = f"nash={weight:.3f}" if weight is not None else ""
            wr_txt = (
                f" · win_rate={team.win_rate:.0%}" if team.win_rate is not None else ""
            )
            with st.expander(f"#{rank}  {wtxt}{wr_txt}  —  {species}"):
                render_team(team, None)


def _generate_panel(run_dir: str, run: RunData) -> None:
    st.subheader("Generate a fresh team from a saved checkpoint")
    run_path = Path(run_dir)
    checkpoints = _find_policy_checkpoints(run_path)
    reg = run.meta.get("reg") if run.meta else None

    if not checkpoints:
        st.caption("No saved policy checkpoints (`policy.pt`) found in this run.")
        return
    if not reg:
        st.caption(
            "This run has no `run_meta.json`, so its regulation is unknown — the "
            "network can't be rebuilt without it."
        )
        return

    labels = [str(p.relative_to(run_path)) for p in checkpoints]
    idx = st.selectbox(
        "Checkpoint", range(len(checkpoints)), format_func=lambda i: labels[i]
    )
    greedy = (
        st.radio("Sampling", ["Greedy (deterministic)", "Stochastic"], horizontal=True)
        == "Greedy (deterministic)"
    )

    if st.button("🎲 Generate team"):
        conditioning = _load_conditioning(run_path)
        if conditioning is None:
            st.error(
                "No opponent conditioning population found in psro_checkpoint.json "
                "— can't sample a team without knowing what it should respond to."
            )
        else:
            texts, weights = conditioning
            with st.spinner("Loading network and sampling…"):
                net = load_network_cached(checkpoints[idx], reg, device="cpu")
                st.session_state["_generated_team"] = generate_team(
                    net, texts, weights, greedy=greedy
                )

    if "_generated_team" in st.session_state:
        render_team(st.session_state["_generated_team"], None)
