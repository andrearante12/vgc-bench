"""
Shared Streamlit/matplotlib rendering pieces for the team-builder dashboard —
team cards, the win-rate curve, the payoff heatmap, and the species-churn grid.

Used by both the live view (viz/live.py) and playback (viz/playback.py) so the
two modes render identically; only the surrounding page (what frame/iteration is
selected, whether it auto-advances) differs between them.
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from vgc_bench.team_builder.viz.loader import RunData, load_run  # noqa: E402
from vgc_bench.team_builder.viz.parse import PokemonView, TeamView  # noqa: E402
from vgc_bench.team_builder.viz.sprites import (  # noqa: E402
    TYPE_COLORS,
    species_types,
    sprite_url,
    type_color,
)

# --- ink tokens (text never wears the series/type color) -------------------------
INK = "#e8e8ea"
MUTED = "#9aa0a6"
SURFACE = "#1a1c1f"
CARD = "#232629"
NEW = "#63bc5a"  # "changed" accent (status-green), used with a text label too


# ------------------------------------------------------------- cached loading ---
def run_signature(run_dir: Path) -> float:
    """Cheap cache key: newest mtime under the run dir, so a run reloads on change."""
    latest = 0.0
    for p in run_dir.rglob("*"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            continue
    return latest


@st.cache_data(show_spinner=False)
def _load_run_cached(run_dir: str, _sig: float) -> RunData:
    return load_run(run_dir)


def load_run_cached(run_dir: str | Path) -> RunData:
    """Load a run, cached until its newest file mtime changes."""
    run_dir = Path(run_dir)
    return _load_run_cached(str(run_dir), run_signature(run_dir))


# ------------------------------------------------------------------ team cards ---
def _pokemon_card(mon: PokemonView, new_species: bool, changed_moves: set[str]) -> str:
    accent = type_color(mon.species)
    types = species_types(mon.species)
    type_badges = "".join(
        f'<span style="background:{TYPE_COLORS.get(t, "#777")};color:#111;'
        f'border-radius:4px;padding:1px 6px;font-size:10px;margin-right:3px">{html.escape(t)}</span>'  # noqa: E501
        for t in types
    )
    safe_species = html.escape(mon.species)
    url = sprite_url(mon.species)
    # onerror hides the broken img and reveals the name-only fallback span.
    img = (
        f'<img src="{url}" width="72" height="72" style="image-rendering:pixelated"'
        f" onerror=\"this.style.display='none';this.nextElementSibling.style.display='block'\">"  # noqa: E501
        f'<span style="display:none;font-size:12px;color:{MUTED}">no sprite</span>'
    )
    new_badge = (
        f'<span style="background:{NEW};color:#111;border-radius:4px;padding:1px 5px;'
        f'font-size:10px;margin-left:4px">NEW</span>'
        if new_species
        else ""
    )
    move_rows = "".join(
        f'<li style="color:{NEW if m in changed_moves else MUTED};'
        f'font-weight:{"600" if m in changed_moves else "400"}">{html.escape(m)}'
        f"{' ●' if m in changed_moves else ''}</li>"
        for m in mon.moves
    )
    item = html.escape(mon.item or "—")
    ability = html.escape(mon.ability or "—")
    nature = html.escape(mon.nature or "—")
    tera = (
        f'<span style="border:1px solid {TYPE_COLORS.get(mon.tera, "#777")};'
        f"color:{TYPE_COLORS.get(mon.tera, MUTED)};border-radius:4px;padding:0 5px;"
        f'font-size:10px">Tera {html.escape(mon.tera)}</span>'
        if mon.tera
        else ""
    )
    evs = html.escape(mon.ev_summary() or "no EVs")
    name_html = f'<div style="color:{INK};font-weight:700;font-size:14px">{safe_species}{new_badge}</div>'  # noqa: E501
    moves_html = f'<ul style="margin:4px 0 0 14px;padding:0;font-size:11px;line-height:1.5">{move_rows}</ul>'  # noqa: E501
    return f"""
    <div style="background:{CARD};border-left:4px solid {accent};border-radius:8px;
                padding:8px 10px;margin-bottom:8px;min-height:210px">
      <div style="display:flex;align-items:center;gap:8px">
        <div>{img}</div>
        <div>
          {name_html}
          <div style="margin:2px 0">{type_badges}</div>
          <div style="color:{MUTED};font-size:11px">@ {item}</div>
        </div>
      </div>
      <div style="color:{MUTED};font-size:11px;margin-top:4px">
        {ability} · {nature} {tera}
      </div>
      <div style="color:{MUTED};font-size:10px;margin:2px 0">EVs: {evs}</div>
      {moves_html}
    </div>
    """


def render_team(team: TeamView, prev: TeamView | None) -> None:
    """Render a team as a row of Pokémon cards, highlighting churn vs prev."""
    prev_species = {m.species for m in prev.members} if prev else set()
    prev_moves_by_sp = {m.species: set(m.moves) for m in prev.members} if prev else {}
    cols = st.columns(len(team.members) or 1)
    for col, mon in zip(cols, team.members):
        new_species = bool(prev) and mon.species not in prev_species
        changed = set()
        if not new_species and mon.species in prev_moves_by_sp:
            changed = set(mon.moves) - prev_moves_by_sp[mon.species]
        col.markdown(_pokemon_card(mon, new_species, changed), unsafe_allow_html=True)


# --------------------------------------------------------------------- charts ---
def payoff_heatmap(payoff: np.ndarray, nash: list[float]) -> Figure:
    n = payoff.shape[0]
    fig, ax = plt.subplots(figsize=(0.55 * n + 1.6, 0.55 * n + 1.2))
    # Win rate is a polarity around 0.5 (even matchup) -> diverging map, gray midpoint.
    im = ax.imshow(payoff, cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=8)
    ax.set_yticklabels([str(i) for i in range(n)], fontsize=8)
    ax.set_xlabel("opponent team", fontsize=9)
    ax.set_ylabel("row team", fontsize=9)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                v = payoff[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#111" if 0.3 < v < 0.7 else "#f5f5f5",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(row beats col)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def winrate_curve(frames, sel: int) -> Figure:
    xs = list(range(len(frames)))
    ys = [f.win_rate if f.win_rate is not None else np.nan for f in frames]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.plot(xs, ys, color="#4d90d5", lw=2.0, marker="o", ms=4, zorder=2)
    # Selected-frame marker: a recessive vertical rule + an emphasized point.
    ax.axvline(sel, color=MUTED, lw=1.0, ls="--", zorder=1)
    if ys[sel] is not None and not np.isnan(ys[sel]):
        ax.plot([sel], [ys[sel]], marker="o", ms=9, color="#ff9d55", zorder=3)
    # Phase boundaries as light guides.
    boundary = 0
    seen = set()
    for f in frames:
        if f.phase not in seen:
            seen.add(f.phase)
            if boundary:
                ax.axvline(boundary - 0.5, color=CARD, lw=1.0, zorder=0)
        boundary += 1
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.5, len(frames) - 0.5)
    ax.set_ylabel("win rate", fontsize=9)
    ax.set_xlabel("timeline frame →", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#333", lw=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


def species_timeline(frames) -> Figure | None:
    if not frames:
        return None
    # Species that ever appear, ordered by first appearance.
    order: list[str] = []
    for f in frames:
        for s in f.species:
            if s not in order:
                order.append(s)
    if not order:
        return None
    row_of = {s: i for i, s in enumerate(order)}
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * len(frames)), 0.32 * len(order) + 1))
    for x, f in enumerate(frames):
        for s in f.species:
            ax.add_patch(
                Rectangle(
                    (x - 0.5, row_of[s] - 0.5),
                    1,
                    1,
                    color=type_color(s),
                    ec=SURFACE,
                    lw=1.0,
                )
            )
    ax.set_xlim(-0.5, len(frames) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("timeline frame →", fontsize=9)
    ax.set_xticks(range(0, len(frames), max(1, len(frames) // 12)))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("Best-team species over the run (color = primary type)", fontsize=10)
    fig.tight_layout()
    return fig
