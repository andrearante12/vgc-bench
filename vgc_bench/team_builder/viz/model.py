"""
Load a saved policy.pt checkpoint and sample a fresh team from it.

This is a pure CPU torch forward pass through the trained TeamBuilderNetwork —
no Showdown server, no GPU, no battle stack. Generating a team is intentionally
decoupled from evaluating one (win rate still requires real battles).

Torch and the network/build-space modules are imported lazily inside the
functions below, so importing viz.model (and everything that transitively
imports it, like the dashboard) stays fast and doesn't pull in torch unless a
checkpoint is actually loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from vgc_bench.team_builder.viz.parse import TeamView, parse_team


def load_network(policy_pt: str | Path, reg: str, device: str = "cpu") -> Any:
    """
    Reconstruct a TeamBuilderNetwork from a policy.pt checkpoint.

    policy.pt only stores tensor weights (state_dict) — it is not self-describing,
    so the caller must supply the regulation the run used (recorded in the run's
    run_meta.json) to rebuild the matching BuildSpace and vocab shapes. Loading
    with a mismatched regulation fails with a state_dict shape mismatch.
    """
    import torch

    from vgc_bench.team_builder.build_space import BuildSpace
    from vgc_bench.team_builder.rl.team_builder_network import TeamBuilderNetwork

    space = BuildSpace.from_regulation(reg)
    network = TeamBuilderNetwork(space).to(device)
    ckpt = torch.load(Path(policy_pt), map_location=device, weights_only=False)
    network.load_state_dict(ckpt["network_state_dict"])
    network.eval()
    return network


@st.cache_resource(show_spinner="Loading model checkpoint…")
def _load_network_cached(policy_pt: str, _mtime: float, reg: str, device: str) -> Any:
    return load_network(policy_pt, reg, device)


def load_network_cached(policy_pt: str | Path, reg: str, device: str = "cpu") -> Any:
    """Same as load_network, cached per (path, mtime, reg, device) for the session."""
    policy_pt = Path(policy_pt)
    return _load_network_cached(str(policy_pt), policy_pt.stat().st_mtime, reg, device)


def generate_team(
    network: Any,
    population_texts: list[str],
    nash_weights: list[float],
    greedy: bool = True,
) -> TeamView:
    """
    Sample one team from a loaded network, conditioned on an opponent mixture.

    greedy=True (argmax at every step) is reproducible; greedy=False samples
    stochastically, matching how the team was generated during training.
    """
    if greedy:
        team = network.generate_greedy(population_texts, nash_weights)
    else:
        team, _log_prob, _value = network.generate(population_texts, nash_weights)
    return parse_team(team.to_showdown_text())
