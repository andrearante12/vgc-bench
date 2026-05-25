"""
Behavioral cloning initialization for TeamBuilderNetwork output heads.

Reads tournament-frequency weights from BuildSpace and sets additive logit
biases on the Species, Item, and Move output heads. Called once before any RL
training to give the network a warm start from tournament data.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vgc_bench.team_builder.build_space import BuildSpace
    from vgc_bench.team_builder.rl.team_builder_network import TeamBuilderNetwork


def apply_bc_biases(network: TeamBuilderNetwork, space: BuildSpace) -> None:
    """
    Write log-frequency biases into the Species, Item, and Move output heads.

    Species head gets per-species tournament-frequency bias. Item and Move
    heads get a global average across all species (the heads are shared so
    species-specific biases cannot be expressed as a single bias vector).

    Args:
        network: TeamBuilderNetwork to initialize.
        space:   BuildSpace with BC-initialized weight tables.
    """
    # --- Species head ---
    species_bias = torch.zeros(network.n_species)
    for i, sp in enumerate(network.species_list):
        w = space.species_weights.get(sp, 1.0)
        species_bias[i] = math.log(max(w, 1e-8))
    with torch.no_grad():
        network.species_head.bias.copy_(species_bias)

    # --- Move heads (average log-freq across species) ---
    move_bias = torch.zeros(network.n_moves)
    move_counts = torch.zeros(network.n_moves)
    for sp in network.species_list:
        mw = space._data[sp].move_weights
        for move_name, w in mw.items():
            idx = network.move_to_idx.get(network._to_id(move_name))
            if idx is not None:
                move_bias[idx] += math.log(max(w, 1e-8))
                move_counts[idx] += 1.0
    mask = move_counts > 0
    move_bias[mask] /= move_counts[mask]
    with torch.no_grad():
        for head in network.move_heads:
            head.bias.copy_(move_bias)

    # --- Item head (average log-freq across species) ---
    item_bias = torch.zeros(network.n_items)
    item_counts = torch.zeros(network.n_items)
    for sp in network.species_list:
        iw = space._data[sp].item_weights
        for item_name, w in iw.items():
            idx = network.item_to_idx.get(network._to_id(item_name))
            if idx is not None:
                item_bias[idx] += math.log(max(w, 1e-8))
                item_counts[idx] += 1.0
    mask = item_counts > 0
    item_bias[mask] /= item_counts[mask]
    with torch.no_grad():
        network.item_head.bias.copy_(item_bias)
