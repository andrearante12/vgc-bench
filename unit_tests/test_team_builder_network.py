"""Unit tests for the RL team builder network — no Showdown server required."""

from __future__ import annotations

import math
import random

import pytest
import torch

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.rl.bc_init import apply_bc_biases
from vgc_bench.team_builder.rl.team_builder_network import (
    TeamBuilderNetwork,
    _TOKEN_DIM,
    _parse_showdown_team,
)
from vgc_bench.team_builder.team import TEAM_SIZE


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("ma")


@pytest.fixture(scope="module")
def net(space):
    n = TeamBuilderNetwork(space)
    apply_bc_biases(n, space)
    return n


@pytest.fixture(scope="module")
def two_teams():
    from pathlib import Path
    teams_dir = Path("teams/reg_ma")
    t1 = (teams_dir / "PC1.txt").read_text(encoding="utf-8")
    t2 = (teams_dir / "PC2.txt").read_text(encoding="utf-8")
    return [t1, t2]


# ------------------------------------------------------------------ #
# Parser
# ------------------------------------------------------------------ #

class TestParseShowdownTeam:
    def test_parses_six_pokemon(self, two_teams):
        pokemon = _parse_showdown_team(two_teams[0])
        assert len(pokemon) == TEAM_SIZE

    def test_fields_present(self, two_teams):
        for p in _parse_showdown_team(two_teams[0]):
            assert p["species"]
            assert len(p["moves"]) == 4
            assert p["nature"]

    def test_evs_are_valid_ints(self, two_teams):
        from vgc_bench.team_builder.pokemon_build import EV_MAX_PER_STAT, EV_MAX_TOTAL
        for p in _parse_showdown_team(two_teams[0]):
            assert sum(p["evs"]) <= EV_MAX_TOTAL
            for ev in p["evs"]:
                assert 0 <= ev <= EV_MAX_PER_STAT


# ------------------------------------------------------------------ #
# Token encoding
# ------------------------------------------------------------------ #

class TestTokenEncoding:
    def test_token_dim(self, net, two_teams):
        pokemon = _parse_showdown_team(two_teams[0])
        tokens = net._encode_team_tokens(pokemon)
        assert tokens.shape == (TEAM_SIZE, _TOKEN_DIM)

    def test_encode_team_output_shape(self, net, two_teams):
        pokemon = _parse_showdown_team(two_teams[0])
        tokens = net._encode_team_tokens(pokemon)
        ctx = net.encode_team(tokens)
        assert ctx.shape == (net.D_MODEL,)


# ------------------------------------------------------------------ #
# Population encoding
# ------------------------------------------------------------------ #

class TestPopulationEncoding:
    def test_single_team_encoding(self, net, two_teams):
        ctx = net.encode_population([two_teams[0]], [1.0])
        assert ctx.shape == (net.D_MODEL,)

    def test_two_teams_encoding(self, net, two_teams):
        ctx = net.encode_population(two_teams, [0.6, 0.4])
        assert ctx.shape == (net.D_MODEL,)

    def test_weights_normalized(self, net, two_teams):
        ctx_norm = net.encode_population(two_teams, [0.6, 0.4])
        ctx_unnorm = net.encode_population(two_teams, [6.0, 4.0])
        assert torch.allclose(ctx_norm, ctx_unnorm, atol=1e-5)


# ------------------------------------------------------------------ #
# Team generation
# ------------------------------------------------------------------ #

class TestGenerate:
    def test_produces_valid_team(self, net, two_teams):
        team, log_prob, value = net.generate(two_teams, [0.5, 0.5])
        assert len(team.members) == TEAM_SIZE
        assert team.win_rate is None

    def test_species_clause(self, net, two_teams):
        for _ in range(10):
            team, _, _ = net.generate(two_teams, [0.5, 0.5])
            species = [m.species for m in team.members]
            assert len(set(species)) == TEAM_SIZE, f"Duplicate species: {species}"

    def test_item_clause(self, net, two_teams):
        for _ in range(10):
            team, _, _ = net.generate(two_teams, [0.5, 0.5])
            items = [m.item for m in team.members]
            assert len(set(items)) == TEAM_SIZE, f"Duplicate items: {items}"

    def test_log_prob_is_scalar(self, net, two_teams):
        _, log_prob, value = net.generate(two_teams, [0.5, 0.5])
        assert log_prob.shape == ()
        assert value.shape == ()

    def test_log_prob_is_negative(self, net, two_teams):
        # All log-probs of a probability distribution are ≤ 0
        _, log_prob, _ = net.generate(two_teams, [0.5, 0.5])
        assert log_prob.item() < 0

    def test_gradient_flows_through_log_prob(self, net, two_teams):
        net_copy = TeamBuilderNetwork(net.space)
        apply_bc_biases(net_copy, net.space)
        team, log_prob, value = net_copy.generate(two_teams, [0.5, 0.5])
        reward = 1.0
        loss = -(log_prob * (reward - value.detach())) + 0.5 * (value - reward) ** 2
        loss.backward()
        grads = [p.grad for p in net_copy.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_generate_greedy_valid(self, net, two_teams):
        team = net.generate_greedy(two_teams, [0.5, 0.5])
        assert len(team.members) == TEAM_SIZE
        assert len(set(m.species for m in team.members)) == TEAM_SIZE
        assert len(set(m.item for m in team.members)) == TEAM_SIZE


# ------------------------------------------------------------------ #
# Training loop smoke test (no Showdown server)
# ------------------------------------------------------------------ #

class TestTrainPolicySmokeTest:
    """Exercises the full REINFORCE loop with mocked battle evaluation."""

    def test_parameters_update_after_training(self, space, two_teams):
        from vgc_bench.team_builder.rl.train import train_policy

        rng = random.Random(42)

        def mock_eval(team, opp_text):
            return rng.random()  # random win rate, no battles

        net_before = TeamBuilderNetwork(space)
        apply_bc_biases(net_before, space)
        params_before = {
            k: v.clone() for k, v in net_before.named_parameters() if v.requires_grad
        }

        trained_net, best_team = train_policy(
            population_texts=two_teams,
            nash_weights=[0.5, 0.5],
            space=space,
            n_steps=10,
            eval_fn=mock_eval,
        )

        changed = sum(
            1 for k, v in trained_net.named_parameters()
            if v.requires_grad and not torch.allclose(v, params_before.get(k, v))
        )
        assert changed > 0, "No parameters changed after 10 training steps"

    def test_best_team_is_valid(self, space, two_teams):
        from vgc_bench.team_builder.rl.train import train_policy

        _, best_team = train_policy(
            population_texts=two_teams,
            nash_weights=[0.5, 0.5],
            space=space,
            n_steps=5,
            eval_fn=lambda t, _: random.random(),
        )

        assert len(best_team.members) == TEAM_SIZE
        assert len(set(m.species for m in best_team.members)) == TEAM_SIZE
        assert len(set(m.item for m in best_team.members)) == TEAM_SIZE
        assert best_team.win_rate is not None

    def test_win_rate_tracked(self, space, two_teams):
        from vgc_bench.team_builder.rl.train import train_policy

        # eval_fn always wins
        _, best_team = train_policy(
            population_texts=two_teams,
            nash_weights=[0.5, 0.5],
            space=space,
            n_steps=5,
            eval_fn=lambda t, _: 1.0,
        )
        assert best_team.win_rate == 1.0


# ------------------------------------------------------------------ #
# BC initialization
# ------------------------------------------------------------------ #

class TestBCInit:
    def test_species_bias_nonuniform(self, net):
        bias = net.species_head.bias.detach()
        assert bias.min().item() != bias.max().item()

    def test_common_species_top_bias(self, net, space):
        bias = net.species_head.bias.detach()
        top_idx = bias.argmax().item()
        top_sp = net.species_list[top_idx]
        # Top species should be among known high-usage tournament Pokémon
        high_usage = {s for s in ("Incineroar", "Rillaboom", "Sneasler", "Garchomp")
                      if s in space.species_weights}
        if high_usage:
            top_by_weight = max(high_usage, key=lambda s: space.species_weights[s])
            assert net.species_list[top_idx] in high_usage, (
                f"Top bias species is {top_sp}, expected one of {high_usage}"
            )

    def test_move_bias_set(self, net):
        bias = net.move_heads[0].bias.detach()
        # Most move biases should be non-zero after BC init
        assert (bias != 0).sum().item() > 100
