"""
Unit tests for vgc_bench.team_builder.viz.model — loading a policy.pt checkpoint
and sampling a team from it. CPU-only, no Showdown server or GPU: generation is a
pure torch forward pass, driven through run_sp_psro's eval_fn hook to produce a
real checkpoint without a live battle backend.
"""

from __future__ import annotations

import random

import pytest

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.rl.train import run_sp_psro
from vgc_bench.team_builder.team import random_team
from vgc_bench.team_builder.viz.model import (
    generate_team,
    load_network,
    load_network_cached,
)


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("ma")


@pytest.fixture(scope="module")
def meta_team_text(space):
    return random_team(space, rng=random.Random(0)).to_showdown_text()


@pytest.fixture(scope="module")
def policy_pt(space, meta_team_text, tmp_path_factory):
    """A real policy.pt checkpoint, produced serverlessly via eval_fn."""
    out = tmp_path_factory.mktemp("model_run")
    meta_path = out / "meta.txt"
    meta_path.write_text(meta_team_text, encoding="utf-8")
    run_sp_psro(
        meta_team_paths=[str(meta_path)],
        space=space,
        reg="ma",
        n_iterations=1,
        n_steps=4,
        log_every=2,
        snapshot_every=2,
        quality_threshold=0.0,
        output_dir=out / "run",
        eval_fn=lambda team, opp_text: random.random(),
        resume=False,
    )
    return out / "run" / "beta_iter000" / "policy.pt"


class TestLoadNetwork:
    def test_loads_from_checkpoint(self, policy_pt):
        net = load_network(policy_pt, reg="ma", device="cpu")
        assert type(net).__name__ == "TeamBuilderNetwork"

    def test_cached_returns_same_instance(self, policy_pt):
        a = load_network_cached(policy_pt, reg="ma", device="cpu")
        b = load_network_cached(policy_pt, reg="ma", device="cpu")
        assert a is b


class TestGenerateTeam:
    def test_greedy_generation_is_deterministic_and_valid(
        self, policy_pt, meta_team_text
    ):
        net = load_network(policy_pt, reg="ma", device="cpu")
        team_a = generate_team(net, [meta_team_text], [1.0], greedy=True)
        team_b = generate_team(net, [meta_team_text], [1.0], greedy=True)
        assert len(team_a.members) == 6
        assert [m.species for m in team_a.members] == [
            m.species for m in team_b.members
        ]

    def test_stochastic_generation_returns_valid_team(self, policy_pt, meta_team_text):
        net = load_network(policy_pt, reg="ma", device="cpu")
        team = generate_team(net, [meta_team_text], [1.0], greedy=False)
        assert len(team.members) == 6
        assert all(len(m.moves) == 4 for m in team.members)
