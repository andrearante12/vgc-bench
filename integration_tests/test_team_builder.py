"""
Integration tests for vgc_bench.team_builder.

Requires a Pokemon Showdown server running on port 8100.
"""

import socket

import pytest

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.evaluator import TeamEvaluator
from vgc_bench.team_builder.optimizer import SearchConfig, best_response
from vgc_bench.team_builder.psro import PSROConfig, run_psro
from vgc_bench.team_builder.team import random_team


def _server_available() -> bool:
    try:
        with socket.create_connection(("localhost", 8100), timeout=1):
            return True
    except OSError:
        return False


requires_server = pytest.mark.skipif(
    not _server_available(),
    reason="Pokemon Showdown server not running on port 8100",
)


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("ma")


@pytest.fixture(scope="module")
def opponent_packed(space):
    """A random team packed into poke-env format to use as the fixed opponent."""
    return random_team(space, rng=None).to_packed_team()


@requires_server
class TestTeamEvaluator:
    def test_evaluate_returns_win_rate_in_range(self, space, opponent_packed):
        team = random_team(space)
        evaluator = TeamEvaluator(opponent_team=opponent_packed, n_battles=4, port=8100)
        scored = evaluator.evaluate(team)
        assert scored.win_rate is not None
        assert 0.0 <= scored.win_rate <= 1.0

    def test_two_evaluations_independent(self, space, opponent_packed):
        """Successive evaluate() calls must each return a valid win rate."""
        team1 = random_team(space)
        team2 = random_team(space)
        evaluator = TeamEvaluator(opponent_team=opponent_packed, n_battles=4, port=8100)
        s1 = evaluator.evaluate(team1)
        s2 = evaluator.evaluate(team2)
        assert s1.win_rate is not None and 0.0 <= s1.win_rate <= 1.0
        assert s2.win_rate is not None and 0.0 <= s2.win_rate <= 1.0

    def test_evaluate_vs_mixture(self, space):
        team = random_team(space)
        opp1 = random_team(space).to_packed_team()
        opp2 = random_team(space).to_packed_team()
        evaluator = TeamEvaluator(opponent_team=opp1, n_battles=4, port=8100)
        scored = evaluator.evaluate_vs_mixture(team, [opp1, opp2], [0.5, 0.5])
        assert scored.win_rate is not None
        assert 0.0 <= scored.win_rate <= 1.0


@requires_server
class TestBestResponse:
    def test_smoke(self, space, opponent_packed):
        """Smoke test: 1 round, tiny population, returns a valid scored team."""
        config = SearchConfig(
            rounds=1,
            population=4,
            candidates=4,
            n_battles=4,
            port=8100,
            seed=0,
        )
        best, history = best_response(opponent_packed, config, space)
        assert best.win_rate is not None
        assert 0.0 <= best.win_rate <= 1.0
        assert len(history) == 2  # round 0 (init) + round 1

    def test_cache_reduces_evaluations(self, space, opponent_packed):
        """Elites surviving to the next round should hit the cache."""
        config = SearchConfig(
            rounds=2,
            population=4,
            candidates=4,
            n_battles=4,
            port=8100,
            seed=1,
        )
        _, history = best_response(opponent_packed, config, space)
        total_hits = sum(r.cache_hits for r in history)
        assert total_hits > 0


@requires_server
class TestPSRO:
    def test_one_iteration_payoff_matrix(self, space, opponent_packed, tmp_path):
        """One PSRO iteration produces a 2×2 payoff matrix and a valid best team."""
        config = PSROConfig(
            n_iterations=1,
            search_config=SearchConfig(
                rounds=1,
                population=4,
                candidates=4,
                n_battles=4,
                port=8100,
                seed=0,
            ),
            port=8100,
            output_dir=tmp_path / "psro_test",
        )
        found = run_psro(config, meta_team=opponent_packed, space=space)
        assert len(found) == 1
        assert found[0].win_rate is not None

        # Payoff matrix file should exist and be 2×2
        import json
        payoff_files = list((tmp_path / "psro_test").glob("payoff_iter*.json"))
        assert len(payoff_files) == 1
        with payoff_files[0].open() as f:
            matrix = json.load(f)
        assert len(matrix) == 2
        assert len(matrix[0]) == 2
