"""
PSRO (Policy Space Response Oracles) loop for adversarial team building.

Iteratively builds a population of teams by finding best responses to the
Nash equilibrium mixture of the current population, converging toward teams
that cannot be easily exploited.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from nashpy import Game

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.evaluator import TeamEvaluator
from vgc_bench.team_builder.optimizer import RoundRecord, SearchConfig, best_response_vs_mixture
from vgc_bench.team_builder.team import CandidateTeam

logger = logging.getLogger(__name__)


@dataclass
class PSROConfig:
    """
    Configuration for the PSRO adversarial team-building loop.

    Attributes:
        n_iterations: Number of PSRO iterations (best-response oracle calls).
        search_config: Configuration for each best-response search.
        port: Showdown server port.
        reg: Regulation pool to sample from.
        output_dir: Directory for logging payoff matrices and found teams.
    """

    n_iterations: int = 10
    search_config: SearchConfig = field(default_factory=SearchConfig)
    port: int = 8100
    reg: str = "ma"
    output_dir: Path = Path("results/team_builder")


def run_psro(
    config: PSROConfig,
    meta_team: str,
    space: BuildSpace | None = None,
) -> list[CandidateTeam]:
    """
    Run the PSRO adversarial team-building loop.

    Starts with one fixed meta opponent team (iteration 0) and iteratively
    adds best-response teams to the population. Converges toward teams that
    perform well against the Nash equilibrium mixture of all found teams.

    Args:
        config: PSRO configuration.
        meta_team: Packed team string for the initial opponent.
        space: Pre-built BuildSpace. Loads from disk when None.

    Returns:
        All found CandidateTeams sorted by final Nash weight (descending).
        The meta_team is not included (it has no CandidateTeam wrapper).

    Algorithm:
        1. teams = [meta_team], payoff = [[0.5]]
        2. For each iteration:
            a. Compute Nash distribution over teams.
            b. Find best_response_vs_mixture(teams, nash_weights).
            c. Evaluate new_team vs all existing teams → new payoff row/col.
            d. Append new_team to population; extend payoff; recompute Nash.
            e. Save payoff matrix and team texts to output_dir.
        3. Return found teams sorted by Nash weight descending.
    """
    if space is None:
        space = BuildSpace.from_regulation(config.reg)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # opponent_teams: packed strings (first entry is the fixed meta team)
    opponent_teams: list[str] = [meta_team]
    # found_teams: CandidateTeam objects for teams we discovered
    found_teams: list[CandidateTeam] = []
    # payoff[i][j] = win rate of opponent_teams[i] vs opponent_teams[j]
    payoff = np.array([[0.5]])

    for iteration in range(config.n_iterations):
        logger.info("PSRO iteration %d/%d", iteration + 1, config.n_iterations)

        # Compute Nash over current population
        nash_weights = _nash_weights(payoff)
        logger.info("Nash weights: %s", [round(w, 3) for w in nash_weights])

        # Run best-response oracle
        best, history = best_response_vs_mixture(
            opponents=opponent_teams,
            weights=nash_weights,
            config=config.search_config,
            space=space,
        )
        logger.info(
            "Best response win rate: %.3f (after %d rounds)",
            best.win_rate,
            len(history),
        )
        _log_search_history(history, config.output_dir, iteration)

        # Evaluate new team against every existing team to extend payoff matrix
        new_col = _eval_new_team_vs_population(
            new_team=best,
            opponent_teams=opponent_teams,
            config=config,
        )
        # new_team vs itself = 0.5 by convention
        new_row = np.array([1.0 - v for v in new_col] + [0.5])
        new_col_full = np.array(new_col + [0.5])

        payoff = np.hstack([payoff, new_col_full[:-1].reshape(-1, 1)])
        payoff = np.vstack([payoff, new_row.reshape(1, -1)])

        opponent_teams.append(best.to_packed_team())
        found_teams.append(best)

        _save_payoff(payoff, config.output_dir, iteration)
        _save_teams(found_teams, config.output_dir, iteration)

    # Final Nash weights over the full population
    final_weights = _nash_weights(payoff)
    # found_teams corresponds to indices 1..n in opponent_teams (index 0 is meta)
    found_with_weights = list(zip(found_teams, final_weights[1:]))
    found_with_weights.sort(key=lambda x: x[1], reverse=True)

    logger.info("PSRO complete. Final Nash weights (found teams only):")
    for t, w in found_with_weights:
        logger.info("  win_rate=%.3f, nash_weight=%.3f", t.win_rate or 0.0, w)

    return [t for t, _ in found_with_weights]


def _nash_weights(payoff: np.ndarray) -> list[float]:
    """Compute Nash equilibrium distribution over rows of the payoff matrix."""
    try:
        weights = Game(payoff).linear_program()[0].tolist()
    except Exception:
        # Fall back to uniform if Nash computation fails (e.g. singular matrix)
        n = payoff.shape[0]
        weights = [1.0 / n] * n
    return weights


def _eval_new_team_vs_population(
    new_team: CandidateTeam,
    opponent_teams: list[str],
    config: PSROConfig,
) -> list[float]:
    """
    Evaluate new_team against each existing opponent team.

    Returns win rates of new_team vs opponent_teams[i] for each i.
    """
    win_rates: list[float] = []
    for opp in opponent_teams:
        evaluator = TeamEvaluator(
            opponent_team=opp,
            n_battles=config.search_config.n_battles,
            port=config.port,
            battle_agent_path=config.search_config.battle_agent_path,
            device=config.search_config.device,
            reg=config.reg,
        )
        scored = evaluator.evaluate(new_team)
        win_rates.append(scored.win_rate or 0.0)
    return win_rates


def _save_payoff(payoff: np.ndarray, output_dir: Path, iteration: int) -> None:
    path = output_dir / f"payoff_iter{iteration:03d}.json"
    with path.open("w") as f:
        json.dump([[round(v, 3) for v in row] for row in payoff.tolist()], f, indent=2)


def _save_teams(teams: list[CandidateTeam], output_dir: Path, iteration: int) -> None:
    path = output_dir / f"teams_iter{iteration:03d}.txt"
    with path.open("w") as f:
        for i, team in enumerate(teams):
            f.write(f"# Team {i + 1}  win_rate={team.win_rate:.3f}\n")
            f.write(team.to_showdown_text())
            f.write("\n\n---\n\n")


def _log_search_history(
    history: list[RoundRecord], output_dir: Path, iteration: int
) -> None:
    path = output_dir / f"search_history_iter{iteration:03d}.json"
    with path.open("w") as f:
        json.dump(
            [
                {
                    "round": r.round,
                    "best_win_rate": round(r.best_win_rate, 4),
                    "mean_win_rate": round(r.mean_win_rate, 4),
                    "cache_hits": r.cache_hits,
                    "cache_misses": r.cache_misses,
                }
                for r in history
            ],
            f,
            indent=2,
        )
