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
from vgc_bench.team_builder.run_log import write_run_meta
from vgc_bench.team_builder.team import CandidateTeam, team_to_dict, team_from_dict

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
    reg: str = "i"
    output_dir: Path = Path("results/team_builder")


def run_psro(
    config: PSROConfig,
    meta_teams: list[str] | str,
    space: BuildSpace | None = None,
    resume: bool = True,
    # Backward-compat alias (single string)
    meta_team: str | None = None,
) -> list[CandidateTeam]:
    """
    Run the PSRO adversarial team-building loop.

    Seeds the population with one or more fixed meta opponent teams, then
    iteratively adds best-response teams. Converges toward teams that perform
    well against the Nash equilibrium mixture of the full population.

    Args:
        config:      PSRO configuration.
        meta_teams:  One packed team string, or a list of packed team strings
                     to use as the initial seed population. When multiple teams
                     are provided the initial payoff is seeded with 0.5 for all
                     pairs (uniform Nash bootstrap) and the first best-response
                     is found against the Nash mixture of all initial teams.
        space:       Pre-built BuildSpace. Loads from disk when None.
        resume:      If True, resume from an existing checkpoint in output_dir.

    Returns:
        All discovered CandidateTeams sorted by final Nash weight (descending).
        Initial meta teams are not included (they have no CandidateTeam wrapper).

    Algorithm:
        1. initial_teams = meta_teams, payoff = 0.5 * ones(N, N)
        2. For each iteration:
            a. Compute Nash distribution over current population.
            b. Find best_response_vs_mixture(teams, nash_weights).
            c. Evaluate new_team vs all existing teams → extend payoff.
            d. Append new_team to population; recompute Nash.
            e. Save payoff matrix and team texts to output_dir.
        3. Return discovered teams sorted by Nash weight descending.
    """
    # Normalise to list; support legacy single-string kwarg
    if meta_team is not None and not meta_teams:
        meta_teams = [meta_team]
    if isinstance(meta_teams, str):
        meta_teams = [meta_teams]

    if space is None:
        space = BuildSpace.from_regulation(config.reg)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_meta(
        config.output_dir,
        run_type="psro",
        reg=config.reg,
        target_iterations=config.n_iterations,
        rounds_per_iteration=config.search_config.rounds,
        population=config.search_config.population,
        candidates=config.search_config.candidates,
        n_battles=config.search_config.n_battles,
        battle_agent_path=(
            str(config.search_config.battle_agent_path)
            if config.search_config.battle_agent_path else None
        ),
        device=config.search_config.device,
        n_meta_teams=len(meta_teams),
    )

    _ckpt_path = config.output_dir / "psro_checkpoint.json"
    completed_iterations = 0

    if resume and _ckpt_path.exists():
        logger.info("Found checkpoint — resuming PSRO from %s", _ckpt_path)
        _ckpt = json.loads(_ckpt_path.read_text(encoding="utf-8"))
        completed_iterations = _ckpt["completed_iterations"]
        opponent_teams = _ckpt["opponent_teams"]
        payoff = np.array(_ckpt["payoff"])
        found_teams = [team_from_dict(d) for d in _ckpt["found_teams"]]
        n_initial = _ckpt.get("n_initial", 1)  # legacy checkpoints had 1 initial team
        logger.info(
            "Resumed: %d/%d iterations done, population size %d.",
            completed_iterations, config.n_iterations, len(opponent_teams),
        )
    else:
        # Seed with all provided meta teams
        opponent_teams = list(meta_teams)
        n_initial = len(opponent_teams)
        # found_teams: CandidateTeam objects for teams we discovered (not the seeds)
        found_teams = []
        # Bootstrap payoff: 0.5 for every pair (uniform Nash → equal initial weights)
        payoff = np.full((n_initial, n_initial), 0.5)
        logger.info(
            "Starting fresh PSRO with %d initial meta team(s).", n_initial
        )

    for iteration in range(completed_iterations, config.n_iterations):
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
            snapshot_dir=config.output_dir / f"search_iter{iteration:03d}",
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
        completed_iterations = iteration + 1
        _save_psro_checkpoint(_ckpt_path, completed_iterations, opponent_teams, payoff, found_teams, n_initial)

    # Final Nash weights over the full population
    final_weights = _nash_weights(payoff)
    # found_teams corresponds to indices n_initial..end in opponent_teams
    found_with_weights = list(zip(found_teams, final_weights[n_initial:]))
    found_with_weights.sort(key=lambda x: x[1], reverse=True)

    logger.info("PSRO complete. Final Nash weights (found teams only):")
    for t, w in found_with_weights:
        logger.info("  win_rate=%.3f, nash_weight=%.3f", t.win_rate or 0.0, w)

    return [t for t, _ in found_with_weights]


def _save_psro_checkpoint(
    path: Path,
    completed_iterations: int,
    opponent_teams: list[str],
    payoff: np.ndarray,
    found_teams: list[CandidateTeam],
    n_initial: int = 1,
) -> None:
    payload = {
        "type": "psro",
        "completed_iterations": completed_iterations,
        "n_initial": n_initial,
        "opponent_teams": opponent_teams,
        "payoff": payoff.tolist(),
        "found_teams": [team_to_dict(t) for t in found_teams],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


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
