"""
Best-response oracle for the VGC-Bench team builder.

Uses a (population + candidates) search loop to find the team with the highest
win rate against a fixed opponent or a Nash mixture of opponents.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.evaluator import TeamEvaluator
from vgc_bench.team_builder.team import (
    TEAM_SIZE,
    CandidateTeam,
    combine_teams,
    mutate_build,
    random_team,
    swap_member,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """
    Configuration for the evolutionary best-response search.

    Attributes:
        rounds:             Number of search rounds.
        population:         Teams retained as elites per round (mu).
        candidates:         New teams evaluated per round (lambda).
        n_battles:          Battles used to score each candidate team.
        port:               Showdown server port.
        battle_agent_path:  PPO checkpoint zip; None uses SimpleHeuristicsPlayer.
        device:             PyTorch device string.
        swap_probability:   Probability of swap_member vs other operators.
        mutate_probability: Probability of mutate_build vs combine_teams.
            combine_teams probability = 1 - swap_probability - mutate_probability.
        n_swaps:            Members replaced per swap_member call.
        diversity_threshold: Minimum dissimilarity between retained elites.
            Set to 0.0 to disable diversity filtering.
        seed:               Random seed; None means non-deterministic.
        reg:                Regulation pool to sample from (default "ma").
    """

    rounds: int = 10
    population: int = 20
    candidates: int = 40
    n_battles: int = 20
    port: int = 8100
    battle_agent_path: Path | None = None
    device: str = "cpu"
    swap_probability: float = 0.3
    mutate_probability: float = 0.3
    n_swaps: int = 1
    diversity_threshold: float = 0.0
    seed: int | None = None
    reg: str = "ma"


@dataclass
class RoundRecord:
    """Snapshot of one search round's results."""

    round: int
    best_win_rate: float
    mean_win_rate: float
    cache_hits: int
    cache_misses: int


def best_response(
    opponent_team: str,
    config: SearchConfig | None = None,
    space: BuildSpace | None = None,
) -> tuple[CandidateTeam, list[RoundRecord]]:
    """
    Find the best team against a single fixed opponent.

    This is the oracle used by one PSRO iteration.

    Args:
        opponent_team: Packed team string for the fixed opponent.
        config: Search configuration. Defaults to SearchConfig().
        space: Pre-built BuildSpace. Loads from disk when None.

    Returns:
        Tuple of (best_team, history) where best_team has the highest win_rate
        seen across all rounds and history is one RoundRecord per round.
    """
    if config is None:
        config = SearchConfig()
    if space is None:
        space = BuildSpace.from_regulation(config.reg)

    evaluator = TeamEvaluator(
        opponent_team=opponent_team,
        n_battles=config.n_battles,
        port=config.port,
        battle_agent_path=config.battle_agent_path,
        device=config.device,
        reg=config.reg,
    )
    return _run_search(evaluator, config, space)


def best_response_vs_mixture(
    opponents: list[str],
    weights: list[float],
    config: SearchConfig | None = None,
    space: BuildSpace | None = None,
) -> tuple[CandidateTeam, list[RoundRecord]]:
    """
    Find the best team against a Nash mixture of opponent teams.

    Args:
        opponents: List of packed opponent team strings.
        weights: Probability weights for each opponent (must sum to ~1).
        config: Search configuration. Defaults to SearchConfig().
        space: Pre-built BuildSpace. Loads from disk when None.

    Returns:
        Tuple of (best_team, history).
    """
    if config is None:
        config = SearchConfig()
    if space is None:
        space = BuildSpace.from_regulation(config.reg)

    # Use the first opponent as the nominal evaluator opponent; evaluate_vs_mixture
    # overrides this per call.
    evaluator = TeamEvaluator(
        opponent_team=opponents[0],
        n_battles=config.n_battles,
        port=config.port,
        battle_agent_path=config.battle_agent_path,
        device=config.device,
        reg=config.reg,
    )

    def mixture_evaluate(team: CandidateTeam) -> CandidateTeam:
        return evaluator.evaluate_vs_mixture(team, opponents, weights)

    return _run_search(evaluator, config, space, evaluate_fn=mixture_evaluate)


def _run_search(
    evaluator: TeamEvaluator,
    config: SearchConfig,
    space: BuildSpace,
    evaluate_fn=None,
) -> tuple[CandidateTeam, list[RoundRecord]]:
    """
    Core (population + candidates) search loop.

    Args:
        evaluator:   TeamEvaluator for single-opponent evaluation.
        config:      Search configuration.
        space:       BuildSpace for team generation.
        evaluate_fn: Override evaluation function. Defaults to evaluator.evaluate.

    Returns:
        Tuple of (overall_best, history).
    """
    if evaluate_fn is None:
        evaluate_fn = evaluator.evaluate

    rng = random.Random(config.seed)
    cache: dict[tuple, float] = {}
    history: list[RoundRecord] = []
    overall_best: CandidateTeam | None = None

    # Initialise population
    elites = [random_team(space, rng) for _ in range(config.population)]
    elites, hits, misses = _evaluate_batch(elites, evaluate_fn, cache)
    elites = _select(elites, config.population, config.diversity_threshold)
    overall_best = elites[0]

    record = RoundRecord(
        round=0,
        best_win_rate=elites[0].win_rate,  # type: ignore[arg-type]
        mean_win_rate=sum(t.win_rate for t in elites) / len(elites),  # type: ignore[arg-type]
        cache_hits=hits,
        cache_misses=misses,
    )
    history.append(record)
    logger.info(
        "Round 0 — best: %.3f, mean: %.3f, cache: %d/%d",
        record.best_win_rate,
        record.mean_win_rate,
        hits,
        misses,
    )

    for r in range(1, config.rounds + 1):
        new_candidates = _produce_candidates(
            elites,
            space,
            config.candidates,
            config.swap_probability,
            config.mutate_probability,
            config.n_swaps,
            rng,
        )
        new_candidates, hits, misses = _evaluate_batch(
            new_candidates, evaluate_fn, cache
        )
        combined = elites + new_candidates
        elites = _select(combined, config.population, config.diversity_threshold)

        if elites[0].win_rate > overall_best.win_rate:  # type: ignore[operator]
            overall_best = elites[0]

        record = RoundRecord(
            round=r,
            best_win_rate=elites[0].win_rate,  # type: ignore[arg-type]
            mean_win_rate=sum(t.win_rate for t in elites) / len(elites),  # type: ignore[arg-type]
            cache_hits=hits,
            cache_misses=misses,
        )
        history.append(record)
        logger.info(
            "Round %d — best: %.3f, mean: %.3f, cache: %d/%d",
            r,
            record.best_win_rate,
            record.mean_win_rate,
            hits,
            misses,
        )

    return overall_best, history


def _evaluate_batch(
    teams: list[CandidateTeam],
    evaluate_fn,
    cache: dict[tuple, float],
) -> tuple[list[CandidateTeam], int, int]:
    """Evaluate teams, using cache to skip already-scored ones."""
    results: list[CandidateTeam] = []
    hits = 0
    misses = 0
    for team in teams:
        key = team.members
        if key in cache:
            results.append(team.with_win_rate(cache[key]))
            hits += 1
        else:
            scored = evaluate_fn(team)
            cache[key] = scored.win_rate  # type: ignore[assignment]
            results.append(scored)
            misses += 1
    return results, hits, misses


def _select(
    teams: list[CandidateTeam],
    k: int,
    diversity_threshold: float = 0.0,
) -> list[CandidateTeam]:
    """
    Select the top-k teams by win_rate, with optional diversity filtering.

    When diversity_threshold > 0, greedily keeps a team only if it differs
    from all already-selected teams by more than the threshold. Falls back to
    top-by-win-rate if too few diverse teams are found.
    """
    sorted_teams = sorted(
        teams, key=lambda t: t.win_rate or 0.0, reverse=True
    )
    if diversity_threshold <= 0.0:
        return sorted_teams[:k]

    selected: list[CandidateTeam] = []
    for team in sorted_teams:
        if len(selected) >= k:
            break
        if all(team.similarity_to(s) < diversity_threshold for s in selected):
            selected.append(team)

    # Fill remaining slots if diversity filter was too aggressive
    if len(selected) < k:
        remaining = [t for t in sorted_teams if t not in selected]
        selected.extend(remaining[: k - len(selected)])

    return selected[:k]


def _produce_candidates(
    elites: list[CandidateTeam],
    space: BuildSpace,
    n: int,
    swap_probability: float,
    mutate_probability: float,
    n_swaps: int,
    rng: random.Random,
) -> list[CandidateTeam]:
    """Generate n new candidate teams from the current elite population."""
    candidates: list[CandidateTeam] = []
    for _ in range(n):
        r = rng.random()
        if r < swap_probability or len(elites) < 2:
            parent = rng.choice(elites)
            candidates.append(swap_member(parent, space, n_swaps=n_swaps, rng=rng))
        elif r < swap_probability + mutate_probability:
            parent = rng.choice(elites)
            slot = rng.randrange(TEAM_SIZE)
            new_members = list(parent.members)
            forbidden = {new_members[i].item for i in range(TEAM_SIZE) if i != slot}
            new_members[slot] = mutate_build(new_members[slot], space, rng, forbidden_items=forbidden)
            candidates.append(CandidateTeam(members=tuple(new_members)))
        else:
            pa, pb = rng.sample(elites, 2)
            candidates.append(combine_teams(pa, pb, space, rng=rng))
    return candidates
