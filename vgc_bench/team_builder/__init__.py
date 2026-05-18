"""
VGC-Bench Team Builder.

PSRO-based optimizer for VGC team building. Generates teams from scratch using
a BuildSpace loaded from tournament-validated Pokémon data, then searches for
team compositions that are competitive against the reg_ma meta.

Public API:
    PokemonBuild     — immutable single-Pokémon build (species, moves, EVs, …)
    CandidateTeam    — immutable team of 6 PokemonBuilds with optional win_rate
    BuildSpace       — valid search space for a regulation (species/moves/items/…)
    SearchConfig     — configuration for the evolutionary best-response oracle
    PSROConfig       — configuration for the full PSRO loop
    RoundRecord      — snapshot of one search round's results
    best_response    — find the best team against a fixed opponent
    run_psro         — run the full PSRO adversarial team-building loop

Example:
    from vgc_bench.team_builder import BuildSpace, SearchConfig, best_response
    space = BuildSpace.from_regulation("ma")
    best, history = best_response(opponent_packed_team, SearchConfig(rounds=5, n_battles=10), space)
    print(f"Best team win rate: {best.win_rate:.1%}")
    print(best.to_showdown_text())
"""

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.optimizer import RoundRecord, SearchConfig, best_response
from vgc_bench.team_builder.pokemon_build import PokemonBuild
from vgc_bench.team_builder.psro import PSROConfig, run_psro
from vgc_bench.team_builder.team import CandidateTeam

__all__ = [
    "BuildSpace",
    "CandidateTeam",
    "PSROConfig",
    "PokemonBuild",
    "RoundRecord",
    "SearchConfig",
    "best_response",
    "run_psro",
]
