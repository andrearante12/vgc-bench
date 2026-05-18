"""
CLI entry point for the VGC-Bench team builder.

Usage:
    python -m vgc_bench.team_builder best_response --meta-team PATH [options]
    python -m vgc_bench.team_builder psro --meta-team PATH [options]

The --meta-team argument accepts a path to a Showdown-format .txt team file.
The file is read, converted to packed format, and used as the initial opponent.

Examples:
    python -m vgc_bench.team_builder best_response \\
        --meta-team teams/reg_ma/PC1.txt \\
        --rounds 5 --n-battles 10 --port 8100

    python -m vgc_bench.team_builder psro \\
        --meta-team teams/reg_ma/PC1.txt \\
        --psro-iterations 5 --rounds 3 --n-battles 10
"""

import argparse
import logging
import sys
from pathlib import Path

from poke_env.teambuilder import Teambuilder

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.optimizer import SearchConfig, best_response
from vgc_bench.team_builder.psro import PSROConfig, run_psro


def _load_packed_team(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(text))


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--meta-team", required=True,
        help="Path to a Showdown .txt team file used as the initial opponent",
    )
    parser.add_argument("--reg", default="ma", help="VGC regulation (default: ma)")
    parser.add_argument("--rounds", type=int, default=10, help="Search rounds per iteration")
    parser.add_argument("--population", type=int, default=20, help="Elite team population size")
    parser.add_argument("--candidates", type=int, default=40, help="Candidates per round")
    parser.add_argument("--n-battles", type=int, default=20, help="Battles per team evaluation")
    parser.add_argument("--port", type=int, default=8100, help="Showdown server port")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", default="results/team_builder", help="Output directory")
    parser.add_argument("--swap-probability", type=float, default=0.3)
    parser.add_argument("--mutate-probability", type=float, default=0.3)
    parser.add_argument("--diversity-threshold", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")


def _make_search_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        rounds=args.rounds,
        population=args.population,
        candidates=args.candidates,
        n_battles=args.n_battles,
        port=args.port,
        swap_probability=args.swap_probability,
        mutate_probability=args.mutate_probability,
        diversity_threshold=args.diversity_threshold,
        seed=args.seed,
        reg=args.reg,
    )


def cmd_best_response(args: argparse.Namespace) -> None:
    meta_packed = _load_packed_team(args.meta_team)
    space = BuildSpace.from_regulation(args.reg)
    config = _make_search_config(args)

    print(f"Running best_response: {config.rounds} rounds, "
          f"{config.population} pop, {config.candidates} candidates, "
          f"{config.n_battles} battles/team")

    best, history = best_response(meta_packed, config, space)

    print(f"\nBest team win rate: {best.win_rate:.1%}")
    print(f"Search history ({len(history)} rounds):")
    for r in history:
        print(f"  Round {r.round:2d}: best={r.best_win_rate:.3f} "
              f"mean={r.mean_win_rate:.3f} "
              f"cache={r.cache_hits}H/{r.cache_misses}M")
    print("\n=== Best Team ===")
    print(best.to_showdown_text())


def cmd_psro(args: argparse.Namespace) -> None:
    meta_packed = _load_packed_team(args.meta_team)
    space = BuildSpace.from_regulation(args.reg)
    search_config = _make_search_config(args)
    psro_config = PSROConfig(
        n_iterations=args.psro_iterations,
        search_config=search_config,
        port=args.port,
        reg=args.reg,
        output_dir=Path(args.output),
    )

    print(f"Running PSRO: {psro_config.n_iterations} iterations")
    found = run_psro(psro_config, meta_team=meta_packed, space=space)

    print(f"\nFound {len(found)} team(s):")
    for i, team in enumerate(found, 1):
        print(f"\n--- Team {i} (win_rate={team.win_rate:.3f}) ---")
        print(team.to_showdown_text())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGC-Bench team builder optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    br_parser = sub.add_parser("best_response", help="Find best team vs a single meta team")
    _add_common_args(br_parser)

    psro_parser = sub.add_parser("psro", help="Run PSRO adversarial team-building loop")
    _add_common_args(psro_parser)
    psro_parser.add_argument("--psro-iterations", type=int, default=10,
                              help="Number of PSRO iterations")

    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    if args.mode == "best_response":
        cmd_best_response(args)
    elif args.mode == "psro":
        cmd_psro(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
