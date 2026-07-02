"""
CLI entry point for the VGC-Bench team builder.

Usage:
    python -m vgc_bench.team_builder best_response --meta-team PATH [PATH ...] [options]
    python -m vgc_bench.team_builder psro --meta-team PATH [PATH ...] [options]

--meta-team accepts one or more paths to Showdown-format .txt team files.
  best_response: single team = fixed opponent; multiple = uniform mixture opponent.
  psro:          single team = one seed; multiple = initial pool (uniform Nash bootstrap).

Examples:
    python -m vgc_bench.team_builder best_response \
        --meta-team teams/reg_i/featured/I1146.txt \
        --rounds 5 --n-battles 10 --port 8100

    python -m vgc_bench.team_builder psro \
        --meta-team teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
                    teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt \
        --psro-iterations 8 --rounds 5 --n-battles 20
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
        "--meta-team", required=True, nargs="+",
        help=(
            "Path(s) to Showdown .txt team file(s) used as opponents. "
            "Provide one path for a single fixed opponent, or multiple paths "
            "(space-separated) to seed PSRO with a pool of meta teams."
        ),
    )
    parser.add_argument("--reg", default="i", help="VGC regulation (default: i)")
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
    parser.add_argument("--battle-agent-path", default=None, help="Path to a trained PPO .zip used as the battle agent (default: SimpleHeuristicsPlayer)")
    parser.add_argument("--device", default="cpu", help="PyTorch device for the battle agent (default: cpu)")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")


def _make_search_config(args: argparse.Namespace) -> SearchConfig:
    from pathlib import Path
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
        battle_agent_path=Path(args.battle_agent_path) if args.battle_agent_path else None,
        device=args.device,
    )


def cmd_best_response(args: argparse.Namespace) -> None:
    from vgc_bench.team_builder.optimizer import best_response_vs_mixture
    meta_paths = args.meta_team  # list of paths (nargs='+')
    meta_packed_list = [_load_packed_team(p) for p in meta_paths]
    space = BuildSpace.from_regulation(args.reg)
    config = _make_search_config(args)

    print(f"Running best_response vs {len(meta_packed_list)} meta team(s): "
          f"{config.rounds} rounds, {config.population} pop, "
          f"{config.candidates} candidates, {config.n_battles} battles/team")

    if len(meta_packed_list) == 1:
        best, history = best_response(
            meta_packed_list[0], config, space, snapshot_dir=Path(args.output)
        )
    else:
        weights = [1.0 / len(meta_packed_list)] * len(meta_packed_list)
        best, history = best_response_vs_mixture(
            meta_packed_list, weights, config, space, snapshot_dir=Path(args.output)
        )

    print(f"\nBest team win rate: {best.win_rate:.1%}")
    print(f"Search history ({len(history)} rounds):")
    for r in history:
        print(f"  Round {r.round:2d}: best={r.best_win_rate:.3f} "
              f"mean={r.mean_win_rate:.3f} "
              f"cache={r.cache_hits}H/{r.cache_misses}M")
    print("\n=== Best Team ===")
    print(best.to_showdown_text())


def cmd_psro(args: argparse.Namespace) -> None:
    meta_paths = args.meta_team  # list of paths (nargs='+')
    meta_packed_list = [_load_packed_team(p) for p in meta_paths]
    space = BuildSpace.from_regulation(args.reg)
    search_config = _make_search_config(args)
    psro_config = PSROConfig(
        n_iterations=args.psro_iterations,
        search_config=search_config,
        port=args.port,
        reg=args.reg,
        output_dir=Path(args.output),
    )

    print(f"Running PSRO: {psro_config.n_iterations} iterations, "
          f"seeded with {len(meta_packed_list)} meta team(s)")
    found = run_psro(
        psro_config, meta_teams=meta_packed_list, space=space, resume=not args.no_resume
    )

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
    psro_parser.add_argument("--no-resume", action="store_true", default=False,
                             help="Ignore any existing checkpoint and start fresh.")

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
