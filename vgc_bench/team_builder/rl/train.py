"""
SP-PSRO training loop for the RL team builder.

Runs Self-Play PSRO (arXiv:2207.06541) starting from a fixed set of meta
teams. Each iteration trains two networks via REINFORCE:

  β  — best response to the current Nash mixture of all population teams.
  ν  — self-play response; trained against β's generated teams.

Both β's best team and ν's best team are added to the population each
iteration. Nash weights are recomputed after each extension. After all
iterations, the team with highest Nash weight is the recommended deploy.

Usage:
    python -m vgc_bench.team_builder.rl.train \\
        --meta-teams teams/reg_i/I1.txt teams/reg_i/I2.txt \\
                     teams/reg_i/I3.txt teams/reg_i/I4.txt \\
        --n-steps 10000 --iterations 5 --port 8100 --output results/rl_psro
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.psro import _nash_weights
from vgc_bench.team_builder.rl.bc_init import apply_bc_biases
from vgc_bench.team_builder.rl.team_builder_env import TeamBuilderBattleEnv
from vgc_bench.team_builder.rl.team_builder_network import TeamBuilderNetwork
from vgc_bench.team_builder.team import CandidateTeam, team_to_dict, team_from_dict

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# REINFORCE training
# ------------------------------------------------------------------ #

_ENTROPY_COEF = 0.05
_VALUE_COEF = 0.5
_LR = 1e-4
_GAMMA = 1.0  # episodic: no discounting needed


def train_policy(
    population_texts: list[str],
    nash_weights: list[float],
    space: BuildSpace,
    n_steps: int = 5_000,
    port: int = 8100,
    n_battles: int = 20,
    reg: str = "i",
    policy_checkpoint: str | None = None,
    device: str = "cpu",
    log_every: int = 100,
    eval_fn: "Callable[[CandidateTeam, str], float] | None" = None,
    snapshot_dir: Path | None = None,
    snapshot_every: int = 500,
    resume_checkpoint: Path | None = None,
    battle_agent_path: Path | None = None,
) -> tuple[TeamBuilderNetwork, CandidateTeam]:
    """
    Train a TeamBuilderNetwork via REINFORCE against a fixed opponent mixture.

    At each step:
      1. Sample one opponent from the population (weighted by Nash weights).
      2. Generate a team stochastically.
      3. Battle the team against the opponent → reward ∈ {0, 1}.
      4. Subtract value-head baseline; update with policy-gradient loss.

    Args:
        population_texts:  Showdown-format team strings, one per pop member.
        nash_weights:      Nash mixture weights (same length as population_texts).
        space:             BuildSpace for the regulation.
        n_steps:           Training steps (team generations).
        port:              Showdown server port.
        n_battles:         Battles per evaluation.
        reg:               VGC regulation identifier.
        policy_checkpoint: Path to a PPO .zip to load frozen embeddings.
        device:            PyTorch device string.
        log_every:         Log win-rate every N steps.
        eval_fn:           Optional callable (team, opponent_text) → win_rate in
                           [0, 1]. When provided, replaces live Showdown battles.
                           Useful for unit tests and smoke tests.

    Returns:
        (network, best_team) — trained network and the highest-scoring team
        found during training.
    """
    from typing import Callable  # noqa: F401 — used in type annotation above

    network = TeamBuilderNetwork(space).to(device)
    apply_bc_biases(network, space)
    if policy_checkpoint:
        network.load_frozen_embeds(policy_checkpoint, device=device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, network.parameters()), lr=_LR
    )

    import random
    w_tensor = torch.tensor(nash_weights, dtype=torch.float32)
    w_norm = w_tensor / w_tensor.sum()

    _resume_step = 0
    _best_win_rate_init = -1.0
    _best_team_init: CandidateTeam | None = None
    _win_rate_window_init: list[float] = []
    if resume_checkpoint is not None and Path(resume_checkpoint).exists():
        logger.info("Resuming train_policy from %s", resume_checkpoint)
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["network_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        _resume_step = ckpt["step"]
        _best_win_rate_init = ckpt["best_win_rate"]
        _win_rate_window_init = ckpt.get("win_rate_window", [])
        if ckpt.get("best_team_dict") is not None:
            _best_team_init = team_from_dict(ckpt["best_team_dict"])

    _snap_dir: Path | None = None
    _metrics_f = None
    if snapshot_dir is not None:
        _snap_dir = Path(snapshot_dir)
        _snap_dir.mkdir(parents=True, exist_ok=True)
        (_snap_dir / "snapshots").mkdir(exist_ok=True)
        _metrics_f = (_snap_dir / "metrics.jsonl").open("w", encoding="utf-8")
    _recent_species: list[list[str]] = []

    # Set up evaluation: either real battles or the injected eval_fn
    if eval_fn is not None:
        def _evaluate(team: CandidateTeam, opp_text: str) -> float:
            return eval_fn(team, opp_text)
        env = None
    else:
        initial_opp_packed = _text_to_packed(population_texts[0])
        env = TeamBuilderBattleEnv(
            initial_opponent=initial_opp_packed,
            n_battles=n_battles,
            port=port,
            reg=reg,
            battle_agent_path=battle_agent_path,
            device=device,
        )

        def _evaluate(team: CandidateTeam, opp_text: str) -> float:
            packed = _text_to_packed(opp_text)
            env.set_opponent(packed)
            return env.evaluate(team)

    best_team: CandidateTeam | None = _best_team_init
    best_win_rate = _best_win_rate_init
    win_rate_window: list[float] = list(_win_rate_window_init)

    _was_interrupted = False
    for step in range(_resume_step, n_steps):
        # Sample one opponent from the Nash mixture
        opp_idx = torch.multinomial(w_norm, 1).item()
        opp_text = population_texts[opp_idx]

        # Generate team
        network.train()
        team, log_prob, value = network.generate(population_texts, nash_weights)

        # Evaluate
        logger.debug("generated team:\n%s", team.to_showdown_text())
        try:
            win_rate = _evaluate(team, opp_text)
        except KeyboardInterrupt:
            logger.warning("train_policy interrupted at step %d", step)
            _was_interrupted = True
            break
        reward = float(win_rate)

        # REINFORCE with baseline
        advantage = reward - value.detach().item()
        policy_loss = -log_prob * advantage
        value_loss = _VALUE_COEF * (value - reward) ** 2

        # Entropy bonus: add back entropy to encourage exploration
        # (log_prob already summed; approximate entropy via -log_prob/n_choices)
        entropy_loss = _ENTROPY_COEF * log_prob  # maximise entropy = minimize log_prob

        loss = policy_loss + value_loss + entropy_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        win_rate_window.append(win_rate)
        _recent_species.append([m.species for m in team.members])
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_team = team.with_win_rate(win_rate)

        if (step + 1) % log_every == 0:
            recent = sum(win_rate_window[-log_every:]) / min(log_every, len(win_rate_window))
            logger.info(
                "step %5d/%d  recent_win_rate=%.3f  best=%.3f",
                step + 1, n_steps, recent, best_win_rate,
            )
            if _metrics_f is not None:
                counts: dict[str, int] = {}
                for slist in _recent_species[-log_every:]:
                    for s in slist:
                        counts[s] = counts.get(s, 0) + 1
                top_sp = sorted(counts, key=lambda s: -counts[s])[:12]
                _metrics_f.write(json.dumps({
                    "step": step + 1,
                    "recent_win_rate": round(recent, 4),
                    "best_win_rate": round(best_win_rate, 4),
                    "policy_loss": round(policy_loss.item(), 6),
                    "value_loss": round(value_loss.item(), 6),
                    "neg_log_prob": round(-log_prob.item(), 4),
                    "top_species": top_sp,
                }) + "\n")
                _metrics_f.flush()

        if _snap_dir is not None and (step + 1) % snapshot_every == 0 and best_team is not None:
            snap = _snap_dir / "snapshots" / f"step_{step+1:06d}.txt"
            snap.write_text(
                f"# step={step+1}  win_rate={best_team.win_rate:.3f}\n"
                + best_team.to_showdown_text(),
                encoding="utf-8",
            )
            _save_policy_checkpoint(
                _snap_dir / "policy.pt", step + 1,
                best_win_rate, best_team, win_rate_window, network, optimizer,
            )

    if _was_interrupted:
        if _snap_dir is not None and best_team is not None:
            _save_policy_checkpoint(
                _snap_dir / "policy.pt", step,
                best_win_rate, best_team, win_rate_window, network, optimizer,
            )
            (_snap_dir / "interrupted_best_team.txt").write_text(
                best_team.to_showdown_text(), encoding="utf-8"
            )
            logger.warning("Saved interrupted state to %s", _snap_dir)
        if _metrics_f is not None:
            _metrics_f.close()
        raise KeyboardInterrupt()

    if best_team is None:
        # No team evaluated — generate one deterministically
        best_team = network.generate_greedy(population_texts, nash_weights).with_win_rate(0.0)

    if _metrics_f is not None:
        _metrics_f.close()

    return network, best_team


# ------------------------------------------------------------------ #
# Checkpoint helpers
# ------------------------------------------------------------------ #

def _save_policy_checkpoint(
    path: Path,
    step: int,
    best_win_rate: float,
    best_team: CandidateTeam,
    win_rate_window: list[float],
    network: TeamBuilderNetwork,
    optimizer: optim.Adam,
) -> None:
    payload = {
        "step": step,
        "best_win_rate": best_win_rate,
        "best_team_showdown": best_team.to_showdown_text(),
        "best_team_dict": team_to_dict(best_team),
        "win_rate_window": list(win_rate_window[-100:]),
        "network_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_sp_psro_checkpoint(
    path: Path,
    completed_iterations: int,
    k0: int,
    population_texts: list[str],
    payoff: "np.ndarray",
    nash_w: list[float],
    found_teams: list[CandidateTeam],
    in_progress: dict | None,
) -> None:
    payload = {
        "type": "sp_psro",
        "completed_iterations": completed_iterations,
        "k0": k0,
        "population_texts": population_texts,
        "payoff": payoff.tolist(),
        "nash_w": list(nash_w),
        "found_teams": [team_to_dict(t) for t in found_teams],
        "in_progress": in_progress,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------ #
# SP-PSRO loop
# ------------------------------------------------------------------ #

def run_sp_psro(
    meta_team_paths: list[str],
    space: BuildSpace,
    n_iterations: int = 5,
    n_steps: int = 10_000,
    port: int = 8100,
    n_battles: int = 20,
    reg: str = "i",
    policy_checkpoint: str | None = None,
    output_dir: Path = Path("results/rl_psro"),
    quality_threshold: float = 0.5,
    snapshot_every: int = 500,
    resume: bool = True,
    battle_agent_path: Path | None = None,
    device: str = "cpu",
) -> list[CandidateTeam]:
    """
    Run SP-PSRO starting from meta teams.

    Each iteration:
      1. Train β against current Nash mixture → add β_team if it clears quality_threshold.
      2. Train ν against β_team alone → add ν_team if it clears quality_threshold.
      3. Evaluate both new teams vs all population members.
      4. Extend payoff matrix and recompute Nash.

    Args:
        quality_threshold: Minimum Nash-weighted win rate a generated team must achieve
                           against the current population to be added. Teams below this
                           threshold are discarded rather than polluting the payoff matrix.
                           Set to 0.0 to disable the gate.

    Returns:
        All generated teams (not the meta teams) sorted by final Nash weight.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _ckpt_path = output_dir / "psro_checkpoint.json"
    completed_iterations = 0
    k0: int
    in_progress: dict | None = None

    if resume and _ckpt_path.exists():
        logger.info("Found checkpoint — resuming SP-PSRO from %s", _ckpt_path)
        _ckpt = json.loads(_ckpt_path.read_text(encoding="utf-8"))
        completed_iterations = _ckpt["completed_iterations"]
        k0 = _ckpt["k0"]
        population_texts = _ckpt["population_texts"]
        payoff = np.array(_ckpt["payoff"])
        nash_w = _ckpt["nash_w"]
        found_teams = [team_from_dict(d) for d in _ckpt["found_teams"]]
        in_progress = _ckpt.get("in_progress")
        logger.info(
            "Resumed: %d/%d iterations done, population size %d.",
            completed_iterations, n_iterations, len(population_texts),
        )
    else:
        population_texts = [Path(p).read_text(encoding="utf-8") for p in meta_team_paths]
        k0 = len(population_texts)
        payoff = np.full((k0, k0), 0.5)
        nash_w = _nash_weights(payoff)
        found_teams = []

    for iteration in range(completed_iterations, n_iterations):
        logger.info("=== SP-PSRO iteration %d/%d ===", iteration + 1, n_iterations)

        # ---- Determine whether β is already done (mid-iteration resume) ----
        _skip_beta = False
        _beta_team_override: CandidateTeam | None = None
        if in_progress is not None and in_progress.get("iteration") == iteration:
            if in_progress.get("beta_done") and in_progress.get("beta_team") is not None:
                _skip_beta = True
                _beta_team_override = team_from_dict(in_progress["beta_team"])
                logger.info("Iteration %d: β done in prior run, restoring from checkpoint.", iteration)
        in_progress = None  # consume; only applies to the first matching iteration

        # ---- Train β ----
        logger.info("Training β (best response)…")
        if _skip_beta:
            beta_team = _beta_team_override
        else:
            _beta_snap = output_dir / f"beta_iter{iteration:03d}"
            _beta_net, beta_team = train_policy(
                population_texts=population_texts,
                nash_weights=nash_w,
                space=space,
                n_steps=n_steps,
                port=port,
                n_battles=n_battles,
                reg=reg,
                policy_checkpoint=policy_checkpoint,
                snapshot_dir=_beta_snap,
                snapshot_every=snapshot_every,
                resume_checkpoint=_beta_snap / "policy.pt",
                battle_agent_path=battle_agent_path,
                device=device,
            )
        logger.info("β best team win_rate=%.3f", beta_team.win_rate or 0.0)

        # Save in-progress checkpoint before ν so a ν-phase interrupt can skip β on resume
        _save_sp_psro_checkpoint(
            _ckpt_path, completed_iterations=iteration, k0=k0,
            population_texts=population_texts, payoff=payoff, nash_w=nash_w,
            found_teams=found_teams,
            in_progress={"iteration": iteration, "beta_done": True, "beta_team": team_to_dict(beta_team)},
        )

        # ---- Train ν (self-play against β) — never resumes; ν's opponent is stochastic ----
        logger.info("Training ν (self-play against β)…")
        beta_text = beta_team.to_showdown_text()
        _nu_net, nu_team = train_policy(
            population_texts=[beta_text],
            nash_weights=[1.0],
            space=space,
            n_steps=n_steps,
            port=port,
            n_battles=n_battles,
            reg=reg,
            policy_checkpoint=policy_checkpoint,
            snapshot_dir=output_dir / f"nu_iter{iteration:03d}",
            snapshot_every=snapshot_every,
            battle_agent_path=battle_agent_path,
            device=device,
        )
        logger.info("ν best team win_rate=%.3f", nu_team.win_rate or 0.0)

        # ---- Extend population ----
        for new_team in (beta_team, nu_team):
            new_text = new_team.to_showdown_text()
            new_col = _eval_vs_population(
                new_team, population_texts, port, n_battles, reg,
                battle_agent_path=battle_agent_path, device=device,
            )

            # Quality gate: Nash-weighted win rate must clear the threshold.
            # nash_w is kept in sync with population_texts (updated inside this loop
            # each time a team is added), so lengths always match.
            nash_win_rate = float(np.dot(new_col, nash_w))
            if nash_win_rate < quality_threshold:
                logger.info(
                    "Team Nash win_rate=%.3f below threshold %.2f — discarding.",
                    nash_win_rate, quality_threshold,
                )
                continue

            new_row = [1.0 - v for v in new_col] + [0.5]
            new_col_full = new_col + [0.5]

            payoff = np.hstack([payoff, np.array(new_col_full[:-1]).reshape(-1, 1)])
            payoff = np.vstack([payoff, np.array(new_row).reshape(1, -1)])
            population_texts.append(new_text)
            found_teams.append(new_team)
            nash_w = _nash_weights(payoff)
            logger.info(
                "Team added to population (Nash win_rate=%.3f). Population size: %d.",
                nash_win_rate, len(population_texts),
            )

        nash_w = _nash_weights(payoff)
        logger.info(
            "Population now %d teams. Nash weights: %s",
            len(population_texts),
            [round(w, 3) for w in nash_w],
        )
        _save_state(found_teams, payoff, nash_w, output_dir, iteration)
        completed_iterations = iteration + 1
        _save_sp_psro_checkpoint(
            _ckpt_path, completed_iterations=completed_iterations, k0=k0,
            population_texts=population_texts, payoff=payoff, nash_w=nash_w,
            found_teams=found_teams, in_progress=None,
        )

    # Sort found teams by final Nash weight (index offset = k0)
    found_nash = nash_w[k0:]
    ranked = sorted(
        zip(found_teams, found_nash), key=lambda x: x[1], reverse=True
    )
    logger.info("SP-PSRO complete.")
    for t, w in ranked:
        logger.info("  nash_weight=%.4f  win_rate=%.3f", w, t.win_rate or 0.0)

    # Save one replay per original meta team for the best team
    if ranked:
        best_team = ranked[0][0]
        meta_texts = [Path(p).read_text(encoding="utf-8") for p in meta_team_paths]
        _save_best_team_replays(
            best_team, meta_texts, output_dir, port, n_battles=1, reg=reg,
            battle_agent_path=battle_agent_path, device=device,
        )

    return [t for t, _ in ranked]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _save_best_team_replays(
    team: CandidateTeam,
    meta_texts: list[str],
    output_dir: Path,
    port: int,
    n_battles: int,
    reg: str,
    battle_agent_path: Path | None = None,
    device: str = "cpu",
) -> None:
    """Battle best_team once against each original meta team and save replays."""
    from vgc_bench.team_builder.evaluator import TeamEvaluator
    replay_dir = output_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving replay battles to %s …", replay_dir)
    for i, opp_text in enumerate(meta_texts):
        packed = _text_to_packed(opp_text)
        ev = TeamEvaluator(
            opponent_team=packed,
            n_battles=n_battles,
            port=port,
            reg=reg,
            save_replays=str(replay_dir),
            battle_agent_path=battle_agent_path,
            device=device,
        )
        ev.evaluate(team)
        logger.info("  Replay vs meta team %d saved.", i + 1)


def _text_to_packed(text: str) -> str:
    from poke_env.teambuilder import Teambuilder
    return Teambuilder.join_team(Teambuilder.parse_showdown_team(text))


def _eval_vs_population(
    team: CandidateTeam,
    population_texts: list[str],
    port: int,
    n_battles: int,
    reg: str,
    battle_agent_path: Path | None = None,
    device: str = "cpu",
) -> list[float]:
    """Return win rates of team vs each member of the current population."""
    from vgc_bench.team_builder.evaluator import TeamEvaluator
    win_rates = []
    for opp_text in population_texts:
        packed = _text_to_packed(opp_text)
        ev = TeamEvaluator(
            opponent_team=packed,
            n_battles=n_battles,
            port=port,
            reg=reg,
            battle_agent_path=battle_agent_path,
            device=device,
        )
        scored = ev.evaluate(team)
        win_rates.append(scored.win_rate or 0.0)
    return win_rates


def _save_state(
    teams: list[CandidateTeam],
    payoff: np.ndarray,
    nash_w: list[float],
    output_dir: Path,
    iteration: int,
) -> None:
    payoff_path = output_dir / f"payoff_iter{iteration:03d}.json"
    with payoff_path.open("w") as f:
        json.dump([[round(v, 3) for v in row] for row in payoff.tolist()], f, indent=2)

    teams_path = output_dir / f"teams_iter{iteration:03d}.txt"
    with teams_path.open("w") as f:
        for i, t in enumerate(teams):
            f.write(f"# Team {i+1}  win_rate={t.win_rate:.3f}\n")
            f.write(t.to_showdown_text())
            f.write("\n\n---\n\n")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="SP-PSRO RL team builder")
    parser.add_argument(
        "--meta-teams",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Showdown-format team files for the initial meta population.",
    )
    parser.add_argument("--n-steps", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--n-battles", type=int, default=20)
    parser.add_argument("--reg", type=str, default="i")
    parser.add_argument("--policy-checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/rl_psro")
    parser.add_argument(
        "--quality-threshold", type=float, default=0.5,
        help="Min Nash-weighted win rate for a generated team to be added to the population (default 0.5).",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=500,
        help="Save a team snapshot every N training steps (default 500).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ignore any existing checkpoint and start fresh.",
    )
    parser.add_argument(
        "--battle-agent-path", type=str, default=None,
        help="Path to a trained PPO .zip. When set, the team evaluator uses "
             "BatchPolicyPlayer with this checkpoint instead of "
             "SimpleHeuristicsPlayer for both sides.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for the battle policy (e.g. 'cpu', 'cuda:0'). "
             "Only used when --battle-agent-path is set.",
    )
    args = parser.parse_args()

    space = BuildSpace.from_regulation(args.reg)
    try:
        ranked = run_sp_psro(
            meta_team_paths=args.meta_teams,
            space=space,
            n_iterations=args.iterations,
            n_steps=args.n_steps,
            port=args.port,
            n_battles=args.n_battles,
            reg=args.reg,
            policy_checkpoint=args.policy_checkpoint,
            output_dir=Path(args.output),
            quality_threshold=args.quality_threshold,
            snapshot_every=args.snapshot_every,
            resume=not args.no_resume,
            battle_agent_path=Path(args.battle_agent_path) if args.battle_agent_path else None,
            device=args.device,
        )
    except KeyboardInterrupt:
        logger.warning("SP-PSRO interrupted. Checkpoint saved; re-run with same --output to resume.")
        return

    if ranked:
        print("\n=== Best team ===")
        print(ranked[0].to_showdown_text())


if __name__ == "__main__":
    main()
