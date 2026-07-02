# VGC-Bench — Team Builder

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **A fork of [VGC-Bench](https://github.com/cameronangliss/vgc-bench)** (Angliss et al., [arXiv:2506.10326](https://arxiv.org/abs/2506.10326)). The original benchmark trains agents to *play* a fixed Pokémon VGC team. This fork adds a **Team Builder** that generates the *team itself*. All credit for the underlying benchmark goes to the original authors — see [Credits](#credits).

## What I added

A **Team Builder** that discovers competitive **Reg I** VGC teams from scratch by optimizing win rate against a pool of meta archetypes. Team construction is framed as a two-level game-theoretic search (PSRO) with two interchangeable best-response oracles:

- **Evolutionary PSRO** — a lightweight approach (no GPU needed) that "breeds" better teams over many rounds: it keeps the strongest teams found so far, then creates new candidates by swapping members, tweaking individual Pokémon, and mixing two good teams together — always staying within real, tournament-legal options.
- **RL SP-PSRO** — a neural network that learns to build teams one Pokémon at a time, getting better through trial and error as it sees which teams actually win. It focuses its effort on beating whatever opponents are currently most threatening, and gets a head start from VGC-Bench's pre-trained battle agent.

Both modes enforce real VGC rules (Species / Item Clause, Limit Two Restricted) and pilot candidate teams with the upstream BC checkpoint for a realistic win signal.

📄 Full details: [run guide](docs/team_builder/README.md) · [architecture](docs/team_builder/architecture.md)

## Demo

<video src="vgc_demo.webm" autoplay loop muted playsinline controls width="100%">
  Your browser does not support embedded video —
  <a href="vgc_demo.webm">download the demo (vgc_demo.webm)</a>.
</video>

> The Team Builder generating and evaluating a Reg I team end-to-end. If the video does not autoplay in your viewer, [open <code>vgc_demo.webm</code> directly](vgc_demo.webm).

## Quick start

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install ".[dev]"

# 2. Start the Showdown server (separate terminal, leave running)
cd pokemon-showdown && node pokemon-showdown start --no-security --port 8100
```

```bash
# RL SP-PSRO — trains the neural team-builder policy
python -m vgc_bench.team_builder.rl.train \
  --meta-teams teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
               teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt \
  --n-steps 10000 --iterations 5 --port 8100 --output results/rl_psro

# Evolutionary PSRO — fast, CPU-only, no neural policy
python -m vgc_bench.team_builder psro \
  --meta-team teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
              teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt \
  --psro-iterations 8 --rounds 10 --n-battles 20 --port 8100 --output results/psro_4meta
```

Add `--battle-agent-path <checkpoint.zip> --device cuda:0` to pilot candidates with a trained
battle agent instead of the default heuristic. See the [run guide](docs/team_builder/README.md)
for checkpoints, all flags, and how to read the results.

```bash
# Tests (no Showdown server needed)
pytest unit_tests/test_pokemon_build.py unit_tests/test_build_space.py \
       unit_tests/test_team_builder_team.py unit_tests/test_team_builder_network.py -v
```

## Credits

Forked from and built on top of **VGC-Bench** by Cameron Angliss, Jiaxun Cui, Jiaheng Hu,
Arrasy Rahman, and Peter Stone (AAMAS 2025). Everything except the Team Builder is their work.

```bibtex
@inproceedings{anglissvgc,
  title={VGC-Bench: Towards Mastering Diverse Team Strategies in Competitive Pok{\'e}mon},
  author={Angliss, Cameron L and Cui, Jiaxun and Hu, Jiaheng and Rahman, Arrasy and Stone, Peter},
  booktitle={The 25th International Conference on Autonomous Agents and Multi-Agent Systems}
}
```
