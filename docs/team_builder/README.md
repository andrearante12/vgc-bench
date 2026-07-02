# Team Builder — Run Guide

This module finds competitive VGC teams from scratch by optimizing win rate
against a pool of meta teams. Two modes are available:

| Mode | Entry point | When to use |
|---|---|---|
| **RL SP-PSRO** | `python -m vgc_bench.team_builder.rl.train` | Primary. Trains a neural team-builder policy; benefits from a GPU |
| **Evolutionary PSRO** | `python -m vgc_bench.team_builder` | Fast, CPU-only iteration; no neural policy |

See [Architecture](architecture.md) for how the system works.

---

## Prerequisites

### 1 — Python environment

Install once into a virtualenv and activate it (so `python` is the project env,
not your base conda):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ".[dev]"
```

All commands below assume the venv is active. (Or prefix each command with
`.venv/bin/python` instead of activating.)

### 2 — Showdown server

Every run that plays actual battles needs the local Showdown server running.
Start it in a dedicated terminal and leave it up for the duration of training:

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security --port 8100
```

Wait until you see `Worker 1 now listening on 0.0.0.0:8100`.
(`EADDRINUSE` just means a server is already running on that port.)

### 3 — Battle-agent checkpoint (strongly recommended)

The win rate of a candidate team is measured by having a battle agent pilot it.
By default this is the rule-based `SimpleHeuristicsPlayer` (fast but weak). For
high-quality win signals, pass a trained model via `--battle-agent-path`. Two are
published in the `cameronangliss/vgc-bench-models` Hugging Face repo:

| Checkpoint | Local path | Notes |
|---|---|---|
| **BCSP** (strongest) | `results/saves_bc_sp/reg_all/seed1/98304000.zip` | BC-initialized self-play, all regs (~98M steps) |
| **BC** | `results/saves_bc/seed1/100.zip` | Behavior cloning only; lighter |

Download once:

```bash
python -c "from huggingface_hub import hf_hub_download as d; \
d(repo_id='cameronangliss/vgc-bench-models', filename='results/saves_bc_sp/reg_all/seed1/98304000.zip', local_dir='.'); \
d(repo_id='cameronangliss/vgc-bench-models', filename='results/saves_bc/seed1/100.zip', local_dir='.')"
```

Add `--device cuda:0` to run the battle agent on the GPU (recommended when using a
trained checkpoint; it is the per-move inference, not the team-builder net, that
benefits most from the GPU).

---

## Regulation: Reg I

All current runs target **Reg I** (`gen9vgc2026regi`):

- Standard Gen 9 VGC doubles, Level 50
- **Limit Two Restricted** — at most 2 restricted legendaries per team
- EVs: 0–252 per stat, total ≤ 510
- Tera type is a first-class optimized field, sampled from the tournament corpus
- Tournament team files in `teams/reg_i/`, curated subset in `teams/reg_i/featured/`

The four meta archetypes used as seeds cover the dominant Reg I strategies:

| File | Restricted core | Playstyle |
|---|---|---|
| `featured/I1146.txt` | Miraidon + Zamazenta-Crowned | Bulky electric offense |
| `featured/I1062.txt` | Calyrex-Shadow + Zamazenta-Crowned | Hyper offense |
| `featured/I1054.txt` | Calyrex-Ice + Groudon | Sun balance |
| `featured/I1063.txt` | Koraidon + Lunala | Offense + speed control |

---

## RL SP-PSRO (primary)

Trains a `TeamBuilderNetwork` policy via REINFORCE. The network autoregressively
generates a full team (species, then item/moves/nature per slot) and learns from
binary battle outcomes (+1 win, 0 loss).

Each SP-PSRO iteration trains two policies, adding each to the population if its
Nash-weighted win rate clears `--quality-threshold`:
- **β** — best response to the current Nash mixture of the population
- **ν** — self-play counter trained against β's generated teams

```bash
# Full run, BCSP battle agent on GPU
python -m vgc_bench.team_builder.rl.train \
  --meta-teams teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
               teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt \
  --battle-agent-path results/saves_bc_sp/reg_all/seed1/98304000.zip --device cuda:0 \
  --n-steps 10000 --iterations 5 --n-battles 20 \
  --port 8100 --output results/rl_psro
```

```bash
# Quick smoke run (heuristic agent, small counts) to verify the loop end-to-end
python -m vgc_bench.team_builder.rl.train \
  --meta-teams teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
  --n-steps 30 --iterations 1 --n-battles 4 --snapshot-every 10 \
  --port 8100 --output results/smoke --no-resume
```

Re-run the exact same command to resume after interruption — the checkpoint at
`<output>/psro_checkpoint.json` is detected automatically (use `--no-resume` to
start fresh).

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--meta-teams` | required | Initial population team files (space-separated) |
| `--battle-agent-path` | None | Battle-agent `.zip`; omit to use SimpleHeuristicsPlayer |
| `--device` | `cpu` | Torch device for the battle agent, e.g. `cuda:0` |
| `--n-steps` | 10000 | REINFORCE gradient steps per training phase |
| `--iterations` | 5 | SP-PSRO outer iterations |
| `--n-battles` | 20 | Battles per team evaluation |
| `--quality-threshold` | 0.5 | Min Nash-weighted win rate to add a team to the population |
| `--policy-checkpoint` | None | PPO `.zip` to initialize move/item/ability embeddings |
| `--log-every` | 100 | Write a `metrics.jsonl`/`live_status.json` update every N steps |
| `--snapshot-every` | 500 | Save a team/network snapshot every N steps |
| `--no-resume` | False | Ignore existing checkpoint; start fresh |
| `--port` | 8100 | Showdown server port |
| `--output` | `results/rl_psro` | Output directory |

---

## Evolutionary PSRO (CPU, no neural policy)

A `(μ + λ)` evolutionary search over teams. `best_response` finds the best team
against a fixed opponent set; `psro` grows a population, computing a Nash
equilibrium each iteration and best-responding to it.

```bash
# best_response — best team vs a fixed meta mixture
python -m vgc_bench.team_builder best_response \
  --meta-team teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
              teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt \
  --battle-agent-path results/saves_bc_sp/reg_all/seed1/98304000.zip --device cuda:0 \
  --rounds 10 --population 20 --candidates 40 --n-battles 20 \
  --port 8100 --output results/br_mixture --verbose

# psro — adversarial population loop
python -m vgc_bench.team_builder psro \
  --meta-team teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
              teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt \
  --battle-agent-path results/saves_bc_sp/reg_all/seed1/98304000.zip --device cuda:0 \
  --psro-iterations 8 --rounds 10 --population 20 --candidates 40 \
  --n-battles 20 --port 8100 --output results/psro_4meta --verbose
```

Key flags: `--rounds` (search rounds per iteration), `--population` (elites kept, μ),
`--candidates` (new teams per round, λ), `--psro-iterations` (PSRO cycles), plus the
shared `--meta-team / --battle-agent-path / --device / --n-battles / --port /
--output / --seed / --verbose`. See `--help` for mutation-rate flags.

---

## Reading the results (evaluation)

Evaluation is produced by the runs themselves — there is no separate eval command.

- **Live win rates** — each phase logs e.g. `β best team win_rate=0.750`; the final
  population and its Nash weights are printed at the end. Run with `--verbose`
  (evolutionary) for INFO-level progress.
- **`payoff_iter*.json`** — N×N win-rate matrix over the population after each
  iteration (the head-to-head evaluation grid).
- **`teams_iter*.txt`** — Showdown text of every team found that iteration; the
  best team is the one with the highest final Nash weight.
- **`psro_checkpoint.json`** — resume state (iterations, payoff, found teams).
- **`replays/`** — saved `.html` battle replays of the best team vs each meta team.
- RL runs also write per-phase `policy.pt`, `metrics.jsonl`, and `snapshots/`.
- **`run_meta.json`** — written once at run start (run type, regulation, target
  iterations, cadence); lets the [dashboard](#visualizing-a-run-interactive-dashboard)
  tell a finished run from an incomplete one.
- **`live_status.json`** — overwritten every step (RL) / round (evolutionary) with the
  run's current point-in-time state (phase, step, win rates, the team currently being
  generated). The signal the dashboard's Live mode follows.

A team's win rate reflects how well the chosen battle agent pilots it against the
meta — so the agent is the yardstick. Note that low `--n-battles` produces noisy
(often `1.000`) win rates; raise it for a stable signal.

---

## Visualizing a run (interactive dashboard)

An interactive Streamlit dashboard reads a run directory and shows the **evolution of
the best team** over the run (with Pokémon sprites and full sets, changed
species/moves highlighted) alongside the **evaluation stats** — win-rate curve, the
payoff/matchup heatmap, Nash weights, and species churn. It handles both run types
(RL SP-PSRO and evolutionary) automatically.

```bash
pip install -e '.[viz]'          # one-time: installs streamlit
streamlit run vgc_bench/team_builder/viz/app.py -- results/<run>
```

Pass a run directory after `--` to seed the sidebar (or pick any run from the browser,
which lists every run under `results/` with a status chip — 🔴 live / ✅ done / ⏸
interrupted / ⚪ idle). Two modes:

- **Live** — for a run that's currently training. Auto-refreshes every few seconds
  (a Streamlit fragment, not a full page reload) and follows the team currently being
  generated via `live_status.json`, which the trainer overwrites every step — far more
  granular than the snapshot cadence. Selected automatically for runs with recent
  activity.
- **Playback** — replay any run (finished or not) like a video: play/pause, 1×/2×/4×
  speed, step ± buttons, or the manual scrubber. Also offers **"Generate a fresh team
  from a saved checkpoint"** — loads a saved `policy.pt` (a pure CPU forward pass, no
  Showdown server) and samples a new team (greedy or stochastic), conditioned on the
  opponent population it was trained against.

Sprites are pulled from the Pokémon Showdown CDN and fall back to the species name
when offline. Notes: the win-rate curve is derived from the `snapshots/*.txt` headers
when `metrics.jsonl` is empty (short runs where `log_every` exceeds the step count);
the payoff heatmap uses a diverging colormap centered on 0.5 (an even matchup); the RL
trainer's `--log-every` flag (default 100) controls how often `metrics.jsonl` and
`live_status.json` update — lower it for a more responsive Live view on short/demo
runs.

---

## Smoke test (no server needed)

```bash
pytest unit_tests/test_pokemon_build.py \
       unit_tests/test_build_space.py \
       unit_tests/test_team_builder_team.py \
       unit_tests/test_team_builder_network.py \
       unit_tests/test_team_builder_run_log.py \
       unit_tests/test_team_builder_viz.py \
       unit_tests/test_team_builder_viz_model.py \
       unit_tests/test_team_builder_viz_live.py \
       unit_tests/test_team_builder_viz_playback.py \
       unit_tests/test_team_builder_viz_app.py -v
```
