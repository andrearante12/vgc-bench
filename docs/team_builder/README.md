# Team Builder — Run Guide

This module finds competitive VGC teams from scratch by optimizing win rate
against a pool of meta teams. Two modes are available:

| Mode | Entry point | When to use |
|---|---|---|
| **Evolutionary PSRO** | `python -m vgc_bench.team_builder` | Fast iteration; no GPU needed |
| **RL SP-PSRO** | `python -m vgc_bench.team_builder.rl.train` | Richer optimization; trains a neural team-builder policy |

See [Architecture](architecture.md) for how the system works.

---

## Prerequisites

### 1 — Showdown server

Every run that plays actual battles needs the local Showdown server running.
Start it in a dedicated terminal and leave it up for the duration of training:

```powershell
cd pokemon-showdown
node pokemon-showdown start --no-security --port 8100
```

Wait until you see `Worker 1 now listening on 0.0.0.0:8100`.

### 2 — BC battle-agent checkpoint (strongly recommended)

The evolutionary optimizer uses a pre-trained BC policy for both sides of every
battle, giving much higher-quality win signals than the built-in heuristic player.

The checkpoint lives at `results/saves_bc/seed1/100.zip`. Download it once:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="cameronangliss/vgc-bench-models",
    filename="results/saves_bc/seed1/100.zip",
    local_dir=".",
)
```

If `--battle-agent-path` is omitted the run falls back to `SimpleHeuristicsPlayer`
(faster but weaker signal).

---

## Regulation: Reg I

All current runs target **Reg I** (`gen9vgc2025regi`):

- Standard Gen 9 VGC doubles, Level 50
- **Limit Two Restricted** — at most 2 restricted legendaries per team
- EVs: 0–252 per stat, total ≤ 510
- Tera type is a first-class optimized field, sampled from the tournament corpus
- 739 tournament team files in `teams/reg_i/`, ~200 curated in `teams/reg_i/featured/`

The four meta archetypes used as seeds cover the dominant Reg I strategies:

| File | Restricted core | Playstyle | Corpus frequency |
|---|---|---|---|
| `featured/I1146.txt` | Miraidon + Zamazenta-Crowned | Bulky electric offense | ~136 entries |
| `featured/I1062.txt` | Calyrex-Shadow + Zamazenta-Crowned | Hyper offense | ~106 entries |
| `featured/I1054.txt` | Calyrex-Ice + Groudon | Sun balance | ~83 entries |
| `featured/I1063.txt` | Koraidon + Lunala | Offense + speed control | ~45 entries |

---

## Evolutionary PSRO

### `best_response` — find best team vs fixed opponents

Runs a (μ + λ) evolutionary search to find the team with the highest win rate
against one or more fixed meta teams.

```powershell
# Single meta team
python -m vgc_bench.team_builder best_response `
  --meta-team teams/reg_i/featured/I1146.txt `
  --battle-agent-path results/saves_bc/seed1/100.zip `
  --rounds 10 --population 20 --candidates 40 --n-battles 20 `
  --port 8100 --output results/br_miraidon --verbose

# Uniform mixture of multiple meta teams
python -m vgc_bench.team_builder best_response `
  --meta-team teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt `
             teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt `
  --battle-agent-path results/saves_bc/seed1/100.zip `
  --rounds 10 --population 20 --candidates 40 --n-battles 20 `
  --port 8100 --output results/br_mixture --verbose
```

### `psro` — adversarial population loop

Iteratively adds best-response teams to a growing population. Each iteration
computes a Nash equilibrium over all current teams, then finds the team that best
counters the Nash mixture. Converges toward teams that cannot be easily exploited.

```powershell
# Recommended production run
python -m vgc_bench.team_builder psro `
  --meta-team teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt `
             teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt `
  --battle-agent-path results/saves_bc/seed1/100.zip `
  --psro-iterations 8 --rounds 10 --population 20 --candidates 40 `
  --n-battles 20 --port 8100 --output results/psro_4meta --verbose
```

Re-run the exact same command to resume after interruption — the checkpoint at
`<output>/psro_checkpoint.json` is detected automatically.

### Flag reference

| Flag | Default | Meaning |
|---|---|---|
| `--meta-team` | required | One or more `.txt` team files (space-separated) |
| `--battle-agent-path` | None | BC checkpoint `.zip`; omit to use SimpleHeuristicsPlayer |
| `--psro-iterations` | 10 | PSRO: best-response cycles |
| `--rounds` | 10 | EA: search rounds per PSRO iteration |
| `--population` | 20 | EA: elite teams retained per round (μ) |
| `--candidates` | 40 | EA: new teams evaluated per round (λ) |
| `--n-battles` | 20 | Battles per team evaluation |
| `--port` | 8100 | Showdown server port |
| `--output` | `results/team_builder` | Directory for checkpoints and snapshots |
| `--seed` | None | Random seed for reproducibility |
| `--swap-probability` | 0.3 | Probability of `swap_member` vs other mutation operators |
| `--mutate-probability` | 0.3 | Probability of `mutate_build` (remainder = `combine_teams`) |
| `--diversity-threshold` | 0.0 | Min inter-team dissimilarity for elite selection (0 = off) |
| `--verbose` | False | Enable INFO logging |

### Output files

```
results/psro_4meta/
  psro_checkpoint.json            resume state (iterations, payoff, found teams)
  payoff_iter000.json             N×N win-rate matrix after each iteration
  teams_iter000.txt               Showdown text of all found teams per iteration
  search_history_iter000.json     per-round win rates for each search phase
  search_iter000/
    round_000.txt                 best team text at each search round
    rounds.jsonl                  compact round-by-round statistics log
```

---

## RL SP-PSRO

Trains a `TeamBuilderNetwork` policy via REINFORCE. The network autoregressively
generates a full team (species then items/moves/natures per slot) and learns from
binary battle outcomes (+1 win, 0 loss).

Each SP-PSRO iteration trains two policies:
- **β** — best response to the current Nash mixture of the population
- **ν** — self-play counter trained against β's generated teams

Both are added to the population if their win rate clears `--quality-threshold`.

```powershell
python -m vgc_bench.team_builder.rl.train `
  --meta-teams teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt `
               teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt `
  --n-steps 10000 --iterations 5 --n-battles 20 `
  --port 8100 --output results/rl_psro --verbose
```

### RL-specific flags

| Flag | Default | Meaning |
|---|---|---|
| `--meta-teams` | required | Initial population team files |
| `--n-steps` | 10000 | REINFORCE gradient steps per training phase |
| `--iterations` | 5 | SP-PSRO outer iterations |
| `--policy-checkpoint` | None | PPO `.zip` to initialize move/item/ability embeddings |
| `--quality-threshold` | 0.5 | Min Nash-weighted win rate to add team to population |
| `--snapshot-every` | 500 | Save network checkpoint every N steps |
| `--no-resume` | False | Ignore existing checkpoint; start fresh |

---

## Smoke test (no server needed)

```powershell
pytest unit_tests/test_pokemon_build.py `
       unit_tests/test_build_space.py `
       unit_tests/test_team_builder_team.py -v
```

All 90 tests pass in ~20 s.

---

## Prerequisites

Battles require a local Showdown server:

```powershell
cd pokemon-showdown
node pokemon-showdown start 8100 --no-security
```

---

## Running

```powershell
# Smoke test — no Showdown server needed
pytest unit_tests/test_team_builder_network.py -v

# Full SP-PSRO run (5 iterations, ~4 meta teams → 14-team population)
python -m vgc_bench.team_builder.rl.train `
  --meta-teams teams/reg_ma/PC1.txt teams/reg_ma/PC2.txt `
               teams/reg_ma/PC3.txt teams/reg_ma/PC4.txt `
  --n-steps 10000 --iterations 5 --port 8100 --output results/rl_psro
```

Output is written to `results/rl_psro/`: team files and payoff matrices per iteration. The best team (highest Nash weight) is printed at the end.
