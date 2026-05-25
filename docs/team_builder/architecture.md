# Team Builder — Architecture

---

## Overview

The team builder solves a two-level optimization problem:

1. **Outer loop (PSRO)** — maintains a population of teams and a Nash
   equilibrium over them. Each iteration adds a new team that best counters
   the current Nash mixture, expanding the population toward a strategy-proof
   set.

2. **Inner loop (best-response oracle)** — finds the single best team against
   a fixed opponent or Nash mixture. Two implementations exist:
   - **Evolutionary** (`optimizer.py`): (μ + λ) search using random
     mutation/crossover operators, no gradient computation.
   - **RL** (`rl/train.py`): REINFORCE policy gradient over a neural
     autoregressive team generator.

---

## Module map

```
vgc_bench/team_builder/
│
├── pokemon_build.py      PokemonBuild dataclass — immutable description of one build
├── build_space.py        BuildSpace — corpus-driven valid search space per regulation
├── team.py               CandidateTeam + operators (random_team, swap_member,
│                           combine_teams, mutate_build)
├── evaluator.py          TeamEvaluator — runs Showdown battles, returns win rate
├── optimizer.py          SearchConfig + best_response / best_response_vs_mixture
├── psro.py               PSROConfig + run_psro (evolutionary outer loop)
├── __main__.py           CLI entry point (best_response / psro subcommands)
│
└── rl/
    ├── team_builder_network.py   TeamBuilderNetwork — autoregressive neural team generator
    ├── team_builder_env.py       TeamBuilderBattleEnv — thin evaluator wrapper for training
    ├── bc_init.py                apply_bc_biases — warm-start from PPO checkpoint
    └── train.py                  run_sp_psro (RL outer loop) + train_policy (REINFORCE)
```

---

## Data model

### `PokemonBuild` (`pokemon_build.py`)

An **immutable, hashable** dataclass describing one competitive build:

```
species  str            e.g. "Miraidon"
item     str            e.g. "Choice Specs"
ability  str            e.g. "Hadron Engine"
nature   str            e.g. "Timid"
evs      (int×6)        HP/Atk/Def/SpA/SpD/Spe  — each 0–252, sum ≤ 510
ivs      (int×6)        defaults to all-31
moves    (str×4)        exactly 4 distinct moves
tera_type str | None    e.g. "Electric"
```

Frozen → hashable → usable as dict keys and cache keys inside the search loop.

### `CandidateTeam` (`team.py`)

Six `PokemonBuild` instances bundled with an optional `win_rate`. Enforces:
- Species Clause (no duplicate species)
- Item Clause (no duplicate held items)
- Restricted limit (≤ 2 restricted legendaries for Reg I)

### `BuildSpace` (`build_space.py`)

The valid search space for a regulation, built from **tournament team files**.
For each species it stores:
- `moves`, `items`, `abilities`, `tera_types` — sorted lists observed in corpus
- `move_weights`, `item_weights` — corpus-frequency sampling weights
- `nat_ev_corpus` — list of (nature, EVs) pairs for BC-style initialization
- Space-level: `species_weights`, `restricted_species`, `restricted_limit`

`random_build()` samples nature+EVs via a 3-way mixture:
- 60% BC corpus (pick an observed (nature, EVs) pair for this species)
- 25% concentrated (max 1–2 stats, common competitive pattern)
- 15% uniform random

---

## Evolutionary PSRO data flow

```
BuildSpace (corpus)
     │ random_team / swap_member / combine_teams / mutate_build
     ▼
CandidateTeam  ──────────────────────────────────────────────┐
     │                                                        │
     │ to_packed_team()                                       │
     ▼                                                        │
TeamEvaluator                                                 │
  ├─ candidate: SimpleHeuristicsPlayer or BatchPolicyPlayer   │
  └─ opponent:  SimpleHeuristicsPlayer or BatchPolicyPlayer   │
     │                                                        │
     │ n_battles Showdown battles                             │
     ▼                                                        │
  win_rate  ──► CandidateTeam.with_win_rate()  ──────────────┘
                        │
                        ▼
              _run_search() [optimizer.py]
              ┌──────────────────────────┐
              │  elites (μ best teams)   │
              │  + candidates (λ new)    │
              │  → select top-μ          │
              │  → update space weights  │
              └──────────────────────────┘
                        │  repeat R rounds
                        ▼
                  best CandidateTeam
                        │
                        ▼ (PSRO outer loop)
              Extend payoff matrix
              Recompute Nash weights (nashpy)
              Next iteration → new best-response search
```

### Mutation operators

| Operator | Description |
|---|---|
| `random_team` | Sample 6 species by weight; build each fresh from corpus |
| `swap_member` | Replace N slots with fresh random builds |
| `combine_teams` | Slot-by-slot crossover of two parent teams |
| `mutate_build` | Change exactly one field of one build (move / nat+EVs / EVs transfer / item / tera type) |

Mutation weights: move 35%, item 20%, nat_ev 20%, evs_transfer 15%, tera 10%.

---

## RL SP-PSRO data flow

```
Population (K teams + Nash weights w₁…wₖ)
           │
           ▼
  Per-team Transformer encoder
  (6 Pokémon tokens → team_context [256])
           │
  Nash-weighted mean: Σ wₖ · team_contextₖ
           │
  population_context [256]
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
Phase 1         Phase 2
Species         Per-slot builds
(autoregressive, (item, 4 moves, nature)
 Species Clause)  masked to species move pool
    │             │
    └──────┬──────┘
           │
    CandidateTeam (stochastically sampled)
           │
    battle vs opponent drawn from Nash mixture
           │
    reward ∈ {0, 1}
           │
    REINFORCE + value-head baseline
           │
    gradient update (lr=1e-4, entropy coef=0.05)

Per SP-PSRO iteration:
  1. Train β  against current Nash mixture  → β_team  (if win_rate ≥ threshold)
  2. Train ν  against β_team alone          → ν_team  (if win_rate ≥ threshold)
  3. Evaluate both new teams vs all population members
  4. Extend payoff matrix → recompute Nash
```

### `TeamBuilderNetwork` per-Pokémon token (291-dim)

```
species_embed(32) | move_embed×4(128) | item_embed(32) | ability_embed(32)
| base_types(36) | base_stats(6) | evs(6) | nature_onehot(25) = 291
```

Move/item/ability embeddings can be warm-started from a PPO battle-agent
checkpoint via `--policy-checkpoint` (frozen by default).

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Immutable `PokemonBuild` | Hashable → cheap caching of evaluated teams in the search loop; `replace()` enables per-field mutation without copies |
| Corpus-driven `BuildSpace` | Only moves/items used by top players are in scope; keeps search space focused and all generated teams format-legal |
| 3-way EV mixture (BC/concentrated/uniform) | BC corpus provides competitive starting points; concentrated covers standard max-2-stats patterns; uniform ensures exploration |
| Restricted limit in `BuildSpace` | Enforced at team-generation time (random_team / swap_member / combine_teams) so invalid teams never reach the evaluator |
| Persistent `TeamEvaluator` player pair | Creating poke-env Player objects is expensive and causes asyncio issues; one pair is reused across all evaluate() calls per run |
| `--battle-agent-path` (BC checkpoint) | SimpleHeuristicsPlayer produces noisy, exploitable play; using the BC policy on both sides gives a realistic skill level and stable win-rate estimates |
| PSRO seeded with multiple meta teams | Starting from 4 archetypes gives the Nash solver meaningful weights immediately, avoiding the early iterations being dominated by the single seed |
