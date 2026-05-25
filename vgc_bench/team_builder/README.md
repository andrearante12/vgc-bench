# Team Builder

> **Full documentation:** [`docs/team_builder/`](../../docs/team_builder/README.md)

| Page | What it covers |
|---|---|
| [Run guide](../../docs/team_builder/README.md) | Prerequisites, setup, all run commands, output files |
| [Architecture](../../docs/team_builder/architecture.md) | Module map, data flow, key classes |

---

## Quick start

```powershell
# Unit tests — no Showdown server needed
pytest unit_tests/test_pokemon_build.py unit_tests/test_build_space.py unit_tests/test_team_builder_team.py -v

# Evolutionary best-response (single meta team, smoke test)
python -m vgc_bench.team_builder best_response `
  --meta-team teams/reg_i/featured/I1146.txt `
  --battle-agent-path results/saves_bc/seed1/100.zip `
  --rounds 3 --candidates 10 --n-battles 5 --port 8100 --verbose

# PSRO seeded with 4 meta teams (production run)
python -m vgc_bench.team_builder psro `
  --meta-team teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt `
             teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt `
  --battle-agent-path results/saves_bc/seed1/100.zip `
  --psro-iterations 8 --rounds 10 --population 20 --candidates 40 `
  --n-battles 20 --port 8100 --output results/psro_4meta --verbose

# SP-PSRO (RL team builder — needs policy checkpoint or uses random init)
python -m vgc_bench.team_builder.rl.train `
  --meta-teams teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt `
               teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt `
  --n-steps 10000 --iterations 5 --port 8100 --output results/rl_psro
```

All runs auto-resume from `<output>/psro_checkpoint.json` if interrupted.
