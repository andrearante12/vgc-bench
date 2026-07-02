# Team Builder

> **Full documentation:** [`docs/team_builder/`](../../docs/team_builder/README.md)

| Page | What it covers |
|---|---|
| [Run guide](../../docs/team_builder/README.md) | Prerequisites, battle agents, run commands, output files |
| [Architecture](../../docs/team_builder/architecture.md) | Module map, data flow, key classes |

---

## Quick start

Activate the venv (`source .venv/bin/activate`) and start the Showdown server
(`cd pokemon-showdown && node pokemon-showdown start --no-security --port 8100`)
first. See the [run guide](../../docs/team_builder/README.md) for downloading the
battle-agent checkpoint.

```bash
# Unit tests — no Showdown server needed
pytest unit_tests/test_pokemon_build.py unit_tests/test_build_space.py \
       unit_tests/test_team_builder_team.py unit_tests/test_team_builder_network.py -v

# RL SP-PSRO team builder (primary), BCSP battle agent on GPU
python -m vgc_bench.team_builder.rl.train \
  --meta-teams teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
               teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt \
  --battle-agent-path results/saves_bc_sp/reg_all/seed1/98304000.zip --device cuda:0 \
  --n-steps 10000 --iterations 5 --port 8100 --output results/rl_psro

# Evolutionary PSRO (CPU, no neural policy)
python -m vgc_bench.team_builder psro \
  --meta-team teams/reg_i/featured/I1146.txt teams/reg_i/featured/I1062.txt \
              teams/reg_i/featured/I1054.txt teams/reg_i/featured/I1063.txt \
  --battle-agent-path results/saves_bc_sp/reg_all/seed1/98304000.zip --device cuda:0 \
  --psro-iterations 8 --rounds 10 --population 20 --candidates 40 \
  --n-battles 20 --port 8100 --output results/psro_4meta --verbose
```

All runs auto-resume from `<output>/psro_checkpoint.json` if interrupted
(`--no-resume` to start fresh).

## Visualize a run

```bash
pip install -e '.[viz]'
streamlit run vgc_bench/team_builder/viz/app.py -- results/<run>
```

**Live** mode auto-refreshes while a run is training; **Playback** replays any run like
a video (play/pause/speed/scrubber) and can sample a fresh team from a saved
`policy.pt` checkpoint. See the
[run guide](../../docs/team_builder/README.md#visualizing-a-run-interactive-dashboard)
for details.
