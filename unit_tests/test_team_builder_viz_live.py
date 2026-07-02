"""
Unit tests for vgc_bench.team_builder.viz.live — the live training view.

Rendered headlessly via Streamlit's AppTest.from_function, so no real training
process or browser is needed; fixture run dirs simulate a trainer mid-run.
"""

from __future__ import annotations

import json
import time

from streamlit.testing.v1 import AppTest

# Reuse the same minimal snapshot fixture as the loader tests.
_SNAP = """\
# step={step}  win_rate={wr:.3f}
Cinccino @ Life Orb
Ability: Skill Link
Level: 50
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Protect
- Rock Blast
- Bullet Seed
- Triple Axel

Calyrex-Shadow @ Covert Cloak
Ability: As One (Spectrier)
Level: 50
Timid Nature
- Protect
- Astral Barrage
- Expanding Force
- Disable

Urshifu-Rapid-Strike @ Safety Goggles
Ability: Unseen Fist
Level: 50
Jolly Nature
- Protect
- Surging Strikes
- Taunt
- Helping Hand

Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 50
Jolly Nature
- Protect
- Moonblast
- Icy Wind
- Fake Tears

Amoonguss @ Focus Sash
Ability: Regenerator
Level: 50
Bold Nature
- Protect
- Spore
- Rage Powder
- Pollen Puff

Chien-Pao @ Never-Melt Ice
Ability: Sword of Ruin
Level: 50
Jolly Nature
- Protect
- Sacred Sword
- Icy Wind
- Throat Chop
"""


def _snap(step: int, wr: float) -> str:
    return _SNAP.format(step=step, wr=wr)


def _write_live_run(root, with_live_status=True):
    (root / "beta_iter000" / "snapshots").mkdir(parents=True)
    (root / "beta_iter000" / "snapshots" / "step_000010.txt").write_text(
        _snap(10, 0.5), encoding="utf-8"
    )
    (root / "psro_checkpoint.json").write_text(
        json.dumps({"type": "sp_psro", "completed_iterations": 0}), encoding="utf-8"
    )
    (root / "run_meta.json").write_text(
        json.dumps(
            {
                "run_type": "sp_psro",
                "reg": "i",
                "target_iterations": 3,
                "started_at": 0.0,
            }
        ),
        encoding="utf-8",
    )
    if with_live_status:
        (root / "live_status.json").write_text(
            json.dumps(
                {
                    "updated_at": time.time(),
                    "phase": "beta",
                    "unit": "step",
                    "iteration": 0,
                    "step": 42,
                    "total": 1000,
                    "current_win_rate": 0.6,
                    "recent_win_rate": 0.55,
                    "best_win_rate": 0.8,
                    "current_team_showdown": _snap(42, 0.6),
                    "best_team_showdown": _snap(30, 0.8),
                }
            ),
            encoding="utf-8",
        )


def _page(run_dir: str) -> None:
    from vgc_bench.team_builder.viz.live import render_live

    render_live(run_dir)


class TestRenderLive:
    def test_renders_with_fresh_live_status(self, tmp_path):
        _write_live_run(tmp_path)
        at = AppTest.from_function(_page, args=(str(tmp_path),), default_timeout=60)
        at.run()
        assert not at.exception
        assert "Live training" in [s.value for s in at.subheader]
        assert "Currently exploring" in [s.value for s in at.subheader]
        assert "Best so far" in [s.value for s in at.subheader]
        header = at.caption[0].value
        assert "🔴 LIVE" in header
        assert "iteration 1/3" in header
        assert "step 42/1000" in header
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Current win rate"] == "60%"
        assert metrics["Best win rate"] == "80%"

    def test_warns_when_no_live_status_yet(self, tmp_path):
        _write_live_run(tmp_path, with_live_status=False)
        at = AppTest.from_function(_page, args=(str(tmp_path),), default_timeout=60)
        at.run()
        assert not at.exception
        assert any("No live_status.json yet" in w.value for w in at.warning)

    def test_no_crash_on_completely_empty_run(self, tmp_path):
        (tmp_path / "psro_checkpoint.json").write_text(
            json.dumps({"type": "sp_psro", "completed_iterations": 0}), encoding="utf-8"
        )
        at = AppTest.from_function(_page, args=(str(tmp_path),), default_timeout=60)
        at.run()
        assert not at.exception
