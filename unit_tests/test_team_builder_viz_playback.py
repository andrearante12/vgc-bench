"""
Unit tests for vgc_bench.team_builder.viz.playback:

* `advance()` — the pure playhead-stepping function, tested directly with no
  Streamlit dependency.
* `render_playback` — rendered headlessly via Streamlit's AppTest, exercising
  transport buttons, the manual slider, and the generate-from-checkpoint panel
  against a real (serverless) RL run.
"""

from __future__ import annotations

import json
import random

import pytest
from streamlit.testing.v1 import AppTest

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.rl.train import run_sp_psro
from vgc_bench.team_builder.team import random_team
from vgc_bench.team_builder.viz.playback import advance


# ------------------------------------------------------------------ advance ---
class TestAdvance:
    def test_steps_forward_by_speed(self):
        assert advance(playhead=2, n_frames=10, speed=1) == 3
        assert advance(playhead=2, n_frames=10, speed=4) == 6

    def test_wraps_around_when_looping(self):
        assert advance(playhead=8, n_frames=10, speed=4, loop=True) == 2  # 12 % 10

    def test_clamps_at_end_when_not_looping(self):
        assert advance(playhead=8, n_frames=10, speed=4, loop=False) == 9

    def test_exact_landing_on_last_frame_no_wrap(self):
        assert advance(playhead=8, n_frames=10, speed=1, loop=True) == 9

    def test_wraps_exactly_to_zero_from_last_frame(self):
        assert advance(playhead=9, n_frames=10, speed=1, loop=True) == 0

    def test_zero_speed_holds_position(self):
        assert advance(playhead=5, n_frames=10, speed=0) == 5

    def test_single_frame_stays_at_zero(self):
        assert advance(playhead=0, n_frames=1, speed=1, loop=True) == 0
        assert advance(playhead=0, n_frames=1, speed=1, loop=False) == 0

    def test_empty_timeline_returns_zero(self):
        assert advance(playhead=0, n_frames=0, speed=1) == 0


# ------------------------------------------------------------- render (RL) ---
def _page(run_dir: str) -> None:
    from vgc_bench.team_builder.viz.playback import render_playback

    render_playback(run_dir)


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("ma")


@pytest.fixture(scope="module")
def rl_run_dir(space, tmp_path_factory):
    """A real, small RL run produced serverlessly via eval_fn — has policy.pt files."""
    out = tmp_path_factory.mktemp("playback_run")
    meta_path = out / "meta.txt"
    meta_path.write_text(
        random_team(space, rng=random.Random(0)).to_showdown_text(), encoding="utf-8"
    )
    run_sp_psro(
        meta_team_paths=[str(meta_path)],
        space=space,
        reg="ma",
        n_iterations=1,
        n_steps=6,
        log_every=2,
        snapshot_every=2,
        quality_threshold=0.0,
        output_dir=out / "run",
        eval_fn=lambda team, opp_text: random.random(),
        resume=False,
    )
    return out / "run"


class TestRenderPlayback:
    def test_renders_with_transport_and_generate_panel(self, rl_run_dir):
        at = AppTest.from_function(_page, args=(str(rl_run_dir),), default_timeout=90)
        at.run()
        assert not at.exception
        subheaders = [s.value for s in at.subheader]
        assert "Playback" in subheaders
        assert "Generate a fresh team from a saved checkpoint" in subheaders
        labels = [b.label for b in at.button]
        assert "▶ Play" in labels
        assert "⏮ Step back" in labels
        assert "⏭ Step fwd" in labels

    def test_play_pause_toggles_state(self, rl_run_dir):
        at = AppTest.from_function(_page, args=(str(rl_run_dir),), default_timeout=90)
        at.run()
        play_button = next(b for b in at.button if b.label == "▶ Play")
        play_button.click().run()
        assert not at.exception
        assert at.session_state["playback_playing"] is True

    def test_step_buttons_move_playhead_without_looping(self, rl_run_dir):
        at = AppTest.from_function(_page, args=(str(rl_run_dir),), default_timeout=90)
        at.run()
        # Start paused (default) so the autoplay tick doesn't also move the head.
        assert at.session_state["playback_playing"] is False
        start = at.session_state["playback_head"]
        step_fwd = next(b for b in at.button if "Step fwd" in b.label)
        step_fwd.click().run()
        assert not at.exception
        # Already at the last frame (reset default) — step-fwd clamps, not loops.
        assert at.session_state["playback_head"] == start

        step_back = next(b for b in at.button if "Step back" in b.label)
        step_back.click().run()
        assert at.session_state["playback_head"] == max(0, start - 1)

    def test_generate_button_samples_a_valid_team(self, rl_run_dir):
        at = AppTest.from_function(_page, args=(str(rl_run_dir),), default_timeout=90)
        at.run()
        generate_button = next(b for b in at.button if "Generate" in b.label)
        generate_button.click().run()
        assert not at.exception
        assert "_generated_team" in at.session_state
        team = at.session_state["_generated_team"]
        assert len(team.members) == 6


# ------------------------------------------------------- render (no meta) ---
def _write_bare_rl_run(root):
    """A run with snapshots but no run_meta.json / policy.pt (older-style run)."""
    snaps = root / "beta_iter000" / "snapshots"
    snaps.mkdir(parents=True)
    snap_text = (
        "# step=10  win_rate=0.500\n"
        "Cinccino @ Life Orb\nAbility: Skill Link\nLevel: 50\nJolly Nature\n"
        "- Protect\n- Rock Blast\n- Bullet Seed\n- Triple Axel"
    )
    (snaps / "step_000010.txt").write_text(snap_text, encoding="utf-8")
    (root / "psro_checkpoint.json").write_text(
        json.dumps({"type": "sp_psro", "completed_iterations": 0}), encoding="utf-8"
    )


class TestGeneratePanelGuards:
    def test_no_checkpoints_shows_message_not_crash(self, tmp_path):
        _write_bare_rl_run(tmp_path)
        at = AppTest.from_function(_page, args=(str(tmp_path),), default_timeout=60)
        at.run()
        assert not at.exception
        captions = [c.value for c in at.caption]
        assert any("No saved policy checkpoints" in c for c in captions)
