"""
Unit tests for vgc_bench.team_builder.viz.app — the sidebar run browser and the
Live/Playback mode routing, rendered headlessly via Streamlit's AppTest.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.rl.train import run_sp_psro
from vgc_bench.team_builder.team import random_team

_APP_PATH = "vgc_bench/team_builder/viz/app.py"
_ABS_APP_PATH = str(Path(__file__).resolve().parents[1] / _APP_PATH)


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("ma")


@pytest.fixture
def fresh_rl_run(space, tmp_path):
    """A just-finished RL run outside results/ — its live_status.json is fresh."""
    meta_path = tmp_path / "meta.txt"
    meta_path.write_text(
        random_team(space, rng=random.Random(0)).to_showdown_text(), encoding="utf-8"
    )
    out = tmp_path / "run"
    run_sp_psro(
        meta_team_paths=[str(meta_path)],
        space=space,
        reg="ma",
        n_iterations=1,
        n_steps=4,
        log_every=2,
        snapshot_every=2,
        quality_threshold=0.0,
        output_dir=out,
        eval_fn=lambda team, opp_text: random.random(),
        resume=False,
    )
    return out


class TestModeDefaulting:
    def test_fresh_run_outside_results_defaults_to_live(
        self, fresh_rl_run, monkeypatch
    ):
        monkeypatch.setattr(sys, "argv", ["app.py", str(fresh_rl_run)])
        at = AppTest.from_file(_ABS_APP_PATH, default_timeout=90)
        at.run()
        assert not at.exception
        assert at.sidebar.radio[0].value == "Live"
        assert "Live training" in [s.value for s in at.subheader]

    def test_finished_results_run_defaults_to_playback(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["app.py", "results/dummy_gpu"])
        at = AppTest.from_file(_ABS_APP_PATH, default_timeout=90)
        at.run()
        assert not at.exception
        assert at.sidebar.radio[0].value == "Playback"
        assert "Playback" in [s.value for s in at.subheader]


class TestModeSwitching:
    def test_explicit_switch_to_playback_renders(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["app.py", "results/dummy_gpu"])
        at = AppTest.from_file(_ABS_APP_PATH, default_timeout=90)
        at.run()
        at.sidebar.radio[0].set_value("Playback").run()
        assert not at.exception
        assert "Generate a fresh team from a saved checkpoint" in [
            s.value for s in at.subheader
        ]

    def test_switching_run_updates_selectbox_and_status(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["app.py", "results/dummy_gpu"])
        at = AppTest.from_file(_ABS_APP_PATH, default_timeout=90)
        at.run()
        at.sidebar.selectbox[0].set_value("results/basic_bcsp").run()
        assert not at.exception
        # basic_bcsp has no run_meta.json/replays and stale mtimes -> idle -> Playback.
        assert at.sidebar.radio[0].value == "Playback"


class TestNoRunsFound:
    def test_missing_results_dir_shows_error_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no results/ dir here, and no CLI arg
        monkeypatch.setattr(sys, "argv", ["app.py"])
        at = AppTest.from_file(_ABS_APP_PATH, default_timeout=60)
        at.run()
        assert not at.exception
        assert any("No runs found" in e.value for e in at.error)
