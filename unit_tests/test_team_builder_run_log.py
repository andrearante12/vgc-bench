"""
Unit tests for the run_meta.json / live_status.json logging added to the RL and
evolutionary team-builder training loops — no Showdown server or GPU required.

RL is driven serverlessly via train_policy's/run_sp_psro's eval_fn hook. The
evolutionary loop has no eval_fn hook, so TeamEvaluator is monkeypatched instead.
"""

from __future__ import annotations

import json
import random

import pytest

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.evaluator import TeamEvaluator
from vgc_bench.team_builder.optimizer import SearchConfig
from vgc_bench.team_builder.psro import PSROConfig, run_psro
from vgc_bench.team_builder.rl.train import run_sp_psro
from vgc_bench.team_builder.team import random_team


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("ma")


@pytest.fixture
def meta_team_path(space, tmp_path):
    team = random_team(space, rng=random.Random(0))
    path = tmp_path / "meta.txt"
    path.write_text(team.to_showdown_text(), encoding="utf-8")
    return path


def _stub_eval(team, opp_text) -> float:
    return random.random()


class TestRLRunLog:
    def test_run_meta_and_live_status_written(self, space, meta_team_path, tmp_path):
        out = tmp_path / "run"
        ranked = run_sp_psro(
            meta_team_paths=[str(meta_team_path)],
            space=space,
            reg="ma",
            n_iterations=1,
            n_steps=6,
            log_every=2,
            snapshot_every=3,
            quality_threshold=0.0,
            output_dir=out,
            eval_fn=_stub_eval,
            resume=False,
        )
        assert len(ranked) >= 1

        meta = json.loads((out / "run_meta.json").read_text())
        assert meta["run_type"] == "sp_psro"
        assert meta["reg"] == "ma"
        assert meta["target_iterations"] == 1
        assert meta["n_steps"] == 6
        assert meta["log_every"] == 2
        assert "started_at" in meta

        live = json.loads((out / "live_status.json").read_text())
        assert live["phase"] in ("beta", "nu")
        assert live["unit"] == "step"
        assert live["step"] == live["total"] == 6
        assert live["current_team_showdown"]
        assert live["best_team_showdown"]
        assert "updated_at" in live

    def test_replays_skipped_under_eval_fn(self, space, meta_team_path, tmp_path):
        out = tmp_path / "run"
        run_sp_psro(
            meta_team_paths=[str(meta_team_path)],
            space=space,
            reg="ma",
            n_iterations=1,
            n_steps=4,
            log_every=2,
            snapshot_every=2,
            quality_threshold=0.0,
            output_dir=out,
            eval_fn=_stub_eval,
            resume=False,
        )
        # Replays require a live battle backend, which eval_fn has none of.
        assert not (out / "replays").exists()

    def test_snapshots_and_metrics_at_cadence(self, space, meta_team_path, tmp_path):
        out = tmp_path / "run"
        run_sp_psro(
            meta_team_paths=[str(meta_team_path)],
            space=space,
            reg="ma",
            n_iterations=1,
            n_steps=6,
            log_every=2,
            snapshot_every=3,
            quality_threshold=0.0,
            output_dir=out,
            eval_fn=_stub_eval,
            resume=False,
        )
        beta_dir = out / "beta_iter000"
        snap_names = sorted(p.name for p in (beta_dir / "snapshots").iterdir())
        assert snap_names == ["step_000003.txt", "step_000006.txt"]
        metrics_lines = (beta_dir / "metrics.jsonl").read_text().splitlines()
        assert len(metrics_lines) == 3  # one every log_every=2 steps, over 6 steps

    def test_run_meta_not_overwritten_on_resume(self, space, meta_team_path, tmp_path):
        out = tmp_path / "run"
        run_sp_psro(
            meta_team_paths=[str(meta_team_path)],
            space=space,
            reg="ma",
            n_iterations=1,
            n_steps=2,
            log_every=1,
            snapshot_every=1,
            quality_threshold=0.0,
            output_dir=out,
            eval_fn=_stub_eval,
            resume=False,
        )
        first_started_at = json.loads((out / "run_meta.json").read_text())["started_at"]

        # A second call with resume=True (checkpoint already complete) must not
        # rewrite run_meta.json — started_at should be preserved.
        run_sp_psro(
            meta_team_paths=[str(meta_team_path)],
            space=space,
            reg="ma",
            n_iterations=1,
            n_steps=2,
            log_every=1,
            snapshot_every=1,
            quality_threshold=0.0,
            output_dir=out,
            eval_fn=_stub_eval,
            resume=True,
        )
        second_started_at = json.loads((out / "run_meta.json").read_text())[
            "started_at"
        ]
        assert first_started_at == second_started_at


class TestEvoRunLog:
    @pytest.fixture(autouse=True)
    def _stub_team_evaluator(self, monkeypatch):
        """TeamEvaluator has no eval_fn hook — stub it for a serverless test."""

        def fake_init(self, **kwargs):
            pass

        def fake_evaluate(self, team):
            return team.with_win_rate(random.random())

        def fake_evaluate_vs_mixture(self, team, opponents, weights):
            return team.with_win_rate(random.random())

        monkeypatch.setattr(TeamEvaluator, "__init__", fake_init)
        monkeypatch.setattr(TeamEvaluator, "evaluate", fake_evaluate)
        monkeypatch.setattr(
            TeamEvaluator, "evaluate_vs_mixture", fake_evaluate_vs_mixture
        )

    def test_run_meta_and_live_status_written(self, space, tmp_path):
        out = tmp_path / "evo_run"
        team = random_team(space, rng=random.Random(1))
        cfg = PSROConfig(
            n_iterations=1,
            search_config=SearchConfig(
                rounds=2, population=3, candidates=3, n_battles=1, reg="ma"
            ),
            reg="ma",
            output_dir=out,
        )
        found = run_psro(
            cfg, meta_teams=[team.to_packed_team()], space=space, resume=False
        )
        assert len(found) == 1

        meta = json.loads((out / "run_meta.json").read_text())
        assert meta["run_type"] == "psro"
        assert meta["reg"] == "ma"
        assert meta["target_iterations"] == 1
        assert meta["rounds_per_iteration"] == 2

        live = json.loads((out / "live_status.json").read_text())
        assert live["phase"] == "search"
        assert live["unit"] == "round"
        assert live["step"] == live["total"] == 2
        assert live["current_team_showdown"]

    def test_live_status_at_run_root_not_search_iter_dir(self, space, tmp_path):
        out = tmp_path / "evo_run"
        team = random_team(space, rng=random.Random(2))
        cfg = PSROConfig(
            n_iterations=1,
            search_config=SearchConfig(
                rounds=1, population=2, candidates=2, n_battles=1, reg="ma"
            ),
            reg="ma",
            output_dir=out,
        )
        run_psro(cfg, meta_teams=[team.to_packed_team()], space=space, resume=False)
        assert (out / "live_status.json").exists()
        assert not (out / "search_iter000" / "live_status.json").exists()
