"""
Unit tests for vgc_bench.team_builder.viz — parser, sprite ids, and the run loader.

No Showdown server, GPU, or network required: the loader is exercised against tiny
fixture run dirs (both RL SP-PSRO and evolutionary layouts) built in tmp_path.
"""

import json
import os
import time

import pytest

from vgc_bench.team_builder.viz.loader import (
    LIVE_FRESHNESS_SECONDS,
    compute_status,
    detect_run_type,
    load_run,
)
from vgc_bench.team_builder.viz.parse import parse_header, parse_team, parse_teams_file
from vgc_bench.team_builder.viz.runs import scan_runs
from vgc_bench.team_builder.viz.sprites import sprite_url

# A minimal but valid snapshot: header line + 6 members (Showdown paste).
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
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
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


# ------------------------------------------------------------------ parse ---
class TestParse:
    def test_parse_team_members_and_fields(self):
        team = parse_team(_snap(10, 0.75))
        assert team.win_rate == pytest.approx(0.75)
        assert len(team.members) == 6
        cinccino = team.members[0]
        assert cinccino.species == "Cinccino"
        assert cinccino.item == "Life Orb"
        assert cinccino.ability == "Skill Link"
        assert cinccino.nature == "Jolly"
        assert cinccino.moves == ("Protect", "Rock Blast", "Bullet Seed", "Triple Axel")
        assert cinccino.evs == {"HP": 4, "Atk": 252, "Spe": 252}

    def test_optional_lines_tolerated(self):
        # Urshifu here has no EVs line and no Tera line; must still parse.
        team = parse_team(_snap(1, 0.5))
        urshifu = team.members[2]
        assert urshifu.species == "Urshifu-Rapid-Strike"
        assert urshifu.evs == {}
        assert urshifu.tera is None
        # Calyrex carries a Tera line.
        assert team.members[1].tera == "Fairy"

    def test_parse_header(self):
        h = parse_header(_snap(40, 1.0))
        assert h["step"] == 40
        assert h["win_rate"] == pytest.approx(1.0)

    def test_parse_teams_file_splits_on_separator(self):
        text = (
            "# Team 1  win_rate=1.000\n"
            + _snap(0, 1.0).split("\n", 1)[1]
            + "\n---\n"
            + "# Team 2  win_rate=0.500\n"
            + _snap(0, 0.5).split("\n", 1)[1]
        )
        teams = parse_teams_file(text)
        assert len(teams) == 2
        assert all(len(t.members) == 6 for t in teams)


# ----------------------------------------------------------------- sprites ---
class TestSprites:
    @pytest.mark.parametrize(
        "species,expected",
        [
            ("Calyrex-Shadow", "calyrex-shadow.png"),
            ("Urshifu-Rapid-Strike", "urshifu-rapidstrike.png"),
            ("Flutter Mane", "fluttermane.png"),
            ("Chien-Pao", "chienpao.png"),
            ("Ogerpon-Wellspring", "ogerpon-wellspring.png"),
        ],
    )
    def test_form_aware_sprite_ids(self, species, expected):
        assert sprite_url(species).endswith(expected)

    def test_unknown_species_falls_back_to_toid(self):
        assert sprite_url("Totally Fake Mon").endswith("totallyfakemon.png")


# ------------------------------------------------------------- loader (RL) ---
def _write_rl_run(root):
    """Build a minimal RL SP-PSRO run dir with one β and one ν phase."""
    for phase in ("beta_iter000", "nu_iter000"):
        snaps = root / phase / "snapshots"
        snaps.mkdir(parents=True)
        (snaps / "step_000010.txt").write_text(_snap(10, 0.5), encoding="utf-8")
        (snaps / "step_000020.txt").write_text(_snap(20, 1.0), encoding="utf-8")
        # metrics.jsonl empty — mimics a short run (log_every > steps).
        (root / phase / "metrics.jsonl").write_text("", encoding="utf-8")
    payoff = [[0.5, 0.0], [1.0, 0.5]]
    (root / "payoff_iter000.json").write_text(json.dumps(payoff), encoding="utf-8")
    ckpt = {
        "type": "sp_psro",
        "completed_iterations": 1,
        "population_texts": [_snap(20, 1.0), _snap(10, 0.5)],
        "nash_w": [0.7, 0.3],
        "payoff": payoff,
        "found_teams": [],
    }
    (root / "psro_checkpoint.json").write_text(json.dumps(ckpt), encoding="utf-8")


def _write_evo_run(root):
    """Build a minimal evolutionary run dir with one search phase."""
    sdir = root / "search_iter000"
    sdir.mkdir(parents=True)
    (sdir / "round_000.txt").write_text(
        _snap(0, 0.4).replace("step=0", "round=0"), encoding="utf-8"
    )
    (sdir / "round_001.txt").write_text(
        _snap(1, 0.9).replace("step=1", "round=1"), encoding="utf-8"
    )
    (sdir / "rounds.jsonl").write_text(
        json.dumps({"round": 0, "best_win_rate": 0.4, "mean_win_rate": 0.3})
        + "\n"
        + json.dumps({"round": 1, "best_win_rate": 0.9, "mean_win_rate": 0.6})
        + "\n",
        encoding="utf-8",
    )
    payoff = [[0.5, 0.2], [0.8, 0.5]]
    (root / "payoff_iter000.json").write_text(json.dumps(payoff), encoding="utf-8")
    ckpt = {
        "type": "psro",
        "completed_iterations": 1,
        "opponent_teams": [],
        "payoff": payoff,
        "found_teams": [],
    }
    (root / "psro_checkpoint.json").write_text(json.dumps(ckpt), encoding="utf-8")


class TestLoaderRL:
    def test_detect_and_load(self, tmp_path):
        _write_rl_run(tmp_path)
        assert detect_run_type(tmp_path) == "rl"
        run = load_run(tmp_path)
        assert run.run_type == "rl"
        # β then ν, ordered.
        assert [p.name for p in run.phases] == ["iter0 · β", "iter0 · ν"]
        assert len(run.frames) == 4
        # steps ascending within the run.
        assert [f.step for f in run.frames] == [10, 20, 10, 20]
        assert all(len(f.team.members) == 6 for f in run.frames)

    def test_nash_weights_sum_to_one(self, tmp_path):
        _write_rl_run(tmp_path)
        run = load_run(tmp_path)
        assert len(run.iterations) == 1
        w = run.iterations[0].nash_weights
        assert sum(w) == pytest.approx(1.0, abs=1e-6)

    def test_final_teams_ranked_by_nash(self, tmp_path):
        _write_rl_run(tmp_path)
        run = load_run(tmp_path)
        # population_texts + nash_w -> two ranked teams, highest weight first.
        assert len(run.final_teams) == 2
        weights = [w for _, w in run.final_teams]
        assert weights == sorted(weights, reverse=True)
        assert weights[0] == pytest.approx(0.7)


class TestLoaderEvo:
    def test_detect_and_load(self, tmp_path):
        _write_evo_run(tmp_path)
        assert detect_run_type(tmp_path) == "evo"
        run = load_run(tmp_path)
        assert run.run_type == "evo"
        assert [p.name for p in run.phases] == ["iter0 search"]
        assert len(run.frames) == 2
        assert [f.step for f in run.frames] == [0, 1]
        # rounds.jsonl parsed into phase metrics.
        assert run.phases[0].metrics[1]["best_win_rate"] == pytest.approx(0.9)

    def test_iterations_present(self, tmp_path):
        _write_evo_run(tmp_path)
        run = load_run(tmp_path)
        assert len(run.iterations) == 1
        assert run.iterations[0].payoff.shape == (2, 2)


# ------------------------------------------------------- run_meta/live_status ---
def _write_run_meta(root, **fields):
    payload = {"started_at": 0.0, **fields}
    (root / "run_meta.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_live_status(root, updated_at, **fields):
    payload = {"updated_at": updated_at, **fields}
    (root / "live_status.json").write_text(json.dumps(payload), encoding="utf-8")


class TestRunMetaAndLiveStatus:
    def test_load_run_parses_meta_and_live(self, tmp_path):
        _write_rl_run(tmp_path)
        _write_run_meta(tmp_path, run_type="sp_psro", reg="i", target_iterations=5)
        _write_live_status(
            tmp_path, updated_at=1000.0, phase="beta", unit="step", step=3, total=10
        )
        run = load_run(tmp_path)
        assert run.meta == {
            "started_at": 0.0,
            "run_type": "sp_psro",
            "reg": "i",
            "target_iterations": 5,
        }
        assert run.live["phase"] == "beta"
        assert run.last_activity == 1000.0

    def test_load_run_without_meta_or_live_is_none(self, tmp_path):
        _write_rl_run(tmp_path)
        run = load_run(tmp_path)
        assert run.meta is None
        assert run.live is None
        # No live_status.json — falls back to the newest file mtime under the dir.
        assert run.last_activity > 0.0


class TestComputeStatus:
    def test_recent_activity_is_live(self, tmp_path):
        status = compute_status(
            tmp_path,
            meta=None,
            live=None,
            checkpoint=None,
            last_activity=1000.0,
            now=1000.0 + LIVE_FRESHNESS_SECONDS - 1,
        )
        assert status == "live"

    def test_stale_with_replays_dir_is_done(self, tmp_path):
        (tmp_path / "replays").mkdir()
        status = compute_status(
            tmp_path,
            meta=None,
            live=None,
            checkpoint=None,
            last_activity=1000.0,
            now=1000.0 + LIVE_FRESHNESS_SECONDS + 1,
        )
        assert status == "done"

    def test_stale_completed_reaches_target_is_done(self, tmp_path):
        meta = {"target_iterations": 3}
        checkpoint = {"completed_iterations": 3}
        status = compute_status(
            tmp_path,
            meta=meta,
            live=None,
            checkpoint=checkpoint,
            last_activity=1000.0,
            now=1000.0 + LIVE_FRESHNESS_SECONDS + 1,
        )
        assert status == "done"

    def test_stale_incomplete_below_target_is_idle(self, tmp_path):
        meta = {"target_iterations": 3}
        checkpoint = {"completed_iterations": 1}
        status = compute_status(
            tmp_path,
            meta=meta,
            live=None,
            checkpoint=checkpoint,
            last_activity=1000.0,
            now=1000.0 + LIVE_FRESHNESS_SECONDS + 1,
        )
        assert status == "idle"

    def test_stale_interrupted_marker_is_interrupted(self, tmp_path):
        phase_dir = tmp_path / "beta_iter000"
        phase_dir.mkdir()
        (phase_dir / "interrupted_best_team.txt").write_text("x", encoding="utf-8")
        status = compute_status(
            tmp_path,
            meta=None,
            live=None,
            checkpoint=None,
            last_activity=1000.0,
            now=1000.0 + LIVE_FRESHNESS_SECONDS + 1,
        )
        assert status == "interrupted"

    def test_no_signal_at_all_is_idle(self, tmp_path):
        status = compute_status(
            tmp_path,
            meta=None,
            live=None,
            checkpoint=None,
            last_activity=1000.0,
            now=1000.0 + LIVE_FRESHNESS_SECONDS + 1,
        )
        assert status == "idle"


def _backdate(root, seconds_ago):
    """Set every file's mtime under root to simulate a run gone stale."""
    stale = time.time() - seconds_ago
    for p in root.rglob("*"):
        os.utime(p, (stale, stale))


# ------------------------------------------------------------------- runs ---
class TestScanRuns:
    def test_scans_and_sorts_live_first(self, tmp_path):
        # An old, finished run (has replays/, stale last_activity via file mtimes).
        done_dir = tmp_path / "old_done"
        _write_rl_run(done_dir)
        (done_dir / "replays").mkdir()
        _backdate(done_dir, LIVE_FRESHNESS_SECONDS + 60)

        # A run with fresh live_status.json — should sort first regardless of name.
        live_dir = tmp_path / "aaa_live"
        _write_rl_run(live_dir)
        _write_live_status(
            live_dir, updated_at=time.time(), phase="beta", step=1, total=10
        )

        summaries = scan_runs(tmp_path)
        assert [s.name for s in summaries] == ["aaa_live", "old_done"]
        assert summaries[0].status == "live"
        assert summaries[1].status == "done"

    def test_summary_fields_from_meta_and_checkpoint(self, tmp_path):
        run_dir = tmp_path / "run1"
        _write_rl_run(run_dir)
        _write_run_meta(run_dir, run_type="sp_psro", reg="ii", target_iterations=5)

        summaries = scan_runs(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.run_type == "rl"
        assert s.reg == "ii"
        assert s.target_iterations == 5
        assert s.completed_iterations == 1  # from psro_checkpoint.json in _write_rl_run

    def test_empty_results_dir(self, tmp_path):
        assert scan_runs(tmp_path / "does_not_exist") == []
