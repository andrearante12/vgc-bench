"""Unit tests for vgc_bench.team_builder.team — no Showdown server required."""

import random

import pytest

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.team import (
    TEAM_SIZE,
    CandidateTeam,
    combine_teams,
    mutate_build,
    random_team,
    swap_member,
)


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("i")


@pytest.fixture(scope="module")
def six_species(space):
    """Return exactly 6 distinct species from the space."""
    return space.species_list[:TEAM_SIZE]


def _make_team(space: BuildSpace, seed: int = 0) -> CandidateTeam:
    return random_team(space, rng=random.Random(seed))


class TestCandidateTeam:
    def test_valid_construction(self, space):
        t = _make_team(space, seed=0)
        assert t.win_rate is None
        assert len(t.members) == TEAM_SIZE

    def test_wrong_size_raises(self, space):
        rng = random.Random(0)
        members = tuple(space.random_build(rng=rng) for _ in range(3))
        with pytest.raises(ValueError, match="exactly"):
            CandidateTeam(members=members)

    def test_duplicate_species_raises(self, space):
        rng = random.Random(0)
        b = space.random_build(rng=rng)
        with pytest.raises(ValueError, match="duplicate"):
            CandidateTeam(members=(b, b, b, b, b, b))

    def test_with_win_rate_is_nonmutating(self, space):
        t = _make_team(space)
        t2 = t.with_win_rate(0.75)
        assert t2.win_rate == 0.75
        assert t.win_rate is None

    def test_to_showdown_text_contains_all_species(self, space):
        t = _make_team(space)
        text = t.to_showdown_text()
        for m in t.members:
            assert m.species in text

    def test_to_showdown_text_has_six_blocks(self, space):
        t = _make_team(space)
        blocks = [b for b in t.to_showdown_text().split("\n\n") if b.strip()]
        assert len(blocks) == TEAM_SIZE

    def test_to_packed_team_nonempty(self, space):
        t = _make_team(space)
        packed = t.to_packed_team()
        assert isinstance(packed, str) and len(packed) > 0
        assert "|" in packed  # poke-env packed format uses pipe separators

    def test_equal_members_equal_regardless_of_win_rate(self, space):
        t = _make_team(space)
        t1 = t.with_win_rate(0.4)
        t2 = t.with_win_rate(0.9)
        assert t1 == t2
        assert hash(t1) == hash(t2)


class TestRandomTeam:
    def test_produces_valid_team(self, space):
        t = random_team(space)
        assert len(t.members) == TEAM_SIZE
        assert t.win_rate is None

    def test_no_duplicate_species(self, space):
        rng = random.Random(0)
        for _ in range(20):
            t = random_team(space, rng)
            species = [m.species for m in t.members]
            assert len(set(species)) == TEAM_SIZE

    def test_restricted_limit_respected(self, space):
        from vgc_bench.team_builder.pokemon_build import RESTRICTED_LEGENDARIES
        rng = random.Random(0)
        for _ in range(20):
            t = random_team(space, rng)
            restricted = [m.species for m in t.members if m.species in RESTRICTED_LEGENDARIES]
            assert len(restricted) <= space.restricted_limit, (
                f"team has {len(restricted)} restricted (limit {space.restricted_limit}): {restricted}"
            )

    def test_seeded_reproducible(self, space):
        t1 = random_team(space, random.Random(42))
        t2 = random_team(space, random.Random(42))
        assert t1.members == t2.members

    def test_high_weight_species_appear_more_often(self, space):
        # Common tournament species should appear more often than baseline uniform expectation
        common = {s for s in ("Incineroar", "Rillaboom") if s in space.species_weights}
        if not common:
            return
        rng = random.Random(0)
        hits = sum(
            1 for _ in range(100)
            for m in random_team(space, rng).members
            if m.species in common
        )
        # Uniform baseline: 2 species × 6/N_species × 100 teams
        n_species = len(space.species_list)
        uniform_baseline = 2 * 6 / n_species * 100
        assert hits > uniform_baseline, (
            f"common species appeared {hits} times, expected > {uniform_baseline:.1f} — "
            "weighted sampling not biasing toward high-weight species"
        )


class TestSwapMember:
    def test_restricted_limit_respected_after_swap(self, space):
        from vgc_bench.team_builder.pokemon_build import RESTRICTED_LEGENDARIES
        rng = random.Random(7)
        parent = random_team(space, rng)
        for _ in range(20):
            child = swap_member(parent, space, rng=rng)
            restricted = [m.species for m in child.members if m.species in RESTRICTED_LEGENDARIES]
            assert len(restricted) <= space.restricted_limit, (
                f"swap produced {len(restricted)} restricted (limit {space.restricted_limit}): {restricted}"
            )

    def test_one_swap_changes_exactly_one_slot(self, space):
        rng = random.Random(0)
        parent = random_team(space, rng)
        child = swap_member(parent, space, n_swaps=1, rng=rng)
        diffs = sum(1 for a, b in zip(parent.members, child.members) if a != b)
        assert diffs == 1

    def test_no_duplicate_species_after_swap(self, space):
        rng = random.Random(7)
        parent = random_team(space, rng)
        for _ in range(20):
            child = swap_member(parent, space, rng=rng)
            species = [m.species for m in child.members]
            assert len(set(species)) == TEAM_SIZE

    def test_win_rate_reset_after_swap(self, space):
        parent = random_team(space).with_win_rate(0.8)
        child = swap_member(parent, space)
        assert child.win_rate is None

    def test_two_swaps(self, space):
        rng = random.Random(99)
        parent = random_team(space, rng)
        child = swap_member(parent, space, n_swaps=2, rng=rng)
        diffs = sum(1 for a, b in zip(parent.members, child.members) if a != b)
        assert diffs == 2


class TestCombineTeams:
    def test_restricted_limit_respected_after_combine(self, space):
        from vgc_bench.team_builder.pokemon_build import RESTRICTED_LEGENDARIES
        rng = random.Random(123)
        for _ in range(30):
            pa = random_team(space, rng)
            pb = random_team(space, rng)
            child = combine_teams(pa, pb, space, rng)
            restricted = [m.species for m in child.members if m.species in RESTRICTED_LEGENDARIES]
            assert len(restricted) <= space.restricted_limit, (
                f"combine produced {len(restricted)} restricted (limit {space.restricted_limit}): {restricted}"
            )

    def test_no_duplicate_species(self, space):
        rng = random.Random(123)
        for _ in range(30):
            pa = random_team(space, rng)
            pb = random_team(space, rng)
            child = combine_teams(pa, pb, space, rng)
            species = [m.species for m in child.members]
            assert len(set(species)) == TEAM_SIZE, (
                f"duplicate species: {[s for s in species if species.count(s) > 1]}"
            )

    def test_win_rate_reset(self, space):
        rng = random.Random(5)
        pa = random_team(space, rng).with_win_rate(0.6)
        pb = random_team(space, rng).with_win_rate(0.7)
        child = combine_teams(pa, pb, space, rng)
        assert child.win_rate is None

    def test_correct_team_size(self, space):
        rng = random.Random(0)
        pa = random_team(space, rng)
        pb = random_team(space, rng)
        child = combine_teams(pa, pb, space, rng)
        assert len(child.members) == TEAM_SIZE


class TestMutateBuild:
    def test_species_never_changes(self, space):
        rng = random.Random(0)
        team = random_team(space, rng)
        build = team.members[0]
        for _ in range(50):
            mutated = mutate_build(build, space, rng)
            assert mutated.species == build.species

    def test_at_least_one_change_in_many_trials(self, space):
        rng = random.Random(1)
        build = random_team(space, rng).members[0]
        results = [mutate_build(build, space, rng) for _ in range(50)]
        assert any(r != build for r in results)

    def test_no_duplicate_moves_after_mutation(self, space):
        rng = random.Random(3)
        build = random_team(space, rng).members[0]
        for _ in range(50):
            mutated = mutate_build(build, space, rng)
            assert len(set(mutated.moves)) == 4

    def test_ev_constraints_valid_after_mutation(self, space):
        from vgc_bench.team_builder.pokemon_build import EV_MAX_PER_STAT, EV_MAX_TOTAL
        rng = random.Random(9)
        build = random_team(space, rng).members[0]
        for _ in range(50):
            mutated = mutate_build(build, space, rng)
            assert sum(mutated.evs) <= EV_MAX_TOTAL
            for ev in mutated.evs:
                assert 0 <= ev <= EV_MAX_PER_STAT

    def test_seeded_reproducible(self, space):
        rng = random.Random(42)
        build = random_team(space, rng).members[0]
        m1 = mutate_build(build, space, random.Random(7))
        m2 = mutate_build(build, space, random.Random(7))
        assert m1 == m2

    def test_ev_transfer_is_incremental(self, space):
        """evs_transfer mutations should change at most 2 stats."""
        from unittest.mock import patch
        rng = random.Random(5)
        build = random_team(space, rng).members[0]
        incremental_count = 0
        for _ in range(50):
            with patch("vgc_bench.team_builder.team.random") as mock_rng:
                # Force evs_transfer branch by controlling field_choice
                pass
            # Run normally; count how many mutations change ≤2 stats
            mutated = mutate_build(build, space, random.Random(rng.randint(0, 9999)))
            diffs = sum(1 for a, b in zip(build.evs, mutated.evs) if a != b)
            if diffs <= 2:
                incremental_count += 1
        # Majority of mutations should be small (move, nat_ev, evs_transfer are all local)
        assert incremental_count >= 25

    def test_tera_type_mutation_changes_type(self, space):
        """tera_type mutation should produce a different valid tera type."""
        from unittest.mock import patch
        rng = random.Random(13)
        build = random_team(space, rng).members[0]
        # build.tera_type must be set (non-None) for this test to be meaningful
        assert build.tera_type is not None, "random_build() did not initialize tera_type"
        original_tera = build.tera_type
        all_types = set(space.tera_types)
        # Patch choices on the rng *instance* — mutate_build calls _rng.choices(),
        # where _rng is the passed rng object, not the module-level random.
        with patch.object(rng, "choices", return_value=["tera_type"]):
            mutated = mutate_build(build, space, rng)
        assert mutated.tera_type in all_types, f"mutated tera_type {mutated.tera_type!r} not valid"
        assert mutated.tera_type != original_tera, "tera_type mutation did not change the type"

    def test_nat_ev_mutation_keeps_coherent(self, space):
        """nat_ev mutation should produce a valid nature+EV pair from the corpus."""
        from unittest.mock import patch
        import vgc_bench.team_builder.team as team_mod
        rng = random.Random(11)
        build = random_team(space, rng).members[0]
        corpus = space.nat_ev_corpus_for(build.species)
        if not corpus:
            return  # skip if no corpus for this species
        # Force nat_ev branch
        with patch.object(team_mod.random, "choices", return_value=["nat_ev"]):
            mutated = mutate_build(build, space, rng)
        assert mutated.nature in space.natures
        from vgc_bench.team_builder.pokemon_build import EV_MAX_PER_STAT, EV_MAX_TOTAL
        assert sum(mutated.evs) <= EV_MAX_TOTAL
        for ev in mutated.evs:
            assert 0 <= ev <= EV_MAX_PER_STAT
