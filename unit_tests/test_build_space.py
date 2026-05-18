"""Unit tests for vgc_bench.team_builder.build_space — no Showdown server required."""

import random

import pytest

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.pokemon_build import EV_MAX_PER_STAT, EV_MAX_TOTAL


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("ma")


class TestBuildSpaceFromRegulation:
    def test_returns_build_space(self, space):
        assert isinstance(space, BuildSpace)

    def test_has_species(self, space):
        assert len(space.species_list) > 0

    def test_all_species_have_at_least_4_moves(self, space):
        for sp in space.species_list:
            moves = space.moves_for(sp)
            assert len(moves) >= 4, f"{sp} has only {len(moves)} moves"

    def test_all_species_have_at_least_1_item(self, space):
        for sp in space.species_list:
            assert len(space.items_for(sp)) >= 1, f"{sp} has no items"

    def test_all_species_have_at_least_1_ability(self, space):
        for sp in space.species_list:
            assert len(space.abilities_for(sp)) >= 1, f"{sp} has no abilities"

    def test_natures_nonempty(self, space):
        assert len(space.natures) > 0

    def test_tera_types_nonempty(self, space):
        assert len(space.tera_types) > 0

    def test_species_list_sorted(self, space):
        assert space.species_list == sorted(space.species_list)

    def test_invalid_reg_raises(self):
        with pytest.raises(ValueError, match="no team files"):
            BuildSpace.from_regulation("zzz_invalid")


class TestRandomBuild:
    def test_returns_pokemon_build(self, space):
        from vgc_bench.team_builder.pokemon_build import PokemonBuild
        b = space.random_build()
        assert isinstance(b, PokemonBuild)

    def test_seeded_reproducible(self, space):
        b1 = space.random_build(rng=random.Random(42))
        b2 = space.random_build(rng=random.Random(42))
        assert b1 == b2

    def test_specific_species_honored(self, space):
        sp = space.species_list[0]
        b = space.random_build(species=sp)
        assert b.species == sp

    def test_moves_valid_for_species(self, space):
        rng = random.Random(0)
        for _ in range(20):
            b = space.random_build(rng=rng)
            allowed = set(space.moves_for(b.species))
            for m in b.moves:
                assert m in allowed, f"{m} not in move pool for {b.species}"

    def test_no_duplicate_moves(self, space):
        rng = random.Random(7)
        for _ in range(20):
            b = space.random_build(rng=rng)
            assert len(set(b.moves)) == 4

    def test_ev_constraints_valid(self, space):
        rng = random.Random(99)
        for _ in range(50):
            b = space.random_build(rng=rng)
            assert sum(b.evs) <= EV_MAX_TOTAL
            for ev in b.evs:
                assert 0 <= ev <= EV_MAX_PER_STAT

    def test_all_ivs_31(self, space):
        b = space.random_build(rng=random.Random(1))
        assert b.ivs == (31, 31, 31, 31, 31, 31)


class TestRandomEvs:
    def test_500_samples_no_violations(self, space):
        rng = random.Random(0)
        for _ in range(500):
            evs = space.random_evs(rng)
            assert len(evs) == 6
            assert 1 <= sum(evs) <= EV_MAX_TOTAL, f"EV sum {sum(evs)} out of range [1, {EV_MAX_TOTAL}]"
            for ev in evs:
                assert 0 <= ev <= EV_MAX_PER_STAT, f"EV {ev} out of range"

    def test_seeded_reproducible(self, space):
        e1 = space.random_evs(random.Random(5))
        e2 = space.random_evs(random.Random(5))
        assert e1 == e2
