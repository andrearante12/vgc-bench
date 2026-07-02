"""Unit tests for vgc_bench.team_builder.build_space — no Showdown server required."""

import random

import pytest

from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.pokemon_build import EV_MAX_PER_STAT, EV_MAX_TOTAL


@pytest.fixture(scope="module")
def space():
    return BuildSpace.from_regulation("i")


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

    def test_species_have_tera_types(self, space):
        # Every species in a regulation that supports Tera should have at least
        # one observed tera type from the tournament corpus.
        for sp in space.species_list:
            assert len(space.tera_types_for(sp)) >= 1, f"{sp} has no tera types"

    def test_species_list_sorted(self, space):
        assert space.species_list == sorted(space.species_list)

    def test_invalid_reg_raises(self):
        with pytest.raises(ValueError, match="no team files"):
            BuildSpace.from_regulation("zzz_invalid")

    def test_restricted_limit_is_two_for_reg_i(self, space):
        assert space.restricted_limit == 2

    def test_restricted_species_nonempty_for_reg_i(self, space):
        # Reg I has restricted Pokémon in the tournament corpus (Koraidon, Miraidon, etc.)
        assert len(space.restricted_species) > 0

    def test_restricted_species_subset_of_corpus(self, space):
        # Every entry in restricted_species must also appear in the species list
        for sp in space.restricted_species:
            assert sp in space.species_list, (
                f"{sp} is in restricted_species but not in species_list"
            )


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

    def test_tera_type_is_set(self, space):
        rng = random.Random(0)
        for _ in range(20):
            b = space.random_build(rng=rng)
            assert b.tera_type is not None, "random_build() returned None tera_type"

    def test_tera_type_is_valid(self, space):
        rng = random.Random(0)
        all_types = set(space.tera_types)
        for _ in range(20):
            b = space.random_build(rng=rng)
            assert b.tera_type in all_types, f"tera_type {b.tera_type!r} not in space.tera_types"

    def test_tera_type_respects_species_corpus(self, space):
        # Builds for a species should use that species' observed tera types,
        # not types that were never seen on it in tournament play.
        rng = random.Random(5)
        sp = next(
            s for s in space.species_list if len(space.tera_types_for(s)) > 0
        )
        species_types = set(space.tera_types_for(sp))
        for _ in range(30):
            b = space.random_build(species=sp, rng=rng)
            assert b.tera_type in species_types, (
                f"{sp}: got tera_type {b.tera_type!r}, not in observed set {species_types}"
            )


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


class TestBCWeights:
    def test_species_weights_present(self, space):
        assert len(space.species_weights) == len(space.species_list)

    def test_species_weights_nonuniform(self, space):
        weights = list(space.species_weights.values())
        assert min(weights) != max(weights), "all species weights are equal — BC init did not apply"

    def test_common_species_outweigh_rare(self, space):
        # Incineroar and Rillaboom appear in almost every tournament team
        common = [s for s in ("Incineroar", "Rillaboom") if s in space.species_weights]
        if len(common) < 2:
            return
        all_weights = list(space.species_weights.values())
        mean_w = sum(all_weights) / len(all_weights)
        for sp in common:
            assert space.species_weights[sp] > mean_w, f"{sp} weight below mean — expected common species to be above average"

    def test_move_weights_nonuniform_for_common_species(self, space):
        sp = next((s for s in ("Incineroar", "Rillaboom") if s in space.species_list), None)
        if sp is None:
            return
        weights = list(space._data[sp].move_weights.values())
        assert min(weights) != max(weights), f"{sp} move weights are all equal — BC init did not apply"

    def test_move_weights_sum_to_n_moves(self, space):
        # _normalize keeps mean = 1, so sum == len
        for sp in space.species_list[:20]:
            mw = space._data[sp].move_weights
            assert abs(sum(mw.values()) - len(mw)) < 1e-6, f"{sp}: move weight sum != n_moves"

    def test_item_weights_sum_to_n_items(self, space):
        for sp in space.species_list[:20]:
            iw = space._data[sp].item_weights
            if iw:
                assert abs(sum(iw.values()) - len(iw)) < 1e-6, f"{sp}: item weight sum != n_items"

    def test_species_order_is_weighted(self, space):
        # Draw many orderings and check that high-weight species appear in position 0 more often
        rng = random.Random(0)
        common = [s for s in ("Incineroar", "Rillaboom") if s in space.species_weights]
        if not common:
            return
        hits = sum(1 for _ in range(200) if space.species_order(rng)[0] in common)
        # With uniform random, probability = 2/N_species (very small). With weighting, much higher.
        assert hits > 5, f"high-weight species only led {hits}/200 orderings — weighting not working"


class TestNatEvCorpus:
    def test_corpus_nonempty_for_common_species(self, space):
        # Common reg I species should have tournament (nature, evs) pairs
        common = [s for s in space.species_list if s in ("Incineroar", "Rillaboom", "Flutter Mane")]
        for sp in common:
            corpus = space.nat_ev_corpus_for(sp)
            assert len(corpus) > 0, f"{sp} has no nat_ev_corpus entries"

    def test_corpus_entries_are_valid(self, space):
        rng = random.Random(0)
        sample = rng.sample(space.species_list, min(20, len(space.species_list)))
        for sp in sample:
            for nature, evs in space.nat_ev_corpus_for(sp):
                assert isinstance(nature, str) and nature
                assert len(evs) == 6
                assert 0 <= sum(evs) <= EV_MAX_TOTAL, f"{sp}: EV sum {sum(evs)}"
                for ev in evs:
                    assert 0 <= ev <= EV_MAX_PER_STAT, f"{sp}: EV {ev}"

    def test_random_build_sometimes_uses_corpus(self, space):
        # With 60% BC probability, most builds should match a corpus entry
        rng = random.Random(0)
        sp = "Incineroar" if "Incineroar" in space.species_list else space.species_list[0]
        corpus = {(n, e) for n, e in space.nat_ev_corpus_for(sp)}
        if not corpus:
            return
        hits = sum(
            1 for _ in range(100)
            if (b := space.random_build(species=sp, rng=rng)) and (b.nature, b.evs) in corpus
        )
        assert hits >= 40, f"only {hits}/100 builds matched corpus (expected ~60)"
