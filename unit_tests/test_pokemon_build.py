"""Unit tests for vgc_bench.team_builder.pokemon_build — no Showdown server required."""

import pytest

from vgc_bench.team_builder.pokemon_build import PokemonBuild

_VALID_KWARGS = dict(
    species="Charizard",
    item="Choice Scarf",
    ability="Solar Power",
    tera_type="Fire",
    nature="Timid",
    evs=(0, 0, 0, 32, 2, 32),   # compact stat-point format: max 32/stat, total ≤ 66
    ivs=(31, 31, 31, 31, 31, 31),
    moves=("Heat Wave", "Air Slash", "Dragon Pulse", "Protect"),
)


def _make(**overrides) -> PokemonBuild:
    kwargs = dict(_VALID_KWARGS)
    kwargs.update(overrides)
    return PokemonBuild(**kwargs)


class TestPokemonBuildConstruction:
    def test_valid_construction(self):
        b = _make()
        assert b.species == "Charizard"
        assert len(b.moves) == 4

    def test_empty_species_raises(self):
        with pytest.raises(ValueError, match="species"):
            _make(species="")

    def test_wrong_move_count_raises(self):
        with pytest.raises(ValueError, match="4 moves"):
            _make(moves=("Heat Wave", "Air Slash", "Protect"))

    def test_duplicate_moves_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            _make(moves=("Heat Wave", "Heat Wave", "Air Slash", "Protect"))

    def test_ev_over_252_raises(self):
        with pytest.raises(ValueError, match="0–252"):
            _make(evs=(253, 0, 0, 0, 0, 0))

    def test_ev_sum_over_510_raises(self):
        with pytest.raises(ValueError, match="510"):
            _make(evs=(252, 252, 8, 0, 0, 0))

    def test_iv_out_of_range_raises(self):
        with pytest.raises(ValueError, match="0–31"):
            _make(ivs=(32, 31, 31, 31, 31, 31))

    def test_zero_evs_valid(self):
        b = _make(evs=(0, 0, 0, 0, 0, 0))
        assert sum(b.evs) == 0

    def test_max_evs_valid(self):
        b = _make(evs=(2, 0, 0, 32, 0, 32))
        assert sum(b.evs) == 66


class TestPokemonBuildShowdownText:
    def test_species_in_text(self):
        b = _make()
        assert "Charizard" in b.to_showdown_text()

    def test_item_in_text(self):
        b = _make()
        assert "Choice Scarf" in b.to_showdown_text()

    def test_ability_in_text(self):
        b = _make()
        assert "Solar Power" in b.to_showdown_text()

    def test_level_50_present(self):
        b = _make()
        assert "Level: 50" in b.to_showdown_text()

    def test_tera_type_present(self):
        b = _make()
        assert "Tera Type: Fire" in b.to_showdown_text()

    def test_nature_present(self):
        b = _make()
        assert "Timid Nature" in b.to_showdown_text()

    def test_all_moves_present(self):
        b = _make()
        text = b.to_showdown_text()
        for move in b.moves:
            assert f"- {move}" in text

    def test_evs_line_present_when_nonzero(self):
        b = _make(evs=(0, 0, 0, 32, 2, 32))
        assert "EVs:" in b.to_showdown_text()

    def test_evs_line_absent_when_all_zero(self):
        b = _make(evs=(0, 0, 0, 0, 0, 0))
        assert "EVs:" not in b.to_showdown_text()

    def test_ivs_line_absent_when_all_31(self):
        b = _make(ivs=(31, 31, 31, 31, 31, 31))
        assert "IVs:" not in b.to_showdown_text()

    def test_ivs_line_present_when_not_all_31(self):
        b = _make(ivs=(31, 0, 31, 31, 31, 31))
        assert "IVs:" in b.to_showdown_text()

    def test_first_line_format(self):
        b = _make()
        first = b.to_showdown_text().splitlines()[0]
        assert first == "Charizard @ Choice Scarf"


class TestPokemonBuildHashability:
    def test_equal_builds_equal(self):
        b1 = _make()
        b2 = _make()
        assert b1 == b2
        assert hash(b1) == hash(b2)

    def test_different_species_not_equal(self):
        b1 = _make(species="Charizard")
        b2 = _make(species="Incineroar")
        assert b1 != b2

    def test_different_moves_not_equal(self):
        b1 = _make(moves=("Heat Wave", "Air Slash", "Dragon Pulse", "Protect"))
        b2 = _make(moves=("Heat Wave", "Air Slash", "Dragon Pulse", "Flamethrower"))
        assert b1 != b2

    def test_usable_as_dict_key(self):
        b = _make()
        d = {b: 0.75}
        assert d[b] == 0.75
