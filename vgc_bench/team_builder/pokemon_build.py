"""
Structured representation of a single Pokémon build for the VGC-Bench team builder.

PokemonBuild holds every field that defines a competitive build — species, held
item, ability, tera type, nature, EV spread, IVs, and moves — as typed fields
rather than a raw Showdown text block. This enables per-field mutation operators
(change one move, re-roll EVs, etc.) and provides a to_showdown_text() method
for conversion back to the format the evaluator expects.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# Stat order used throughout poke-env and Showdown: HP, Atk, Def, SpA, SpD, Spe
_STAT_NAMES: tuple[str, ...] = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")

# Standard Gen 9 VGC uses official EVs: 0–252 per stat, total ≤ 510.
EV_MAX_PER_STAT: int = 252
EV_MAX_TOTAL: int = 510
IV_MAX: int = 31

# Pokémon that count toward the "Limit N Restricted" rule in VGC formats.
# Covers all restricted legendaries introduced through Gen 9.
# ---------------------------------------------------------------------------
# Species Clause grouping
# ---------------------------------------------------------------------------
# Maps variant/form species names → the canonical name used for Species Clause
# checking.  Any two species that map to the same key cannot coexist on a team.
#
# Regional variants (-Alola, -Galar, -Hisui, -Paldean) are intentionally kept
# separate because they are treated as distinct Pokémon in competitive play and
# Showdown does not group them under the same clause key.
#
# Gender forms (e.g. Indeedee / Indeedee-F) are also kept separate because
# competitive teams regularly run both and Showdown accepts them as a pair.
_SPECIES_CLAUSE_GROUPS: dict[str, str] = {
    # Ogerpon mask forms (dex #1017)
    "Ogerpon-Cornerstone": "Ogerpon",
    "Ogerpon-Hearthflame": "Ogerpon",
    "Ogerpon-Wellspring":  "Ogerpon",
    # Calyrex fusions (dex #898)
    "Calyrex-Ice":    "Calyrex",
    "Calyrex-Shadow": "Calyrex",
    # Urshifu styles (dex #892)
    "Urshifu-Rapid-Strike": "Urshifu",
    # Zacian (dex #888)
    "Zacian-Crowned": "Zacian",
    # Zamazenta (dex #889)
    "Zamazenta-Crowned": "Zamazenta",
    # Dialga (dex #483)
    "Dialga-Origin": "Dialga",
    # Giratina (dex #487)
    "Giratina-Origin": "Giratina",
    # Necrozma fusions (dex #800)
    "Necrozma-Dusk-Mane":  "Necrozma",
    "Necrozma-Dawn-Wings": "Necrozma",
    "Necrozma-Ultra":      "Necrozma",
    # Kyurem fusions (dex #646)
    "Kyurem-Black": "Kyurem",
    "Kyurem-White": "Kyurem",
    # Ursaluna (dex #901)
    "Ursaluna-Bloodmoon": "Ursaluna",
    # Terapagos (dex #1024)
    "Terapagos-Terastal": "Terapagos",
    "Terapagos-Stellar":  "Terapagos",
    # Landorus (dex #645)
    "Landorus-Therian": "Landorus",
    # Tatsugiri forms (dex #978)
    "Tatsugiri-Droopy":    "Tatsugiri",
    "Tatsugiri-Stretchy":  "Tatsugiri",
    # Sinistcha forms (dex #1013)
    "Sinistcha-Masterpiece": "Sinistcha",
    # Gastrodon (dex #423)
    "Gastrodon-East": "Gastrodon",
    # Rotom forms (dex #479)
    "Rotom-Heat":  "Rotom",
    "Rotom-Wash":  "Rotom",
    "Rotom-Frost": "Rotom",
    "Rotom-Fan":   "Rotom",
    "Rotom-Mow":   "Rotom",
}


def species_clause_key(species: str) -> str:
    """Return the Species Clause key for a species name.

    Two species that return the same key cannot coexist on the same team.
    For most species this is the species name itself; for form variants it
    is the canonical base-species name (e.g. "Ogerpon-Wellspring" → "Ogerpon").
    """
    return _SPECIES_CLAUSE_GROUPS.get(species, species)


RESTRICTED_LEGENDARIES: frozenset[str] = frozenset({
    # Gen 1
    "Mewtwo",
    # Gen 2
    "Lugia", "Ho-Oh",
    # Gen 3
    "Kyogre", "Groudon", "Rayquaza",
    # Gen 4
    "Dialga", "Dialga-Origin", "Palkia", "Palkia-Origin",
    "Giratina", "Giratina-Origin",
    # Gen 5
    "Reshiram", "Zekrom", "Kyurem", "Kyurem-Black", "Kyurem-White",
    # Gen 6
    "Xerneas", "Yveltal", "Zygarde",
    # Gen 7
    "Cosmog", "Cosmoem", "Solgaleo", "Lunala",
    "Necrozma", "Necrozma-Dawn-Wings", "Necrozma-Dusk-Mane", "Necrozma-Ultra",
    # Gen 8
    "Zacian", "Zacian-Crowned", "Zamazenta", "Zamazenta-Crowned", "Eternatus",
    "Kubfu", "Urshifu", "Urshifu-Rapid-Strike",
    "Calyrex", "Calyrex-Ice", "Calyrex-Shadow",
    # Gen 9
    "Koraidon", "Miraidon",
    "Terapagos",
})


@dataclass(frozen=True)
class PokemonBuild:
    """
    An immutable description of a single competitive Pokémon build.

    All fields are individually typed so mutation operators can replace exactly
    one field at a time via dataclasses.replace(). The instance is hashable,
    making it usable as a cache key component inside CandidateTeam.

    Attributes:
        species:   Pokémon species name (e.g. "Charizard", "Ogerpon-Wellspring").
        item:      Held item name (e.g. "Choice Scarf").
        ability:   Ability name (e.g. "Solar Power").
        tera_type: Tera type name (e.g. "Fire"), or None for formats without Tera.
        nature:    Nature name (e.g. "Timid").
        evs:       EV spread (HP, Atk, Def, SpA, SpD, Spe). Each 0–252, mult of 4,
                   total ≤ 508.
        ivs:       IV spread (HP, Atk, Def, SpA, SpD, Spe). Each 0–31.
                   Defaults to all 31; the IVs line is omitted from Showdown text
                   when all are 31 (treated as all-max by the server).
        moves:     Exactly 4 distinct move names.
    """

    species: str
    item: str
    ability: str
    nature: str
    evs: tuple[int, int, int, int, int, int]
    ivs: tuple[int, int, int, int, int, int]
    moves: tuple[str, str, str, str]
    tera_type: str | None = None

    def __post_init__(self) -> None:
        if not self.species:
            raise ValueError("species must be non-empty")
        if len(self.moves) != 4:
            raise ValueError(f"exactly 4 moves required, got {len(self.moves)}")
        if len(set(self.moves)) != 4:
            raise ValueError(f"duplicate moves: {self.moves}")
        if len(self.evs) != 6:
            raise ValueError(f"evs must have 6 values, got {len(self.evs)}")
        for i, ev in enumerate(self.evs):
            if not (0 <= ev <= EV_MAX_PER_STAT):
                raise ValueError(
                    f"EV for {_STAT_NAMES[i]} must be 0–{EV_MAX_PER_STAT}, got {ev}"
                )
        if sum(self.evs) > EV_MAX_TOTAL:
            raise ValueError(
                f"total EVs must be ≤ {EV_MAX_TOTAL}, got {sum(self.evs)}"
            )
        if len(self.ivs) != 6:
            raise ValueError(f"ivs must have 6 values, got {len(self.ivs)}")
        for i, iv in enumerate(self.ivs):
            if not (0 <= iv <= IV_MAX):
                raise ValueError(
                    f"IV for {_STAT_NAMES[i]} must be 0–{IV_MAX}, got {iv}"
                )

    def to_showdown_text(self) -> str:
        """
        Render this build as a Showdown-format text block.

        The EVs line is omitted when all EVs are zero. The IVs line is omitted
        when all IVs are 31 (Showdown treats absent IVs as all-max).
        """
        lines: list[str] = []
        lines.append(f"{self.species} @ {self.item}")
        lines.append(f"Ability: {self.ability}")
        lines.append("Level: 50")
        if self.tera_type is not None:
            lines.append(f"Tera Type: {self.tera_type}")

        ev_parts = [
            f"{v} {name}"
            for v, name in zip(self.evs, _STAT_NAMES)
            if v > 0
        ]
        if ev_parts:
            lines.append("EVs: " + " / ".join(ev_parts))

        lines.append(f"{self.nature} Nature")

        if any(iv != IV_MAX for iv in self.ivs):
            iv_parts = [
                f"{v} {name}"
                for v, name in zip(self.ivs, _STAT_NAMES)
                if v != IV_MAX
            ]
            lines.append("IVs: " + " / ".join(iv_parts))

        for move in self.moves:
            lines.append(f"- {move}")

        return "\n".join(lines)

    def replace(self, **changes) -> PokemonBuild:
        """Return a new PokemonBuild with the given fields replaced."""
        return dataclasses.replace(self, **changes)
