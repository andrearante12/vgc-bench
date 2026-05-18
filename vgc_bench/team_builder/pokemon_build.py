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

# Pokemon Champions format (gen9championsvgc2026regma) uses compact "stat points":
# 0–32 per stat, total ≤ 66. These are NOT standard EVs — do not multiply by 8.
EV_MAX_PER_STAT: int = 32
EV_MAX_TOTAL: int = 66
IV_MAX: int = 31


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
        tera_type: Tera type name (e.g. "Fire").
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
    tera_type: str
    nature: str
    evs: tuple[int, int, int, int, int, int]
    ivs: tuple[int, int, int, int, int, int]
    moves: tuple[str, str, str, str]

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
