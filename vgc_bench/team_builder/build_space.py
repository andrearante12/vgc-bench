"""
Build space for the VGC-Bench from-scratch team builder.

BuildSpace defines the valid search space for a given VGC regulation: which
species are available, and for each species which moves, items, and abilities
have been observed in tournament play. It provides random_build() and
random_evs() to generate fresh PokemonBuild instances from scratch without
copying pre-existing tournament build blocks.

Tournament data is the source of truth for what is competitive and format-legal
in a given regulation. No full learnset data is required — only moves actually
used by top players are included, which keeps the search space focused.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.team_builder.pokemon_build import EV_MAX_PER_STAT, EV_MAX_TOTAL, PokemonBuild

_ALL_NATURES: list[str] = [
    "Adamant", "Bashful", "Bold", "Brave", "Calm",
    "Careful", "Docile", "Gentle", "Hardy", "Hasty",
    "Impish", "Jolly", "Lax", "Lonely", "Mild",
    "Modest", "Naive", "Naughty", "Quiet", "Quirky",
    "Rash", "Relaxed", "Sassy", "Serious", "Timid",
]

_ALL_TERA_TYPES: list[str] = [
    "Bug", "Dark", "Dragon", "Electric", "Fairy",
    "Fighting", "Fire", "Flying", "Ghost", "Grass",
    "Ground", "Ice", "Normal", "Poison", "Psychic",
    "Rock", "Steel", "Water",
]

# EV allocation constants (compact stat-point format: 0-32 per stat, total ≤ 66)
_MAX_UNITS_PER_STAT = EV_MAX_PER_STAT   # 32
_MAX_UNITS_TOTAL = EV_MAX_TOTAL          # 66


@dataclass(frozen=True)
class _SpeciesData:
    moves: list[str]
    items: list[str]
    abilities: list[str]


@dataclass
class BuildSpace:
    """
    Valid search space for generating Pokémon builds from scratch.

    Populated from tournament team files so every species, move, item, and
    ability in the space is guaranteed to be format-legal and competitively
    relevant.

    Attributes:
        natures:    Sorted list of nature names observed in the corpus.
        tera_types: Sorted list of tera type names observed in the corpus.
    """

    _data: dict[str, _SpeciesData] = field(repr=False)
    natures: list[str]
    tera_types: list[str]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_regulation(cls, reg: str = "ma") -> BuildSpace:
        """
        Build a BuildSpace from all tournament team files for the given regulation.

        Args:
            reg: Regulation identifier (e.g. "ma", "i", "g").

        Returns:
            A BuildSpace populated with per-species move/item/ability data.

        Raises:
            ValueError: If no team files are found for the regulation.
            ValueError: If any species has fewer than 4 observed moves (would
                prevent generating a valid 4-move build for that species).
        """
        paths = RandomTeamBuilder.get_team_paths(reg, prefer_featured=False)
        if not paths:
            raise ValueError(f"no team files found for regulation '{reg}'")

        moves_by_species: dict[str, set[str]] = {}
        items_by_species: dict[str, set[str]] = {}
        abilities_by_species: dict[str, set[str]] = {}
        observed_natures: set[str] = set()
        observed_tera_types: set[str] = set()

        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            for block in blocks:
                _parse_block(
                    block,
                    moves_by_species,
                    items_by_species,
                    abilities_by_species,
                    observed_natures,
                    observed_tera_types,
                )

        data: dict[str, _SpeciesData] = {}
        for species in sorted(moves_by_species):
            moves = sorted(moves_by_species[species])
            if len(moves) < 4:
                raise ValueError(
                    f"species '{species}' has only {len(moves)} observed move(s); "
                    "need at least 4 to generate a valid build"
                )
            data[species] = _SpeciesData(
                moves=moves,
                items=sorted(items_by_species.get(species, set())),
                abilities=sorted(abilities_by_species.get(species, set())),
            )

        # Fall back to hardcoded lists if the corpus is sparse
        natures = sorted(observed_natures) if observed_natures else _ALL_NATURES
        tera_types = sorted(observed_tera_types) if observed_tera_types else _ALL_TERA_TYPES

        return cls(_data=data, natures=natures, tera_types=tera_types)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def species_list(self) -> list[str]:
        """Sorted list of all species in the search space."""
        return list(self._data.keys())

    def moves_for(self, species: str) -> list[str]:
        return self._data[species].moves

    def items_for(self, species: str) -> list[str]:
        return self._data[species].items

    def abilities_for(self, species: str) -> list[str]:
        return self._data[species].abilities

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def random_build(
        self,
        species: str | None = None,
        rng: random.Random | None = None,
        forbidden_items: set[str] | None = None,
    ) -> PokemonBuild:
        """
        Generate a fresh PokemonBuild from scratch.

        All fields are independently sampled from the valid space for the
        given species. EVs are randomly distributed (not copied from any
        tournament build). IVs are all 31 for Phase 1.

        Args:
            species:         Fix the species. Randomly chosen when None.
            rng:             Seeded random instance for reproducibility.
            forbidden_items: Items already in use by other team members.
                             The sampled item will not be in this set if
                             alternatives exist; otherwise falls back freely.

        Returns:
            A new PokemonBuild with win_rate not yet assigned.
        """
        _rng = rng or random
        sp = species if species is not None else _rng.choice(self.species_list)
        data = self._data[sp]

        forbidden = forbidden_items or set()
        available_items = [i for i in data.items if i not in forbidden]
        # Fall back to unrestricted pool if item clause cannot be satisfied
        item_pool = available_items if available_items else data.items
        item = _rng.choice(item_pool) if item_pool else "None"
        ability = _rng.choice(data.abilities) if data.abilities else "No Ability"

        return PokemonBuild(
            species=sp,
            item=item,
            ability=ability,
            tera_type=_rng.choice(self.tera_types),
            nature=_rng.choice(self.natures),
            evs=self.random_evs(_rng),
            ivs=(31, 31, 31, 31, 31, 31),
            moves=tuple(_rng.sample(data.moves, 4)),  # type: ignore[arg-type]
        )

    def random_evs(
        self,
        rng: random.Random | None = None,
    ) -> tuple[int, int, int, int, int, int]:
        """
        Generate a random, valid EV spread.

        Works in compact stat-point units. Randomly allocates a total of
        1–66 points across the 6 stats, each capped at 32. The result always
        satisfies:
            - each value in 0–32
            - sum in 1–66 (never zero, required by Showdown)

        Args:
            rng: Seeded random instance for reproducibility.

        Returns:
            Tuple of 6 EV values in stat order (HP, Atk, Def, SpA, SpD, Spe).
        """
        _rng = rng or random
        units = [0] * 6
        remaining = _rng.randint(1, _MAX_UNITS_TOTAL)
        indices = list(range(6))
        _rng.shuffle(indices)
        for idx in indices[:-1]:
            can_give = min(_MAX_UNITS_PER_STAT, remaining)
            units[idx] = _rng.randint(0, can_give)
            remaining -= units[idx]
            if remaining == 0:
                break
        # Give whatever is left to the last shuffled index, capped at max
        units[indices[-1]] = min(_MAX_UNITS_PER_STAT, remaining)
        return tuple(units)  # type: ignore[return-value]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _species_name(first_line: str) -> str:
    """Extract the species name from a build block's first line.

    Normalizes Mega-form entries (e.g. "Dragonite-Mega") to their base species
    ("Dragonite") so both map to the same BuildSpace entry and never trigger a
    Species Clause violation.  Regional variants (-Hisui, -Alola, etc.) are kept
    as-is because Showdown treats them as distinct Pokémon.
    """
    name = first_line.strip()
    if " @ " in name:
        name = name.split(" @ ")[0]
    if " (" in name:
        name = name.split(" (")[0]
    name = name.strip()
    if name.endswith("-Mega"):
        name = name[:-5]
    return name


def _item_name(first_line: str) -> str | None:
    """Extract the held item from a build block's first line, or None."""
    if " @ " in first_line:
        return first_line.split(" @ ", 1)[1].strip()
    return None


def _parse_block(
    block: str,
    moves_by_species: dict[str, set[str]],
    items_by_species: dict[str, set[str]],
    abilities_by_species: dict[str, set[str]],
    observed_natures: set[str],
    observed_tera_types: set[str],
) -> None:
    """Parse one Showdown build block and update the accumulation dicts."""
    lines = block.splitlines()
    if not lines:
        return

    species = _species_name(lines[0])
    if not species:
        return

    item = _item_name(lines[0])
    if item:
        items_by_species.setdefault(species, set()).add(item)

    for line in lines[1:]:
        line = line.strip()
        if line.startswith("Ability: "):
            abilities_by_species.setdefault(species, set()).add(line[len("Ability: "):])
        elif line.startswith("Tera Type: "):
            observed_tera_types.add(line[len("Tera Type: "):])
        elif line.endswith(" Nature"):
            # Format: "{NatureName} Nature"
            nature = line.split()[0]
            observed_natures.add(nature)
        elif line.startswith("- "):
            moves_by_species.setdefault(species, set()).add(line[2:])
