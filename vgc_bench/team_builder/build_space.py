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
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.team_builder.pokemon_build import (
    EV_MAX_PER_STAT,
    EV_MAX_TOTAL,
    PokemonBuild,
    RESTRICTED_LEGENDARIES,
)

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

# EV allocation constants (standard Gen 9 EVs: 0–252 per stat, total ≤ 510)
_MAX_UNITS_PER_STAT = EV_MAX_PER_STAT   # 252
_MAX_UNITS_TOTAL = EV_MAX_TOTAL          # 510

# Map regulation letter → maximum number of restricted Pokémon allowed per team.
# 0 means "no restriction" (either format has no restricted list, or limit is unconstrained).
# Reg G/H: Series 1/2 — Limit One Restricted.
# Reg I/J: Series 3/4 — Limit Two Restricted.
_REG_RESTRICTED_LIMIT: dict[str, int] = {
    "g": 1, "h": 1, "i": 2, "j": 2,
}

# Stat name → index in the 6-tuple (HP, Atk, Def, SpA, SpD, Spe)
_STAT_INDEX: dict[str, int] = {
    "HP": 0, "Atk": 1, "Def": 2, "SpA": 3, "SpD": 4, "Spe": 5,
}

# Type alias for a (nature, evs) pair from tournament data
_NatEv = tuple[str, tuple[int, int, int, int, int, int]]


@dataclass
class _SpeciesData:
    moves:         list[str]
    items:         list[str]
    abilities:     list[str]
    tera_types:    list[str]
    nat_ev_corpus: list[_NatEv]       # (nature, evs) pairs observed in tournament files
    move_weights:  dict[str, float]   # per-move sampling weight; starts uniform
    item_weights:  dict[str, float]   # per-item sampling weight; starts uniform


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
    species_weights: dict[str, float]  # per-species sampling weight, BC-initialized from tournament frequency
    restricted_species: frozenset[str] = field(default_factory=frozenset)
    restricted_limit: int = 0  # 0 = no restriction; >0 = max restricted Pokémon per team

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_regulation(cls, reg: str = "i") -> BuildSpace:
        """
        Build a BuildSpace from all tournament team files for the given regulation.

        Args:
            reg: Regulation identifier (e.g. "ma", "i", "g").

        Returns:
            A BuildSpace populated with per-species move/item/ability data and
            a (nature, EVs) corpus for BC-style initialization.

        Raises:
            ValueError: If no team files are found for the regulation.
            ValueError: If any species has fewer than 4 observed moves.
        """
        paths = RandomTeamBuilder.get_team_paths(reg, prefer_featured=False)
        if not paths:
            raise ValueError(f"no team files found for regulation '{reg}'")

        moves_by_species: dict[str, Counter[str]] = {}
        items_by_species: dict[str, Counter[str]] = {}
        abilities_by_species: dict[str, set[str]] = {}
        tera_types_by_species: dict[str, set[str]] = {}
        nat_ev_by_species: dict[str, list[_NatEv]] = {}
        observed_natures: set[str] = set()
        observed_tera_types: set[str] = set()
        species_counts: Counter[str] = Counter()

        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
            for block in blocks:
                _parse_block(
                    block,
                    moves_by_species,
                    items_by_species,
                    abilities_by_species,
                    tera_types_by_species,
                    observed_natures,
                    observed_tera_types,
                    nat_ev_by_species,
                    species_counts,
                )

        # Validate corpus abilities against poke-env GenData base-form ability lists.
        # • Removes mega-only abilities that leaked into base-species ability sets
        #   (e.g. "Shadow Tag" on Gengar from a corpus entry with Gengar @ Gengarite).
        # • Fills in abilities for species whose only corpus entries were "-Mega" form
        #   blocks (e.g. Floette) — those entries were skipped by is_mega_entry, so
        #   the ability set is empty; GenData provides the correct base-form fallback.
        import re as _re
        from poke_env.data import GenData
        _gd = GenData.from_gen(9)
        for species in moves_by_species:
            dex_key = _re.sub(r"[^a-z0-9]", "", species.lower())
            entry = _gd.pokedex.get(dex_key, {})
            gendata_abilities = set(entry.get("abilities", {}).values())
            if not gendata_abilities:
                continue
            corpus_abilities = abilities_by_species.get(species, set())
            if corpus_abilities:
                # Filter: keep only abilities valid for the base form
                filtered = corpus_abilities & gendata_abilities
                if filtered:
                    abilities_by_species[species] = filtered
            else:
                # No corpus abilities (all entries were mega-form) — use GenData
                abilities_by_species[species] = gendata_abilities

        data: dict[str, _SpeciesData] = {}
        for species in sorted(moves_by_species):
            move_counts = moves_by_species[species]
            moves = sorted(move_counts)
            if len(moves) < 4:
                import warnings
                warnings.warn(
                    f"species '{species}' has only {len(moves)} observed move(s); "
                    "skipping (need at least 4 to generate a valid build)",
                    stacklevel=2,
                )
                continue
            item_counts = items_by_species.get(species, Counter())
            items = sorted(item_counts)
            move_weights = {m: float(move_counts[m]) for m in moves}
            _normalize(move_weights)
            item_weights = {i: float(item_counts[i]) for i in items}
            _normalize(item_weights)
            data[species] = _SpeciesData(
                moves=moves,
                items=items,
                abilities=sorted(abilities_by_species.get(species, set())),
                tera_types=sorted(tera_types_by_species.get(species, set())),
                nat_ev_corpus=nat_ev_by_species.get(species, []),
                move_weights=move_weights,
                item_weights=item_weights,
            )

        species_weights = {sp: float(species_counts.get(sp, 1)) for sp in sorted(data)}
        _normalize(species_weights)

        # Fall back to hardcoded lists if the corpus is sparse
        natures = sorted(observed_natures) if observed_natures else _ALL_NATURES
        tera_types = sorted(observed_tera_types) if observed_tera_types else _ALL_TERA_TYPES

        # Restricted Pokémon: intersection of corpus species with the canonical list.
        # restricted_limit = 0 means "no restriction" (0 is the default for unknown regs).
        restricted_species = frozenset(sp for sp in data if sp in RESTRICTED_LEGENDARIES)
        restricted_limit = _REG_RESTRICTED_LIMIT.get(reg.lower(), 0)

        return cls(
            _data=data,
            natures=natures,
            tera_types=tera_types,
            species_weights=species_weights,
            restricted_species=restricted_species,
            restricted_limit=restricted_limit,
        )

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

    def tera_types_for(self, species: str) -> list[str]:
        return self._data[species].tera_types

    def nat_ev_corpus_for(self, species: str) -> list[_NatEv]:
        """Return the list of (nature, evs) pairs observed for this species."""
        return self._data[species].nat_ev_corpus

    def species_order(self, rng: random.Random | None = None) -> list[str]:
        """Return all species in a weighted random order for team assembly.

        Higher-weight (more common tournament) species tend to appear earlier,
        but all species are reachable. Used by random_team() and swap_member()
        in place of a uniform shuffle.
        """
        _rng = rng or random
        return _sample_weighted_no_replace(
            _rng,
            self.species_list,
            [self.species_weights[s] for s in self.species_list],
            len(self.species_list),
        )

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

        Nature and EVs are sampled jointly using a 3-way mixture:
          - 60%: BC corpus — draw a (nature, evs) pair observed for this species
          - 25%: concentrated — random nature + max 1-2 stats, small remainder
          - 15%: uniform — random nature + uniform random EVs

        Args:
            species:         Fix the species. Randomly chosen when None.
            rng:             Seeded random instance for reproducibility.
            forbidden_items: Items already in use by other team members.

        Returns:
            A new PokemonBuild with IVs all 31.
        """
        _rng = rng or random
        sp = species if species is not None else _rng.choice(self.species_list)
        data = self._data[sp]

        forbidden = forbidden_items or set()
        available_items = [i for i in data.items if i not in forbidden]
        item_pool = available_items if available_items else data.items
        item = _rng.choices(
            list(data.item_weights.keys()),
            weights=list(data.item_weights.values()),
        )[0] if not forbidden else (
            _rng.choice(item_pool) if item_pool else "None"
        )
        # Re-select if weighted item is forbidden (fall back to available pool)
        if item in forbidden and item_pool:
            item = _rng.choice(item_pool)

        ability = _rng.choice(data.abilities) if data.abilities else "No Ability"

        # Joint (nature, evs) sampling
        corpus = data.nat_ev_corpus
        mode = _rng.random()
        if corpus and mode < 0.60:
            nature, evs = _rng.choice(corpus)
        elif mode < 0.85:
            nature = _rng.choice(self.natures)
            evs = self._concentrated_evs(_rng)
        else:
            nature = _rng.choice(self.natures)
            evs = self.random_evs(_rng)

        moves_pool = list(data.move_weights.keys())
        moves_weights = list(data.move_weights.values())
        moves = tuple(_sample_weighted_no_replace(_rng, moves_pool, moves_weights, 4))

        # Sample tera type: prefer species-observed types, fall back to global list.
        # Gives None only for formats without Tera (i.e. both lists are empty).
        tera_options = data.tera_types if data.tera_types else self.tera_types
        tera_type = _rng.choice(tera_options) if tera_options else None

        return PokemonBuild(
            species=sp,
            item=item,
            ability=ability,
            nature=nature,
            evs=evs,
            ivs=(31, 31, 31, 31, 31, 31),
            moves=moves,  # type: ignore[arg-type]
            tera_type=tera_type,
        )

    def random_evs(
        self,
        rng: random.Random | None = None,
    ) -> tuple[int, int, int, int, int, int]:
        """
        Generate a random, valid EV spread (uniform mode).

        Works in standard EV units. Allocates a total of 1–510 points
        across the 6 stats uniformly at random, each capped at 252.

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
        units[indices[-1]] = min(_MAX_UNITS_PER_STAT, remaining)
        return tuple(units)  # type: ignore[return-value]

    def update_from_elites(self, elites: Any, lr: float = 0.0) -> None:
        """
        Nudge per-species sampling weights toward choices in elite teams.

        No-op when lr=0 (Phase 1 default). Phase 2 will enable online updates
        by setting SearchConfig.learning_rate > 0.

        Args:
            elites: list[CandidateTeam] — top-performing teams from the current round.
            lr:     Learning rate; 0.0 disables updates.
        """
        if lr <= 0.0:
            return
        for team in elites:
            for member in team.members:
                data = self._data.get(member.species)
                if data is None:
                    continue
                for move in member.moves:
                    if move in data.move_weights:
                        data.move_weights[move] *= 1.0 + lr
                if member.item in data.item_weights:
                    data.item_weights[member.item] *= 1.0 + lr
                _normalize(data.move_weights)
                _normalize(data.item_weights)
                if member.species in self.species_weights:
                    self.species_weights[member.species] *= 1.0 + lr
        _normalize(self.species_weights)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _concentrated_evs(
        self,
        rng: random.Random,
    ) -> tuple[int, int, int, int, int, int]:
        """Generate a concentrated EV spread: max 1-2 stats, small remainder in a third."""
        units = [0] * 6
        n_focus = rng.randint(1, 2)
        focus = rng.sample(range(6), n_focus)
        remaining = _MAX_UNITS_TOTAL
        for idx in focus:
            give = min(_MAX_UNITS_PER_STAT, remaining)
            units[idx] = give
            remaining -= give
            if remaining == 0:
                break
        if remaining > 0:
            others = [i for i in range(6) if i not in focus]
            if others:
                third = rng.choice(others)
                units[third] = min(_MAX_UNITS_PER_STAT, remaining)
        return tuple(units)  # type: ignore[return-value]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _normalize(weights: dict[str, float]) -> None:
    """Normalize a weight dict in-place so values sum to len(weights) (keep mean=1)."""
    total = sum(weights.values())
    if total <= 0:
        for k in weights:
            weights[k] = 1.0
        return
    scale = len(weights) / total
    for k in weights:
        weights[k] *= scale


def _sample_weighted_no_replace(
    rng: random.Random,
    pool: list[str],
    weights: list[float],
    k: int,
) -> list[str]:
    """Sample k distinct items from pool without replacement, respecting weights."""
    if len(pool) < k:
        return list(pool)
    remaining_pool = list(pool)
    remaining_weights = list(weights)
    chosen: list[str] = []
    for _ in range(k):
        pick = rng.choices(remaining_pool, weights=remaining_weights, k=1)[0]
        idx = remaining_pool.index(pick)
        chosen.append(pick)
        remaining_pool.pop(idx)
        remaining_weights.pop(idx)
    return chosen


def _parse_evs(line: str) -> tuple[int, int, int, int, int, int] | None:
    """Parse 'EVs: 21 HP / 32 SpA / ...' into a 6-tuple. Returns None on error."""
    if not line.startswith("EVs: "):
        return None
    evs = [0] * 6
    for part in line[5:].split(" / "):
        tokens = part.strip().split()
        if len(tokens) != 2:
            return None
        try:
            val = int(tokens[0])
        except ValueError:
            return None
        idx = _STAT_INDEX.get(tokens[1])
        if idx is None:
            return None
        evs[idx] = val
    return tuple(evs)  # type: ignore[return-value]


# Mega forms whose base species name is NOT simply "<name>-Mega" → "<name>".
# Floette-Eternal mega-evolves into "Floette-Mega" in Showdown export, but the
# correct team-sheet species is "Floette-Eternal", not "Floette".
_MEGA_TO_BASE: dict[str, str] = {
    "Floette-Mega": "Floette-Eternal",
}

# Non-mega form aliases: consolidate a rarely-used base form into the competitive
# standard form so only one BuildSpace entry exists and team sheets are correct.
# Maushold (3-mouse) → Maushold-Four (4-mouse): only the Four form has Friend Guard
# and is used in competitive play.
_FORM_ALIASES: dict[str, str] = {
    "Maushold": "Maushold-Four",
}


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
        name = _MEGA_TO_BASE.get(name, name[:-5])
    return _FORM_ALIASES.get(name, name)


def _item_name(first_line: str) -> str | None:
    """Extract the held item from a build block's first line, or None."""
    if " @ " in first_line:
        return first_line.split(" @ ", 1)[1].strip()
    return None


def _parse_block(
    block: str,
    moves_by_species: dict[str, Counter[str]],
    items_by_species: dict[str, Counter[str]],
    abilities_by_species: dict[str, set[str]],
    tera_types_by_species: dict[str, set[str]],
    observed_natures: set[str],
    observed_tera_types: set[str],
    nat_ev_by_species: dict[str, list[_NatEv]],
    species_counts: Counter[str],
) -> None:
    """Parse one Showdown build block and update the accumulation dicts."""
    lines = block.splitlines()
    if not lines:
        return

    # Detect mega-form entries before _species_name() strips the suffix.
    # Mega abilities (e.g. Pixilate on Gardevoir-Mega) must not be stored
    # under the base species — Showdown expects the base-form ability on the
    # team sheet and applies the mega ability automatically upon evolution.
    raw_name = lines[0].strip().split(" @ ")[0].split(" (")[0].strip()
    is_mega_entry = raw_name.endswith("-Mega")

    species = _species_name(lines[0])
    if not species:
        return

    species_counts[species] += 1

    item = _item_name(lines[0])
    if item:
        items_by_species.setdefault(species, Counter())[item] += 1

    block_nature: str | None = None
    block_evs: tuple[int, int, int, int, int, int] | None = None

    for line in lines[1:]:
        line = line.strip()
        if line.startswith("Ability: "):
            if not is_mega_entry:
                abilities_by_species.setdefault(species, set()).add(line[len("Ability: "):])
        elif line.startswith("Tera Type: "):
            tera = line[len("Tera Type: "):]
            tera_types_by_species.setdefault(species, set()).add(tera)
            observed_tera_types.add(tera)
        elif line.endswith(" Nature"):
            block_nature = line.split()[0]
            observed_natures.add(block_nature)
        elif line.startswith("EVs: "):
            block_evs = _parse_evs(line)
        elif line.startswith("- "):
            moves_by_species.setdefault(species, Counter())[line[2:]] += 1

    if block_nature and block_evs is not None:
        nat_ev_by_species.setdefault(species, []).append((block_nature, block_evs))
