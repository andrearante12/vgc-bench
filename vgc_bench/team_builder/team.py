"""
Team representation and operators for the VGC-Bench team builder.

A CandidateTeam is an ordered collection of 6 PokemonBuild instances assembled
from scratch using a BuildSpace. All operators produce new CandidateTeam
instances without modifying their inputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from poke_env.teambuilder import Teambuilder

from vgc_bench.src.teams import calc_team_similarity_score
from vgc_bench.team_builder.build_space import BuildSpace
from vgc_bench.team_builder.pokemon_build import EV_MAX_PER_STAT, PokemonBuild

TEAM_SIZE: int = 6

# Per-stat EV cap — mirrors EV_MAX_PER_STAT from pokemon_build (standard Gen 9: 252)
_EV_MAX: int = EV_MAX_PER_STAT

# Weights for the mutate_build operator: probability of mutating each field.
# - move:         swap one move for another from the species' pool
# - nat_ev:       re-sample (nature, evs) jointly from the BC corpus
# - evs_transfer: transfer points between two stats (fine-grained EV tuning)
# - item:         swap item
# - tera_type:    swap tera type
_MUTATION_FIELDS: tuple[str, ...] = ("move", "nat_ev", "evs_transfer", "item", "tera_type")
_MUTATION_WEIGHTS: tuple[float, ...] = (0.35, 0.20, 0.15, 0.20, 0.10)


@dataclass(frozen=True)
class CandidateTeam:
    """
    An immutable candidate team of exactly 6 Pokémon builds.

    Attributes:
        members:  Tuple of exactly 6 PokemonBuild instances.
        win_rate: Win rate in [0, 1] after evaluation; None if not yet scored.
    """

    members: tuple[PokemonBuild, ...]
    win_rate: float | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if len(self.members) != TEAM_SIZE:
            raise ValueError(
                f"a team must have exactly {TEAM_SIZE} members, got {len(self.members)}"
            )
        species = [m.species for m in self.members]
        dup_species = [s for s in species if species.count(s) > 1]
        if dup_species:
            raise ValueError(f"duplicate species in team: {list(set(dup_species))}")
        items = [m.item for m in self.members]
        dup_items = [i for i in items if items.count(i) > 1]
        if dup_items:
            raise ValueError(f"item clause violation — duplicate items: {list(set(dup_items))}")

    def with_win_rate(self, win_rate: float) -> CandidateTeam:
        """Return a new CandidateTeam with win_rate set."""
        return CandidateTeam(members=self.members, win_rate=win_rate)

    def to_showdown_text(self) -> str:
        """Assemble all 6 builds into a single Showdown-format team string."""
        return "\n\n".join(m.to_showdown_text() for m in self.members)

    def to_packed_team(self) -> str:
        """Convert to poke-env packed team format."""
        return Teambuilder.join_team(
            Teambuilder.parse_showdown_team(self.to_showdown_text())
        )

    def similarity_to(self, other: CandidateTeam) -> float:
        """Compute similarity score in [0, 1] to another candidate team."""
        return calc_team_similarity_score(
            self.to_showdown_text(), other.to_showdown_text()
        )


def pokemon_build_to_dict(build: PokemonBuild) -> dict:
    return {
        "species": build.species,
        "item": build.item,
        "ability": build.ability,
        "nature": build.nature,
        "evs": list(build.evs),
        "ivs": list(build.ivs),
        "moves": list(build.moves),
        "tera_type": build.tera_type,
    }


def pokemon_build_from_dict(d: dict) -> PokemonBuild:
    return PokemonBuild(
        species=d["species"],
        item=d["item"],
        ability=d["ability"],
        nature=d["nature"],
        evs=tuple(d["evs"]),
        ivs=tuple(d["ivs"]),
        moves=tuple(d["moves"]),
        tera_type=d.get("tera_type"),
    )


def team_to_dict(team: CandidateTeam) -> dict:
    return {
        "win_rate": team.win_rate,
        "members": [pokemon_build_to_dict(m) for m in team.members],
    }


def team_from_dict(d: dict) -> CandidateTeam:
    members = tuple(pokemon_build_from_dict(m) for m in d["members"])
    team = CandidateTeam(members=members)
    return team.with_win_rate(d["win_rate"]) if d.get("win_rate") is not None else team


def random_team(
    space: BuildSpace,
    rng: random.Random | None = None,
) -> CandidateTeam:
    """
    Sample a random CandidateTeam from the build space, respecting species clause.

    Args:
        space: BuildSpace defining the valid search space.
        rng:   Optional seeded random instance for reproducibility.

    Returns:
        A new CandidateTeam with win_rate=None.
    """
    _rng = rng or random
    used_species: set[str] = set()
    used_items: set[str] = set()
    members: list[PokemonBuild] = []
    available = space.species_order(_rng)

    restricted_limit: int = space.restricted_limit
    restricted_species_set: frozenset[str] = space.restricted_species
    restricted_count: int = 0

    for species in available:
        if len(members) == TEAM_SIZE:
            break
        if species in used_species:
            continue
        # Enforce restricted legendary limit (0 = no restriction)
        if (
            restricted_limit > 0
            and species in restricted_species_set
            and restricted_count >= restricted_limit
        ):
            continue
        build = space.random_build(species=species, rng=_rng, forbidden_items=used_items)
        if build.item in used_items:
            # Species has only one item in corpus and it's already used; skip
            continue
        members.append(build)
        used_species.add(species)
        used_items.add(build.item)
        if species in restricted_species_set:
            restricted_count += 1

    if len(members) < TEAM_SIZE:
        raise ValueError(
            f"build space too small to fill {TEAM_SIZE} slots with distinct species and items"
        )
    return CandidateTeam(members=tuple(members))


def swap_member(
    team: CandidateTeam,
    space: BuildSpace,
    n_swaps: int = 1,
    rng: random.Random | None = None,
) -> CandidateTeam:
    """
    Replace n_swaps members of the team with freshly generated builds.

    Replacement builds are drawn from the space with species distinct from all
    retained team members, respecting the species clause.

    Args:
        team:    The team whose members to partially replace.
        space:   BuildSpace for sampling replacements.
        n_swaps: Number of members to replace (default 1).
        rng:     Optional seeded random instance.

    Returns:
        A new CandidateTeam with win_rate=None.
    """
    _rng = rng or random
    members = list(team.members)
    slots = list(range(TEAM_SIZE))
    _rng.shuffle(slots)
    swap_slots = slots[:n_swaps]
    swap_slot_set = set(swap_slots)

    current_species = {m.species for m in members}
    # Items held by slots that are NOT being swapped are fixed constraints
    used_items = {members[i].item for i in range(TEAM_SIZE) if i not in swap_slot_set}

    restricted_limit: int = space.restricted_limit
    restricted_species_set: frozenset[str] = space.restricted_species

    for slot in swap_slots:
        evicted_species = members[slot].species
        current_species.discard(evicted_species)

        # Count restricted Pokémon already committed (everything in current_species
        # after discarding the evicted slot).
        current_restricted = sum(1 for s in current_species if s in restricted_species_set)

        all_ordered = space.species_order(_rng)
        if restricted_limit > 0:
            species_candidates = [
                s for s in all_ordered
                if s not in current_species
                and (
                    s not in restricted_species_set
                    or current_restricted < restricted_limit
                )
            ]
            if not species_candidates:
                # Corpus too small to satisfy restricted limit — relax constraint
                species_candidates = [s for s in all_ordered if s not in current_species]
        else:
            species_candidates = [s for s in all_ordered if s not in current_species]

        if not species_candidates:
            raise ValueError(
                "build space too small to find a replacement respecting the species clause"
            )
        new_build = None
        for candidate_species in species_candidates:
            build = space.random_build(
                species=candidate_species, rng=_rng, forbidden_items=used_items
            )
            if build.item not in used_items:
                new_build = build
                break
        if new_build is None:
            # All candidates have item conflicts; pick any species and ignore item clause
            new_build = space.random_build(
                species=species_candidates[0], rng=_rng
            )
        members[slot] = new_build
        current_species.add(new_build.species)
        used_items.add(new_build.item)

    return CandidateTeam(members=tuple(members))


def mutate_build(
    build: PokemonBuild,
    space: BuildSpace,
    rng: random.Random | None = None,
    forbidden_items: set[str] | None = None,
) -> PokemonBuild:
    """
    Mutate exactly one field of a PokemonBuild, returning a new instance.

    The field to mutate is chosen randomly according to _MUTATION_WEIGHTS.
    The species is never changed — use swap_member to replace a whole slot.

    Args:
        build:           The build to mutate.
        space:           BuildSpace providing valid alternatives for each field.
        rng:             Optional seeded random instance.
        forbidden_items: Items held by other team members (Item Clause). When
                         mutating the item field, the new item will not be in
                         this set if alternatives exist.

    Returns:
        A new PokemonBuild with exactly one field changed, or the same build
        if no valid alternative exists for the chosen field.
    """
    _rng = rng or random
    field_choice = _rng.choices(_MUTATION_FIELDS, weights=_MUTATION_WEIGHTS, k=1)[0]

    if field_choice == "move":
        species_moves = space.moves_for(build.species)
        current_set = set(build.moves)
        slot = _rng.randrange(4)
        alternatives = [m for m in species_moves if m not in current_set]
        if not alternatives:
            return build
        new_moves = list(build.moves)
        new_moves[slot] = _rng.choice(alternatives)
        return build.replace(moves=tuple(new_moves))  # type: ignore[arg-type]

    elif field_choice == "nat_ev":
        # Re-sample (nature, evs) jointly from the BC corpus for this species.
        corpus = space.nat_ev_corpus_for(build.species)
        if corpus:
            new_nature, new_evs = _rng.choice(corpus)
        else:
            new_nature = _rng.choice(space.natures)
            new_evs = space.random_evs(_rng)
        return build.replace(nature=new_nature, evs=new_evs)

    elif field_choice == "evs_transfer":
        # Transfer a random number of EV points from one stat to another.
        current = list(build.evs)
        sources = [i for i in range(6) if current[i] > 0]
        dests = [i for i in range(6) if current[i] < _EV_MAX]
        pairs = [(s, d) for s in sources for d in dests if s != d]
        if not pairs:
            return build.replace(evs=space.random_evs(_rng))
        src, dst = _rng.choice(pairs)
        amount = _rng.randint(1, min(current[src], _EV_MAX - current[dst]))
        current[src] -= amount
        current[dst] += amount
        return build.replace(evs=tuple(current))  # type: ignore[arg-type]

    elif field_choice == "item":
        forbidden = (forbidden_items or set()) | {build.item}
        alternatives = [i for i in space.items_for(build.species) if i not in forbidden]
        if not alternatives:
            alternatives = [i for i in space.items_for(build.species) if i != build.item]
        if not alternatives:
            return build
        return build.replace(item=_rng.choice(alternatives))

    else:  # tera_type
        alternatives = [t for t in space.tera_types if t != build.tera_type]
        if not alternatives:
            return build
        return build.replace(tera_type=_rng.choice(alternatives))


def combine_teams(
    team_a: CandidateTeam,
    team_b: CandidateTeam,
    space: BuildSpace,
    rng: random.Random | None = None,
) -> CandidateTeam:
    """
    Combine two teams slot-by-slot, resolving species conflicts via the space.

    For each of the 6 slots, independently chooses the build from team_a or
    team_b (50/50). If the chosen build would introduce a duplicate species,
    tries the other parent's build, then falls back to a fresh random build
    from the space.

    Args:
        team_a: First parent team.
        team_b: Second parent team.
        space:  BuildSpace for conflict resolution.
        rng:    Optional seeded random instance.

    Returns:
        A new CandidateTeam with win_rate=None.
    """
    _rng = rng or random
    result: list[PokemonBuild] = []
    used_species: set[str] = set()
    used_items: set[str] = set()

    restricted_limit: int = space.restricted_limit
    restricted_species_set: frozenset[str] = space.restricted_species
    restricted_count: int = 0

    def _restricted_ok(species: str) -> bool:
        """Return True if adding this species respects the restricted limit."""
        if restricted_limit <= 0 or species not in restricted_species_set:
            return True
        return restricted_count < restricted_limit

    for a, b in zip(team_a.members, team_b.members):
        chosen, fallback = (a, b) if _rng.random() < 0.5 else (b, a)
        placed = False
        # Try each parent candidate; accept if no species or item conflict
        for candidate in (chosen, fallback):
            if (
                candidate.species not in used_species
                and candidate.item not in used_items
                and _restricted_ok(candidate.species)
            ):
                result.append(candidate)
                used_species.add(candidate.species)
                used_items.add(candidate.item)
                if candidate.species in restricted_species_set:
                    restricted_count += 1
                placed = True
                break
        if not placed:
            # Try parents again, accepting item conflict (re-roll item only)
            for candidate in (chosen, fallback):
                if candidate.species not in used_species and _restricted_ok(candidate.species):
                    new_build = space.random_build(
                        species=candidate.species, rng=_rng, forbidden_items=used_items
                    )
                    if new_build.item not in used_items:
                        result.append(new_build)
                        used_species.add(new_build.species)
                        used_items.add(new_build.item)
                        if new_build.species in restricted_species_set:
                            restricted_count += 1
                        placed = True
                        break
        if not placed:
            # Both parent species conflict or restricted limit reached; scan fresh species
            options = [
                s for s in space.species_list
                if s not in used_species and _restricted_ok(s)
            ]
            if not options:
                # Relax restricted limit as a last resort (shouldn't happen normally)
                options = [s for s in space.species_list if s not in used_species]
            if not options:
                raise ValueError(
                    "build space too small to resolve species conflict in combine_teams"
                )
            new_build = None
            for new_species in _rng.sample(options, len(options)):
                build = space.random_build(
                    species=new_species, rng=_rng, forbidden_items=used_items
                )
                if build.item not in used_items:
                    new_build = build
                    break
            if new_build is None:
                # All options have item conflicts; pick first and relax item clause
                new_build = space.random_build(species=options[0], rng=_rng)
            result.append(new_build)
            used_species.add(new_build.species)
            used_items.add(new_build.item)
            if new_build.species in restricted_species_set:
                restricted_count += 1

    return CandidateTeam(members=tuple(result))
