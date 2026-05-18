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
from vgc_bench.team_builder.pokemon_build import PokemonBuild

TEAM_SIZE: int = 6

# Weights for the mutate_build operator: probability of mutating each field.
# Moves are weighted highest (most options, biggest impact on team synergy).
_MUTATION_FIELDS: tuple[str, ...] = ("move", "evs", "item", "nature", "tera_type")
_MUTATION_WEIGHTS: tuple[float, ...] = (0.40, 0.25, 0.15, 0.10, 0.10)


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
    available = list(space.species_list)
    _rng.shuffle(available)

    for species in available:
        if len(members) == TEAM_SIZE:
            break
        if species not in used_species:
            build = space.random_build(species=species, rng=_rng, forbidden_items=used_items)
            if build.item in used_items:
                # Species has only one item in corpus and it's already used; skip
                continue
            members.append(build)
            used_species.add(species)
            used_items.add(build.item)

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

    for slot in swap_slots:
        evicted_species = members[slot].species
        current_species.discard(evicted_species)

        species_candidates = [s for s in space.species_list if s not in current_species]
        if not species_candidates:
            raise ValueError(
                "build space too small to find a replacement respecting the species clause"
            )
        _rng.shuffle(species_candidates)
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
        # Pick a slot to replace
        slot = _rng.randrange(4)
        current_move = build.moves[slot]
        alternatives = [m for m in species_moves if m not in current_set]
        if not alternatives:
            return build  # no alternative available
        new_moves = list(build.moves)
        new_moves[slot] = _rng.choice(alternatives)
        return build.replace(moves=tuple(new_moves))  # type: ignore[arg-type]

    elif field_choice == "evs":
        return build.replace(evs=space.random_evs(_rng))

    elif field_choice == "item":
        forbidden = (forbidden_items or set()) | {build.item}
        alternatives = [i for i in space.items_for(build.species) if i not in forbidden]
        if not alternatives:
            # Relax item-clause constraint as a last resort
            alternatives = [i for i in space.items_for(build.species) if i != build.item]
        if not alternatives:
            return build
        return build.replace(item=_rng.choice(alternatives))

    elif field_choice == "nature":
        alternatives = [n for n in space.natures if n != build.nature]
        if not alternatives:
            return build
        return build.replace(nature=_rng.choice(alternatives))

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

    for a, b in zip(team_a.members, team_b.members):
        chosen, fallback = (a, b) if _rng.random() < 0.5 else (b, a)
        placed = False
        # Try each parent candidate; accept if no species or item conflict
        for candidate in (chosen, fallback):
            if candidate.species not in used_species and candidate.item not in used_items:
                result.append(candidate)
                used_species.add(candidate.species)
                used_items.add(candidate.item)
                placed = True
                break
        if not placed:
            # Try parents again, accepting species conflict (re-roll item)
            for candidate in (chosen, fallback):
                if candidate.species not in used_species:
                    new_build = space.random_build(
                        species=candidate.species, rng=_rng, forbidden_items=used_items
                    )
                    if new_build.item not in used_items:
                        result.append(new_build)
                        used_species.add(new_build.species)
                        used_items.add(new_build.item)
                        placed = True
                        break
        if not placed:
            # Both species conflict or item still conflicts; scan fresh species
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

    return CandidateTeam(members=tuple(result))
