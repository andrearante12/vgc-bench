"""
Parse the Showdown-paste snapshots the team builder writes into a display model.

The snapshot / teams files (``snapshots/step_*.txt``, ``round_*.txt``,
``teams_iter*.txt``) are exactly the output of ``CandidateTeam.to_showdown_text()``
(see ``pokemon_build.py:to_showdown_text``), optionally prefixed by a ``# ...`` header
line. We parse them ourselves rather than via poke-env so we control precisely which
fields the dashboard shows and the species id used to look up a sprite.

Note the two lines the writer omits: the ``EVs:`` line when all EVs are zero, and the
``IVs:`` line when all IVs are 31 — the parser therefore treats both as optional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Stat order used in Showdown "EVs:"/"IVs:" lines.
_STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")

_HEADER_RE = re.compile(r"win_rate=([0-9.]+)")
_STEP_RE = re.compile(r"\bstep=(\d+)")
_ROUND_RE = re.compile(r"\bround=(\d+)")
_TEAM_RE = re.compile(r"#\s*Team\s+(\d+)")


@dataclass(frozen=True)
class PokemonView:
    """A single Pokémon set, flattened for display."""

    species: str
    item: str | None
    ability: str | None
    nature: str | None
    tera: str | None
    evs: dict[str, int] = field(default_factory=dict)
    moves: tuple[str, ...] = ()

    def ev_summary(self) -> str:
        """Compact EV spread, e.g. '252 Atk / 4 Def / 252 Spe' (empty if none)."""
        return " / ".join(f"{v} {s}" for s, v in self.evs.items() if v > 0)


@dataclass(frozen=True)
class TeamView:
    """A parsed team: up to 6 members plus any win_rate read from a header line."""

    members: tuple[PokemonView, ...]
    win_rate: float | None = None


def _parse_stat_line(line: str) -> dict[str, int]:
    """Parse 'EVs: 4 HP / 252 Atk / 252 Spe' -> {'HP': 4, 'Atk': 252, 'Spe': 252}."""
    out: dict[str, int] = {}
    body = line.split(":", 1)[1] if ":" in line else line
    for part in body.split("/"):
        part = part.strip()
        if not part:
            continue
        num, _, stat = part.partition(" ")
        stat = stat.strip()
        if num.isdigit() and stat in _STATS:
            out[stat] = int(num)
    return out


def parse_pokemon(block: str) -> PokemonView | None:
    """Parse one Pokémon text block (lines separated by newlines)."""
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return None

    # First line: "Species @ Item"  (item optional).
    head = lines[0]
    if "@" in head:
        species, _, item = head.partition("@")
        species, item = species.strip(), item.strip() or None
    else:
        species, item = head.strip(), None

    ability = nature = tera = None
    evs: dict[str, int] = {}
    moves: list[str] = []
    for line in lines[1:]:
        if line.startswith("Ability:"):
            ability = line.split(":", 1)[1].strip()
        elif line.startswith("Tera Type:"):
            tera = line.split(":", 1)[1].strip()
        elif line.startswith("EVs:"):
            evs = _parse_stat_line(line)
        elif line.startswith("IVs:"):
            continue  # not surfaced in the UI
        elif line.startswith("Level:"):
            continue
        elif line.endswith(" Nature"):
            nature = line[: -len(" Nature")].strip()
        elif line.startswith("- "):
            moves.append(line[2:].strip())

    return PokemonView(
        species=species,
        item=item,
        ability=ability,
        nature=nature,
        tera=tera,
        evs=evs,
        moves=tuple(moves),
    )


def parse_team(text: str) -> TeamView:
    """
    Parse a single team's Showdown paste (optionally with a leading ``# ...`` header)
    into a TeamView. Members are separated by blank lines.
    """
    win_rate: float | None = None
    body_lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            m = _HEADER_RE.search(line)
            if m:
                win_rate = float(m.group(1))
            continue
        body_lines.append(line)

    # Split into member blocks on blank-line boundaries.
    blocks: list[str] = []
    current: list[str] = []
    for line in body_lines:
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))

    members = tuple(p for p in (parse_pokemon(b) for b in blocks) if p is not None)
    return TeamView(members=members, win_rate=win_rate)


def parse_teams_file(text: str) -> list[TeamView]:
    """Parse a ``teams_iter*.txt`` file (multiple teams separated by ``---``)."""
    chunks = re.split(r"^\s*---\s*$", text, flags=re.MULTILINE)
    teams = [parse_team(c) for c in chunks if c.strip()]
    return [t for t in teams if t.members]


def parse_header(text: str) -> dict:
    """
    Extract metadata from a snapshot's leading ``# ...`` header line:
    step / round index and win_rate (any that are present).
    """
    out: dict = {}
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            break
        if m := _STEP_RE.search(line):
            out["step"] = int(m.group(1))
        if m := _ROUND_RE.search(line):
            out["round"] = int(m.group(1))
        if m := _HEADER_RE.search(line):
            out["win_rate"] = float(m.group(1))
        if m := _TEAM_RE.search(line):
            out["team"] = int(m.group(1))
    return out
