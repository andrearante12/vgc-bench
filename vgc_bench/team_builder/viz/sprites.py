"""
Map a species name to a Pokémon Showdown sprite URL and to its type colors.

Sprites are served from the Showdown CDN (external network). We derive Showdown's
form-aware "sprite id" from poke-env's bundled gen9 pokedex (``baseSpecies`` +
``forme``) so forms such as Calyrex-Shadow and Urshifu-Rapid-Strike resolve to an image
without a hand-maintained override table. The dashboard renders the URL in an ``<img>``
whose ``onerror`` falls back to the species name, so an offline/404 case degrades
gracefully to text.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

_SPRITE_BASE = "https://play.pokemonshowdown.com/sprites/gen5"

# Type -> accent color, matching Showdown's type palette closely enough for card bars.
TYPE_COLORS: dict[str, str] = {
    "Normal": "#9099a1",
    "Fire": "#ff9d55",
    "Water": "#4d90d5",
    "Electric": "#f4d23c",
    "Grass": "#63bc5a",
    "Ice": "#73cec0",
    "Fighting": "#ce4069",
    "Poison": "#ab6ac8",
    "Ground": "#d97746",
    "Flying": "#8fa8dd",
    "Psychic": "#fa7179",
    "Bug": "#90c12c",
    "Rock": "#c7b78b",
    "Ghost": "#5269ad",
    "Dragon": "#0b6dc3",
    "Dark": "#5a5366",
    "Steel": "#5a8ea1",
    "Fairy": "#ec8fe6",
}
_DEFAULT_COLOR = "#9099a1"


def _to_id(name: str) -> str:
    """Showdown's toID: lowercase and strip everything but a-z0-9."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


@functools.lru_cache(maxsize=1)
def _pokedex() -> dict:
    """Load poke-env's bundled gen9 pokedex (keyed by toID). Empty dict if missing."""
    try:
        import poke_env  # noqa: F401

        root = Path(poke_env.__file__).parent / "data" / "static" / "pokedex"
        path = root / "gen9pokedex.json"
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sprite_id(species: str) -> str:
    """
    Compute Showdown's sprite id for a species.

    For a form (has baseSpecies + forme in the dex) the id is
    ``toID(baseSpecies)-toID(forme)``; otherwise it is ``toID(name)``. Falls back to
    ``toID(species)`` when the species is not in the dex.
    """
    entry = _pokedex().get(_to_id(species))
    if entry:
        base = entry.get("baseSpecies")
        forme = entry.get("forme")
        if base and forme:
            return f"{_to_id(base)}-{_to_id(forme)}"
        return _to_id(entry.get("name", species))
    return _to_id(species)


def sprite_url(species: str) -> str:
    """Return a Showdown gen5 sprite PNG URL for the given species."""
    return f"{_SPRITE_BASE}/{_sprite_id(species)}.png"


def species_types(species: str) -> list[str]:
    """Return the species' types (e.g. ['Psychic', 'Ghost']); [] if unknown."""
    entry = _pokedex().get(_to_id(species))
    return list(entry.get("types", [])) if entry else []


def type_color(species: str) -> str:
    """Return an accent color for the species' primary type."""
    types = species_types(species)
    return TYPE_COLORS.get(types[0], _DEFAULT_COLOR) if types else _DEFAULT_COLOR
