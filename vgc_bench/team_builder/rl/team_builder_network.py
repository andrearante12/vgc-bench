"""
Neural network team builder for SP-PSRO.

TeamBuilderNetwork takes the current population of teams (+ Nash weights) as
context and generates a 6-Pokémon counter-team via two-phase autoregressive
decoding:

  Phase 1 — Species: pick 6 species one at a time (masked for Species Clause).
  Phase 2 — Builds: for each slot in parallel, pick item, 4 moves, and nature
             (masked for Item Clause and species move pool).

EVs are sampled from the BuildSpace BC corpus conditioned on the chosen nature;
tera type is fixed to the first available (not optimized for Reg MA).

Per-Pokémon token (291-dim):
  species_embed(32) | move_embed×4(128) | item_embed(32) | ability_embed(32)
  | base_types(36) | base_stats(6) | evs(6) | nature_onehot(25) = 291

Shared move/item/ability embeddings are loaded from a PPO checkpoint and frozen
by default (--policy-checkpoint flag in train.py). If no checkpoint is provided
they are initialized randomly and left trainable.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from vgc_bench.src.utils import abilities as ABILITY_VOCAB, items as ITEM_VOCAB, moves as MOVE_VOCAB
from vgc_bench.team_builder.pokemon_build import PokemonBuild
from vgc_bench.team_builder.team import TEAM_SIZE, CandidateTeam

if TYPE_CHECKING:
    from vgc_bench.team_builder.build_space import BuildSpace


_ALL_NATURES: list[str] = [
    "Adamant", "Bashful", "Bold", "Brave", "Calm",
    "Careful", "Docile", "Gentle", "Hardy", "Hasty",
    "Impish", "Jolly", "Lax", "Lonely", "Mild",
    "Modest", "Naive", "Naughty", "Quiet", "Quirky",
    "Rash", "Relaxed", "Sassy", "Serious", "Timid",
]

_TYPE_NAMES: list[str] = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting",
    "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost",
    "Dragon", "Dark", "Steel", "Fairy",
]

# Dimension breakdown for per-Pokémon tokens
_EMBED_DIM = 32
_TOKEN_DIM = (
    _EMBED_DIM      # species
    + 4 * _EMBED_DIM  # 4 moves
    + _EMBED_DIM    # item
    + _EMBED_DIM    # ability
    + 2 * 18        # base types (2 × 18 one-hot)
    + 6             # base stats
    + 6             # EVs
    + 25            # nature one-hot
)  # = 297


def _to_id(name: str) -> str:
    """Normalize a Showdown name to the poke-env vocab ID format."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _parse_showdown_team(text: str) -> list[dict]:
    """
    Parse Pokémon Showdown team text into a list of per-Pokémon dicts.

    Each dict has keys: species, moves (list[str] len 4), item, ability,
    evs (tuple[int,...]  len 6), nature.
    """
    stat_idx = {"HP": 0, "Atk": 1, "Def": 2, "SpA": 3, "SpD": 4, "Spe": 5}
    result: list[dict] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if not lines:
            continue

        first = lines[0].strip()
        if " @ " in first:
            raw_species, item = first.split(" @ ", 1)
            item = item.strip()
        else:
            raw_species, item = first, ""

        # Strip gender indicator and -Mega suffix
        if " (" in raw_species:
            raw_species = raw_species.split(" (")[0]
        raw_species = raw_species.strip()
        if raw_species.endswith("-Mega"):
            raw_species = raw_species[:-5]
        species = raw_species

        ability = "No Ability"
        nature = "Hardy"
        evs_list = [0] * 6
        moves: list[str] = []

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("Ability: "):
                ability = line[len("Ability: "):]
            elif line.endswith(" Nature"):
                nature = line.split()[0]
            elif line.startswith("EVs: "):
                for part in line[5:].split(" / "):
                    toks = part.strip().split()
                    if len(toks) == 2:
                        try:
                            val = int(toks[0])
                            idx = stat_idx.get(toks[1])
                            if idx is not None:
                                evs_list[idx] = val
                        except ValueError:
                            pass
            elif line.startswith("- "):
                moves.append(line[2:].strip())

        moves = (moves + ["Protect"] * 4)[:4]
        result.append(
            {
                "species": species,
                "moves": moves,
                "item": item,
                "ability": ability,
                "evs": tuple(evs_list),
                "nature": nature,
            }
        )
        if len(result) == TEAM_SIZE:
            break
    return result


class TeamBuilderNetwork(nn.Module):
    """
    Opponent-conditioned team generator.

    Encodes the current population (K teams with Nash weights) into a single
    context vector, then autoregressively generates a 6-Pokémon counter-team.
    """

    D_MODEL: int = 256

    def __init__(self, space: BuildSpace) -> None:
        super().__init__()
        self.space = space
        self.species_list: list[str] = space.species_list
        self.n_species: int = len(self.species_list)
        self.species_to_idx: dict[str, int] = {
            sp: i for i, sp in enumerate(self.species_list)
        }

        # Poke-env vocab lists
        self.all_moves: list[str] = MOVE_VOCAB
        self.all_items: list[str] = ITEM_VOCAB
        self.all_abilities: list[str] = ABILITY_VOCAB
        self.n_moves: int = len(self.all_moves)
        self.n_items: int = len(self.all_items)
        self.n_abilities: int = len(self.all_abilities)

        self.move_to_idx: dict[str, int] = {m: i for i, m in enumerate(self.all_moves)}
        self.item_to_idx: dict[str, int] = {it: i for i, it in enumerate(self.all_items)}
        self.ability_to_idx: dict[str, int] = {ab: i for i, ab in enumerate(self.all_abilities)}

        # Reverse: poke-env ID → Showdown name (built from space data)
        self._move_id_to_name: dict[str, str] = {}
        self._item_id_to_name: dict[str, str] = {}
        self._ability_id_to_name: dict[str, str] = {}
        for sp in self.species_list:
            for m in space.moves_for(sp):
                self._move_id_to_name.setdefault(_to_id(m), m)
            for it in space.items_for(sp):
                self._item_id_to_name.setdefault(_to_id(it), it)
            for ab in space.abilities_for(sp):
                self._ability_id_to_name.setdefault(_to_id(ab), ab)

        # Precompute valid vocab indices per species (for generation-time masking)
        self._species_valid_move_idxs: dict[str, list[int]] = {}
        self._species_valid_item_idxs: dict[str, list[int]] = {}
        for sp in self.species_list:
            self._species_valid_move_idxs[sp] = [
                self.move_to_idx[_to_id(m)]
                for m in space.moves_for(sp)
                if _to_id(m) in self.move_to_idx
            ]
            self._species_valid_item_idxs[sp] = [
                self.item_to_idx[_to_id(it)]
                for it in space.items_for(sp)
                if _to_id(it) in self.item_to_idx
            ]

        # Mega stone constraint: IDs ending in "ite"/"itex"/"itey" (excludes eviolite)
        self._mega_stone_idxs: frozenset[int] = frozenset(
            idx for id_, idx in self.item_to_idx.items()
            if id_.endswith(("ite", "itex", "itey")) and id_ != "eviolite"
        )
        self._species_mega_idxs: dict[str, list[int]] = {
            sp: [i for i in self._species_valid_item_idxs.get(sp, []) if i in self._mega_stone_idxs]
            for sp in self.species_list
        }
        self._mega_capable_sp_idxs: frozenset[int] = frozenset(
            i for i, sp in enumerate(self.species_list) if self._species_mega_idxs.get(sp)
        )
        # Restricted-legendary indices and per-team cap (e.g. "Limit Two
        # Restricted" in Reg I). Used to mask species sampling once the cap is hit
        # so generated teams aren't rejected by Showdown.
        self._restricted_sp_idxs: frozenset[int] = frozenset(
            i for i, sp in enumerate(self.species_list)
            if sp in self.space.restricted_species
        )
        self._restricted_limit: int = self.space.restricted_limit

        # Load GenData once for static type/stat lookups and Species Clause grouping
        from poke_env.data import GenData
        self._gd = GenData.from_gen(9)

        # For each species index: the set of other species indices that must be
        # blocked when this species is chosen (Species Clause groups).
        # Built automatically from GenData: species sharing the same baseSpecies
        # (case-insensitive) are treated as the same Pokémon by Showdown's Species
        # Clause (e.g. Rotom forms, Basculegion-F, Tauros-Paldea forms).
        self._clause_blocked_by: dict[int, frozenset[int]] = {}
        _group_to_idxs: dict[str, list[int]] = {}
        for i, sp in enumerate(self.species_list):
            dex_key = _to_id(sp)
            entry = self._gd.pokedex.get(dex_key, {})
            base = entry.get("baseSpecies", sp).lower()
            _group_to_idxs.setdefault(base, []).append(i)
        for idxs in _group_to_idxs.values():
            if len(idxs) > 1:
                for i in idxs:
                    self._clause_blocked_by[i] = frozenset(j for j in idxs if j != i)

        self.natures: list[str] = _ALL_NATURES
        self.n_natures: int = len(_ALL_NATURES)

        # ------------------------------------------------------------------ #
        # Embeddings
        # ------------------------------------------------------------------ #
        self.species_embed = nn.Embedding(self.n_species, _EMBED_DIM)
        self.move_embed = nn.Embedding(
            self.n_moves, _EMBED_DIM, max_norm=_EMBED_DIM**0.5
        )
        self.item_embed = nn.Embedding(
            self.n_items, _EMBED_DIM, max_norm=_EMBED_DIM**0.5
        )
        self.ability_embed = nn.Embedding(
            self.n_abilities, _EMBED_DIM, max_norm=_EMBED_DIM**0.5
        )

        # ------------------------------------------------------------------ #
        # Team encoder: 6 Pokémon tokens → CLS context [D_MODEL]
        # ------------------------------------------------------------------ #
        self.input_proj = nn.Linear(_TOKEN_DIM, self.D_MODEL)
        self.team_cls = nn.Parameter(torch.randn(1, 1, self.D_MODEL))
        self.team_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.D_MODEL,
                nhead=4,
                dim_feedforward=self.D_MODEL,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=3,
            enable_nested_tensor=False,
        )

        # ------------------------------------------------------------------ #
        # Phase 1 — Species head
        # ------------------------------------------------------------------ #
        self.species_head = nn.Linear(self.D_MODEL, self.n_species)
        # Projects each chosen species embed into D_MODEL for running context
        self.chosen_species_proj = nn.Linear(_EMBED_DIM, self.D_MODEL)

        # ------------------------------------------------------------------ #
        # Phase 2 — Build heads
        # ------------------------------------------------------------------ #
        self.slot_proj = nn.Linear(_EMBED_DIM, self.D_MODEL)
        self.build_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.D_MODEL,
                nhead=4,
                dim_feedforward=self.D_MODEL,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.item_head = nn.Linear(self.D_MODEL, self.n_items)
        self.move_heads = nn.ModuleList(
            [nn.Linear(self.D_MODEL, self.n_moves) for _ in range(4)]
        )
        self.nature_head = nn.Linear(self.D_MODEL, self.n_natures)

        # Value head for REINFORCE baseline
        self.value_head = nn.Linear(self.D_MODEL, 1)

    # ------------------------------------------------------------------ #
    # Static helper: poke-env ID normalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_id(name: str) -> str:
        return _to_id(name)

    # ------------------------------------------------------------------ #
    # Per-Pokémon token encoding
    # ------------------------------------------------------------------ #

    def _static_features(self, species: str) -> torch.Tensor:
        """Return the 60-dim static features (types + base stats) for a species."""
        dex_key = re.sub(r"[^a-z0-9]", "", species.split(" (")[0].lower())
        entry = self._gd.pokedex.get(dex_key, {})
        types = entry.get("types", [])
        type_vec = torch.zeros(36)
        for i, t in enumerate(types[:2]):
            if t in _TYPE_NAMES:
                type_vec[i * 18 + _TYPE_NAMES.index(t)] = 1.0
        bs = entry.get("baseStats", {})
        base_stats = torch.tensor(
            [bs.get("hp", 0), bs.get("atk", 0), bs.get("def", 0),
             bs.get("spa", 0), bs.get("spd", 0), bs.get("spe", 0)],
            dtype=torch.float32,
        ) / 255.0
        return torch.cat([type_vec, base_stats])  # [42]

    def _encode_pokemon_token(
        self,
        species: str,
        moves: list[str],
        item: str,
        ability: str,
        evs: tuple,
        nature: str,
    ) -> torch.Tensor:
        """Encode one Pokémon to a 291-dim token tensor."""
        device = self.species_embed.weight.device
        sp_idx = self.species_to_idx.get(species, 0)
        sp_emb = self.species_embed(torch.tensor(sp_idx, device=device))

        move_embs = []
        for m in (moves + ["Protect"] * 4)[:4]:
            mid = _to_id(m)
            m_idx = self.move_to_idx.get(mid, 0)
            move_embs.append(self.move_embed(torch.tensor(m_idx, device=device)))
        move_emb = torch.cat(move_embs)

        i_idx = self.item_to_idx.get(_to_id(item), 0)
        item_emb = self.item_embed(torch.tensor(i_idx, device=device))

        a_idx = self.ability_to_idx.get(_to_id(ability), 0)
        ab_emb = self.ability_embed(torch.tensor(a_idx, device=device))

        static = self._static_features(species).to(device)

        evs_t = torch.tensor([ev / 32.0 for ev in evs], dtype=torch.float32, device=device)

        n_idx = self.natures.index(nature) if nature in self.natures else 0
        nature_oh = F.one_hot(
            torch.tensor(n_idx, device=device), num_classes=self.n_natures
        ).float()

        return torch.cat([sp_emb, move_emb, item_emb, ab_emb, static, evs_t, nature_oh])

    def _encode_team_tokens(self, pokemon_list: list[dict]) -> torch.Tensor:
        """List of 6 Pokémon dicts → [6, TOKEN_DIM]."""
        return torch.stack([
            self._encode_pokemon_token(
                p["species"], p["moves"], p["item"],
                p["ability"], p["evs"], p["nature"],
            )
            for p in pokemon_list
        ])

    def encode_team(self, tokens: torch.Tensor) -> torch.Tensor:
        """[6, TOKEN_DIM] → [D_MODEL] team context via CLS Transformer."""
        projected = self.input_proj(tokens)  # [6, D_MODEL]
        cls = self.team_cls  # [1, 1, D_MODEL]
        seq = torch.cat([cls.squeeze(0), projected], dim=0).unsqueeze(0)  # [1, 7, D_MODEL]
        out = self.team_encoder(seq)  # [1, 7, D_MODEL]
        return out[0, 0]  # [D_MODEL]

    def encode_population(
        self, team_texts: list[str], weights: list[float]
    ) -> torch.Tensor:
        """
        Encode population as Nash-weighted mean of team context vectors.

        Args:
            team_texts: Showdown-format team strings, one per population member.
            weights:    Nash weights (need not sum to 1; will be normalized).

        Returns:
            [D_MODEL] population context tensor.
        """
        contexts = torch.stack([
            self.encode_team(self._encode_team_tokens(_parse_showdown_team(t)))
            for t in team_texts
        ])  # [K, D_MODEL]
        w = torch.tensor(weights, dtype=torch.float32, device=contexts.device)
        w = w / w.sum()
        return (contexts * w.unsqueeze(-1)).sum(dim=0)  # [D_MODEL]

    # ------------------------------------------------------------------ #
    # EVs helper
    # ------------------------------------------------------------------ #

    def _sample_evs(self, species: str, nature: str) -> tuple:
        """Sample EVs from BC corpus conditioned on nature, falling back to any corpus entry."""
        import random
        corpus = self.space.nat_ev_corpus_for(species)
        matching = [evs for n, evs in corpus if n == nature]
        if matching:
            return random.choice(matching)
        if corpus:
            return random.choice(corpus)[1]
        return self.space.random_evs()

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        population_texts: list[str],
        nash_weights: list[float],
    ) -> tuple[CandidateTeam, torch.Tensor, torch.Tensor]:
        """
        Stochastically generate a CandidateTeam from the population context.

        Returns:
            team:     Generated CandidateTeam.
            log_prob: Scalar tensor — sum of log-probs of all discrete choices.
            value:    Scalar baseline estimate from value head.
        """
        pop_ctx = self.encode_population(population_texts, nash_weights)  # [D_MODEL]
        value = self.value_head(pop_ctx).squeeze()

        # ---- Phase 1: species ----
        chosen_species: list[str] = []
        log_probs: list[torch.Tensor] = []
        running_ctx = pop_ctx.clone()

        mega_sp_chosen = False
        used_sp_idxs: set[int] = set()
        restricted_count = 0
        for slot_i in range(TEAM_SIZE):
            logits = self.species_head(running_ctx)  # [n_species]
            for idx in used_sp_idxs:
                logits[idx] = -1e9
            # Enforce the restricted-legendary cap (e.g. Limit Two Restricted):
            # once the cap is hit, mask out all remaining restricted species so
            # Showdown doesn't reject the team.
            if self._restricted_limit > 0 and restricted_count >= self._restricted_limit:
                for i in self._restricted_sp_idxs:
                    logits[i] = -1e9
            # Force a mega-capable species on the last slot if none chosen yet
            # (only relevant in regulations that actually have Mega Evolutions).
            if (
                not mega_sp_chosen
                and slot_i == TEAM_SIZE - 1
                and self._mega_capable_sp_idxs
            ):
                for i in range(self.n_species):
                    if i not in self._mega_capable_sp_idxs:
                        logits[i] = -1e9
            dist = Categorical(logits=logits)
            idx = dist.sample()
            log_probs.append(dist.log_prob(idx))
            sp = self.species_list[idx.item()]
            chosen_species.append(sp)
            used_sp_idxs.add(idx.item())
            # Block all other forms in the same Species Clause group
            used_sp_idxs.update(self._clause_blocked_by.get(idx.item(), frozenset()))
            if idx.item() in self._mega_capable_sp_idxs:
                mega_sp_chosen = True
            if idx.item() in self._restricted_sp_idxs:
                restricted_count += 1
            running_ctx = running_ctx + self.chosen_species_proj(
                self.species_embed(idx)
            )

        # ---- Phase 2: builds ----
        device = self.species_embed.weight.device
        slot_tokens = torch.stack([
            self.slot_proj(
                self.species_embed(torch.tensor(self.species_to_idx[sp], device=device))
            ) + pop_ctx
            for sp in chosen_species
        ]).unsqueeze(0)  # [1, 6, D_MODEL]
        slot_contexts = self.build_encoder(slot_tokens).squeeze(0)  # [6, D_MODEL]

        members: list[PokemonBuild] = []
        used_item_idxs: set[int] = set()
        mega_capable_slots = [i for i, sp in enumerate(chosen_species) if self._species_mega_idxs.get(sp)]
        mega_assigned = False

        for slot_i, (sp, ctx) in enumerate(zip(chosen_species, slot_contexts)):
            valid_item_idxs = set(self._species_valid_item_idxs.get(sp, []))

            # Item
            item_logits = self.item_head(ctx).clone()
            allowed_items = valid_item_idxs - used_item_idxs
            if not allowed_items:
                allowed_items = valid_item_idxs  # relax item clause as last resort

            # Force mega stone on last mega-capable slot if none assigned yet
            if not mega_assigned and mega_capable_slots and slot_i == mega_capable_slots[-1]:
                mega_options = set(self._species_mega_idxs[sp]) & allowed_items
                if mega_options:
                    allowed_items = mega_options

            mask = torch.ones(self.n_items, dtype=torch.bool, device=item_logits.device)
            for i in allowed_items:
                mask[i] = False
            item_logits[mask] = -1e9
            item_dist = Categorical(logits=item_logits)
            item_idx = item_dist.sample()
            log_probs.append(item_dist.log_prob(item_idx))
            used_item_idxs.add(item_idx.item())
            if item_idx.item() in self._mega_stone_idxs:
                mega_assigned = True
            item_id = self.all_items[item_idx.item()]
            item_name = self._item_id_to_name.get(item_id, item_id)

            # Moves
            valid_move_idxs = set(self._species_valid_move_idxs.get(sp, []))
            chosen_move_idxs: set[int] = set()
            move_names: list[str] = []
            for head in self.move_heads:
                move_logits = head(ctx).clone()
                allowed = valid_move_idxs - chosen_move_idxs
                if not allowed:
                    allowed = valid_move_idxs
                move_mask = torch.ones(self.n_moves, dtype=torch.bool, device=move_logits.device)
                for i in allowed:
                    move_mask[i] = False
                move_logits[move_mask] = -1e9
                move_dist = Categorical(logits=move_logits)
                m_idx = move_dist.sample()
                log_probs.append(move_dist.log_prob(m_idx))
                chosen_move_idxs.add(m_idx.item())
                mid = self.all_moves[m_idx.item()]
                move_names.append(self._move_id_to_name.get(mid, mid))

            # Nature
            nature_logits = self.nature_head(ctx)
            nature_dist = Categorical(logits=nature_logits)
            n_idx = nature_dist.sample()
            log_probs.append(nature_dist.log_prob(n_idx))
            nature = self.natures[n_idx.item()]

            # EVs from BC corpus
            evs = self._sample_evs(sp, nature)

            import random as _random
            abilities = self.space.abilities_for(sp)
            ability = _random.choice(abilities) if abilities else "No Ability"

            members.append(PokemonBuild(
                species=sp,
                item=item_name,
                ability=ability,
                nature=nature,
                evs=evs,
                ivs=(31, 31, 31, 31, 31, 31),
                moves=tuple(move_names),  # type: ignore[arg-type]
            ))

        try:
            team = CandidateTeam(members=tuple(members))
        except ValueError:
            team = self._random_mega_team()

        return team, torch.stack(log_probs).sum(), value

    def generate_greedy(
        self,
        population_texts: list[str],
        nash_weights: list[float],
    ) -> CandidateTeam:
        """
        Generate a team deterministically (argmax at each step).

        Used for evaluation after training, not for the training rollout.
        """
        with torch.no_grad():
            pop_ctx = self.encode_population(population_texts, nash_weights)
            chosen_species: list[str] = []
            running_ctx = pop_ctx.clone()
            mega_sp_chosen = False
            used_sp_idxs: set[int] = set()
            restricted_count = 0

            for slot_i in range(TEAM_SIZE):
                logits = self.species_head(running_ctx)
                for idx in used_sp_idxs:
                    logits[idx] = -1e9
                # Enforce the restricted-legendary cap (e.g. Limit Two Restricted).
                if self._restricted_limit > 0 and restricted_count >= self._restricted_limit:
                    for i in self._restricted_sp_idxs:
                        logits[i] = -1e9
                if (
                    not mega_sp_chosen
                    and slot_i == TEAM_SIZE - 1
                    and self._mega_capable_sp_idxs
                ):
                    for i in range(self.n_species):
                        if i not in self._mega_capable_sp_idxs:
                            logits[i] = -1e9
                idx = logits.argmax()
                sp = self.species_list[idx.item()]
                chosen_species.append(sp)
                used_sp_idxs.add(idx.item())
                # Block all other forms in the same Species Clause group
                used_sp_idxs.update(self._clause_blocked_by.get(idx.item(), frozenset()))
                if idx.item() in self._mega_capable_sp_idxs:
                    mega_sp_chosen = True
                if idx.item() in self._restricted_sp_idxs:
                    restricted_count += 1
                running_ctx = running_ctx + self.chosen_species_proj(
                    self.species_embed(idx)
                )

            device = self.species_embed.weight.device
            slot_tokens = torch.stack([
                self.slot_proj(
                    self.species_embed(torch.tensor(self.species_to_idx[sp], device=device))
                ) + pop_ctx
                for sp in chosen_species
            ]).unsqueeze(0)
            slot_contexts = self.build_encoder(slot_tokens).squeeze(0)

            members: list[PokemonBuild] = []
            used_item_idxs: set[int] = set()
            mega_capable_slots = [i for i, sp in enumerate(chosen_species) if self._species_mega_idxs.get(sp)]
            mega_assigned = False

            for slot_i, (sp, ctx) in enumerate(zip(chosen_species, slot_contexts)):
                valid_item_idxs = set(self._species_valid_item_idxs.get(sp, []))
                item_logits = self.item_head(ctx).clone()
                allowed = valid_item_idxs - used_item_idxs or valid_item_idxs

                # Force mega stone on last mega-capable slot if none assigned yet
                if not mega_assigned and mega_capable_slots and slot_i == mega_capable_slots[-1]:
                    mega_options = set(self._species_mega_idxs[sp]) & allowed
                    if mega_options:
                        allowed = mega_options

                mask = torch.ones(self.n_items, dtype=torch.bool, device=item_logits.device)
                for i in allowed:
                    mask[i] = False
                item_logits[mask] = -1e9
                item_idx = item_logits.argmax()
                used_item_idxs.add(item_idx.item())
                if item_idx.item() in self._mega_stone_idxs:
                    mega_assigned = True
                item_id = self.all_items[item_idx.item()]
                item_name = self._item_id_to_name.get(item_id, item_id)

                valid_move_idxs = set(self._species_valid_move_idxs.get(sp, []))
                chosen_move_idxs: set[int] = set()
                move_names: list[str] = []
                for head in self.move_heads:
                    move_logits = head(ctx).clone()
                    allowed_m = valid_move_idxs - chosen_move_idxs or valid_move_idxs
                    move_mask = torch.ones(self.n_moves, dtype=torch.bool, device=move_logits.device)
                    for i in allowed_m:
                        move_mask[i] = False
                    move_logits[move_mask] = -1e9
                    m_idx = move_logits.argmax()
                    chosen_move_idxs.add(m_idx.item())
                    mid = self.all_moves[m_idx.item()]
                    move_names.append(self._move_id_to_name.get(mid, mid))

                nature_logits = self.nature_head(ctx)
                n_idx = nature_logits.argmax()
                nature = self.natures[n_idx.item()]
                evs = self._sample_evs(sp, nature)

                import random as _random
                abilities = self.space.abilities_for(sp)
                ability = _random.choice(abilities) if abilities else "No Ability"

                members.append(PokemonBuild(
                    species=sp,
                    item=item_name,
                    ability=ability,
                    nature=nature,
                    evs=evs,
                    ivs=(31, 31, 31, 31, 31, 31),
                    moves=tuple(move_names),  # type: ignore[arg-type]
                ))

        try:
            return CandidateTeam(members=tuple(members))
        except ValueError:
            return self._random_mega_team()

    # ------------------------------------------------------------------ #
    # Fallback helper
    # ------------------------------------------------------------------ #

    def _random_mega_team(self) -> CandidateTeam:
        """
        Return a random team guaranteed to contain at least one mega stone.

        Used as a fallback when the autoregressive generation produces an
        invalid team (species/item clause edge case). Retries random_team()
        until a mega-capable species appears and swaps in its mega stone.
        """
        import random as _random
        from vgc_bench.team_builder.team import random_team

        mega_capable = {
            sp for sp in self.species_list if self._species_mega_idxs.get(sp)
        }

        # No Mega Evolutions in this regulation (e.g. every Gen 9 VGC format):
        # the autoregressive generation only hit a clause edge case, so fall
        # back to a plain clause-valid random team instead of the mega-only
        # retry loop below (which would never return and then crash on an
        # empty mega_capable set).
        if not mega_capable:
            return random_team(self.space)

        # Build a reverse lookup: species name → clause group key (for filtering)
        clause_group_of: dict[str, str] = {}
        for i, sp in enumerate(self.species_list):
            blocked = self._clause_blocked_by.get(i)
            if blocked:
                clause_group_of[sp] = sp  # representative key per group
                for j in blocked:
                    clause_group_of[self.species_list[j]] = sp

        for _ in range(20):
            team = random_team(self.space)
            # Reject teams with Showdown Species Clause violations
            used_groups: set[str] = set()
            clause_ok = True
            for m in team.members:
                g = clause_group_of.get(m.species, m.species)
                if g in used_groups:
                    clause_ok = False
                    break
                used_groups.add(g)
            if not clause_ok:
                continue

            if any(m.species in mega_capable for m in team.members):
                # Ensure the mega-capable member carries a mega stone
                members = list(team.members)
                for i, m in enumerate(members):
                    if m.species in mega_capable:
                        mega_item_idxs = self._species_mega_idxs[m.species]
                        mega_item_id = self.all_items[_random.choice(mega_item_idxs)]
                        mega_item_name = self._item_id_to_name.get(mega_item_id, mega_item_id)
                        # Only swap if item is not already held by another member
                        other_items = {members[j].item for j in range(len(members)) if j != i}
                        if mega_item_name not in other_items:
                            members[i] = m.replace(item=mega_item_name)
                            break
                try:
                    return CandidateTeam(members=tuple(members))
                except ValueError:
                    continue

        # Last resort: force a known mega-capable species into slot 0
        mega_sp = next(iter(mega_capable))
        mega_item_idxs = self._species_mega_idxs[mega_sp]
        mega_item_id = self.all_items[_random.choice(mega_item_idxs)]
        mega_item_name = self._item_id_to_name.get(mega_item_id, mega_item_id)
        build = self.space.random_build(species=mega_sp)
        build = build.replace(item=mega_item_name)
        fallback = random_team(self.space)
        members = [build] + [m for m in fallback.members if m.species != mega_sp][:5]
        return CandidateTeam(members=tuple(members[:6]))

    # ------------------------------------------------------------------ #
    # Checkpoint utilities
    # ------------------------------------------------------------------ #

    def load_frozen_embeds(self, checkpoint_path: str, device: str = "cpu") -> None:
        """
        Load move/item/ability embeddings from a PPO checkpoint and freeze them.

        Expects the checkpoint to contain an AttentionExtractor under
        policy.features_extractor (Stable Baselines3 format).

        Args:
            checkpoint_path: Path to a .zip PPO checkpoint file.
            device:          PyTorch device string.
        """
        from stable_baselines3 import PPO
        ckpt_policy = PPO.load(checkpoint_path, device=device).policy
        extractor = ckpt_policy.features_extractor
        self.move_embed.weight.data.copy_(extractor.move_embed.weight.data)
        self.item_embed.weight.data.copy_(extractor.item_embed.weight.data)
        self.ability_embed.weight.data.copy_(extractor.ability_embed.weight.data)
        for emb in (self.move_embed, self.item_embed, self.ability_embed):
            for p in emb.parameters():
                p.requires_grad = False
