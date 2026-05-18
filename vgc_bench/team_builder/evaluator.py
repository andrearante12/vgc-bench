"""
Battle-based fitness evaluation for the VGC-Bench team builder.

TeamEvaluator runs live Showdown battles between a candidate team and a fixed
opponent, returning the candidate's win rate as its score.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from poke_env.teambuilder import Teambuilder

from vgc_bench.src.utils import format_map
from vgc_bench.team_builder.team import CandidateTeam

_WS_URL_TEMPLATE: str = "ws://localhost:{port}/showdown/websocket"
_AUTH_URL: str = "https://play.pokemonshowdown.com/action.php?"


class MutableTeamBuilder(Teambuilder):
    """
    A Teambuilder whose team string can be swapped between battles.

    Allows a persistent player instance to be reused across evaluate() calls
    by updating the packed team before each battle run — mirrors the pattern
    used in Callback.compare().
    """

    def __init__(self, packed: str) -> None:
        self._packed = packed

    def yield_team(self) -> str:
        return self._packed

    def set_team(self, packed: str) -> None:
        self._packed = packed


class TeamEvaluator:
    """
    Evaluates a CandidateTeam by running live Showdown battles against a
    fixed opponent team.

    Constructs one persistent candidate/opponent player pair at initialisation
    and reuses it across repeated evaluate() calls, resetting battle counters
    between runs. This mirrors the Callback.compare() pattern used throughout
    VGC-Bench to avoid asyncio thread interference from creating new players.

    Attributes:
        n_battles: Number of battles played per evaluation.
        port: Showdown server WebSocket port.
    """

    def __init__(
        self,
        opponent_team: str,
        n_battles: int = 20,
        port: int = 8100,
        battle_agent_path: Path | None = None,
        device: str = "cpu",
        log_level: int = logging.WARNING,
        max_concurrent: int = 10,
        reg: str = "ma",
    ) -> None:
        """
        Initialise the evaluator and create the persistent player pair.

        Args:
            opponent_team: Packed team string for the fixed opponent. Use
                CandidateTeam.to_packed_team() to convert a CandidateTeam.
            n_battles: How many battles to play when scoring a candidate team.
            port: Local Showdown server port (default 8100).
            battle_agent_path: Path to a PPO checkpoint zip. When provided,
                both candidate and opponent use that policy for move decisions.
                When None, both use SimpleHeuristicsPlayer.
            device: PyTorch device string (only used when battle_agent_path
                is set).
            log_level: Logging verbosity (default WARNING to suppress noise).
            max_concurrent: Maximum concurrent battles per player.
            reg: VGC regulation identifier used to look up the battle format
                (default "ma" for gen9championsvgc2026regma).
        """
        self.n_battles = n_battles
        self.port = port
        self._battle_agent_path = battle_agent_path
        self._device = device
        self._log_level = log_level
        self._max_concurrent = max_concurrent
        self._battle_format = format_map[reg]

        self._candidate_builder = MutableTeamBuilder("")
        self._opponent_builder = MutableTeamBuilder(opponent_team)
        self._candidate = self._make_player(self._candidate_builder)
        self._opponent = self._make_player(self._opponent_builder)

    def evaluate(self, team: CandidateTeam) -> CandidateTeam:
        """
        Score a candidate team by playing self.n_battles against the opponent.

        Reuses the persistent player pair, swapping the candidate's team via
        MutableTeamBuilder and resetting battle counters between calls.

        Args:
            team: The candidate team to evaluate.

        Returns:
            A new CandidateTeam with win_rate set to the fraction of battles won.
        """
        self._candidate_builder.set_team(team.to_packed_team())
        asyncio.run(
            self._candidate.battle_against(self._opponent, n_battles=self.n_battles)
        )
        win_rate = self._candidate.win_rate
        self._candidate.reset_battles()
        self._opponent.reset_battles()
        return team.with_win_rate(win_rate)

    def evaluate_vs_mixture(
        self,
        team: CandidateTeam,
        opponents: list[str],
        weights: list[float],
    ) -> CandidateTeam:
        """
        Score a candidate team against a weighted mixture of opponent teams.

        Runs n_battles against each opponent in sequence, reusing the persistent
        player pair by swapping both builders between opponent teams.

        Args:
            team: The candidate team to evaluate.
            opponents: List of packed opponent team strings.
            weights: Probability weights for each opponent (must sum to ~1).

        Returns:
            A new CandidateTeam with win_rate set to the weighted average.
        """
        if len(opponents) != len(weights):
            raise ValueError("opponents and weights must have the same length")
        candidate_packed = team.to_packed_team()
        self._candidate_builder.set_team(candidate_packed)
        total = 0.0
        for opp_packed, weight in zip(opponents, weights):
            if weight <= 0:
                continue
            self._opponent_builder.set_team(opp_packed)
            asyncio.run(
                self._candidate.battle_against(
                    self._opponent, n_battles=self.n_battles
                )
            )
            total += self._candidate.win_rate * weight
            self._candidate.reset_battles()
            self._opponent.reset_battles()
        return team.with_win_rate(total)

    def _make_server_config(self) -> ServerConfiguration:
        return ServerConfiguration(
            _WS_URL_TEMPLATE.format(port=self.port),
            _AUTH_URL,
        )

    def _make_player(self, builder: MutableTeamBuilder) -> SimpleHeuristicsPlayer:
        return SimpleHeuristicsPlayer(
            server_configuration=self._make_server_config(),
            battle_format=self._battle_format,
            log_level=self._log_level,
            max_concurrent_battles=self._max_concurrent,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=builder,
        )
