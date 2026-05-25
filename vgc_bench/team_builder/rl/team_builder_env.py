"""
Battle evaluation helper for SP-PSRO team builder training.

TeamBuilderBattleEnv wraps TeamEvaluator so the training loop can evaluate
generated teams against population members without managing player lifecycle.
"""

from __future__ import annotations

from pathlib import Path

from vgc_bench.team_builder.evaluator import TeamEvaluator
from vgc_bench.team_builder.team import CandidateTeam


class TeamBuilderBattleEnv:
    """
    Thin wrapper around TeamEvaluator for SP-PSRO training.

    One env instance holds one opponent slot. Swap opponents by calling
    set_opponent() rather than creating new instances — this reuses the
    poke-env player objects and avoids asyncio state issues.

    Args:
        initial_opponent:   Packed team string for the first opponent.
        n_battles:          Battles per evaluation (more = lower variance reward).
        port:               Showdown server port.
        reg:                VGC regulation identifier.
        battle_agent_path:  Optional path to a trained PPO checkpoint. When set,
                            both sides of the battle use that policy instead of
                            SimpleHeuristicsPlayer.
        device:             PyTorch device for the policy (default "cpu").
    """

    def __init__(
        self,
        initial_opponent: str,
        n_battles: int = 5,
        port: int = 8100,
        reg: str = "i",
        battle_agent_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self._evaluator = TeamEvaluator(
            opponent_team=initial_opponent,
            n_battles=n_battles,
            port=port,
            reg=reg,
            battle_agent_path=battle_agent_path,
            device=device,
        )
        self._current_opponent = initial_opponent

    def set_opponent(self, packed_opponent: str) -> None:
        """Swap the fixed opponent without recreating the player pair."""
        self._evaluator._opponent_builder.set_team(packed_opponent)
        self._current_opponent = packed_opponent

    def evaluate(self, team: CandidateTeam) -> float:
        """
        Battle team against the current opponent.

        Returns:
            Win rate in [0, 1] over n_battles.
        """
        scored = self._evaluator.evaluate(team)
        return scored.win_rate or 0.0

    def evaluate_vs_mixture(
        self,
        team: CandidateTeam,
        opponent_texts: list[str],
        weights: list[float],
    ) -> float:
        """
        Evaluate team against a weighted mixture of opponents.

        Args:
            team:           The team to evaluate.
            opponent_texts: Showdown-format team strings.
            weights:        Nash weights (need not sum to 1; will be normalized).

        Returns:
            Weighted average win rate in [0, 1].
        """
        from poke_env.teambuilder import Teambuilder
        total_w = sum(weights)
        if total_w <= 0:
            return 0.0

        total = 0.0
        for text, w in zip(opponent_texts, weights):
            if w <= 0:
                continue
            packed = Teambuilder.join_team(Teambuilder.parse_showdown_team(text))
            self.set_opponent(packed)
            total += self.evaluate(team) * (w / total_w)
        return total
