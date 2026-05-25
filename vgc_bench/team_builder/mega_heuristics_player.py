from __future__ import annotations

from poke_env.battle.move import Move
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder


class MegaSimpleHeuristicsPlayer(SimpleHeuristicsPlayer):
    """SimpleHeuristicsPlayer that uses Mega Evolution on the first available turn."""

    def choose_move(self, battle):
        order = super().choose_move(battle)

        if isinstance(order, DoubleBattleOrder):
            mega_flags = battle.can_mega_evolve  # List[bool], one per active slot
            first = order.first_order
            second = order.second_order

            if len(mega_flags) > 0 and mega_flags[0] and isinstance(first.order, Move):
                first = SingleBattleOrder(first.order, mega=True, move_target=first.move_target)
            elif len(mega_flags) > 1 and mega_flags[1] and isinstance(second.order, Move):
                second = SingleBattleOrder(second.order, mega=True, move_target=second.move_target)

            return DoubleBattleOrder(first_order=first, second_order=second)

        # Singles
        if battle.can_mega_evolve and isinstance(order.order, Move):
            return SingleBattleOrder(order.order, mega=True, move_target=order.move_target)

        return order
