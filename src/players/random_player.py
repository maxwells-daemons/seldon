import random
from typing import Optional

from board import Board, Loc
from player import PlayerABC


class RandomPlayer(PlayerABC):
    def _get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int] = None
    ) -> Loc:
        if not board.has_moves(self.color):
            return Loc.pass_loc()

        return random.choice(board.find_moves(self.color).loc_list)
