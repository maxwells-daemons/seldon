"""
Wrapper code to make players compatible with Caltech's CS2 framework.
A player should be configured with a "player config" file, which
can be called as a script or embedded into C with `cython --embed`.
"""

import sys
from typing import Optional

from board import Board, Loc, PlayerColor
from player import PlayerABC


def run_player(player: PlayerABC, **player_kwargs) -> None:
    color = {"Black": PlayerColor.BLACK, "White": PlayerColor.WHITE}[sys.argv[1]]
    board = Board.starting_board()
    print(f"Player ready: {player.__class__.__name__} ({color.value})")

    while True:
        opp_x_, opp_y_, ms_left_ = input().split()
        opp_x, opp_y = int(opp_x_), int(opp_y_)
        ms_left: Optional[int] = int(ms_left_)
        if ms_left == -1:
            ms_left = None

        if not player._initialized:
            player.initialize(color, ms_left)

        if opp_x < 0:  # Opponent passed
            last_move = None
        else:
            last_move = Loc(opp_x, opp_y)
            board = board.resolve_move(last_move, color.opponent)

        move = player.get_move(board, last_move, ms_left)
        if move != Loc.pass_loc():
            board = board.resolve_move(move, color)
        print(f"{move.x} {move.y}")
