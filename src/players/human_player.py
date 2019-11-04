from string import ascii_lowercase
from typing import Optional

import numpy as np  # type: ignore

from board import BOARD_SIZE, Board, Loc
from player import PlayerABC


def _parse_input(input_: str) -> Loc:
    x_, y_ = input_
    x = ascii_lowercase[:BOARD_SIZE].index(x_)
    y = int(y_) - 1
    return Loc(x, y)


class HumanPlayer(PlayerABC):
    def _get_move(
        self, board: Board, opponent_move: Optional[Loc], ms_left: Optional[int] = None
    ) -> Loc:
        moves_bitboard = board.find_moves(self.color)
        all_moves = moves_bitboard.loc_list
        board_rep = board._string_array()
        board_rep[1:, 1:][np.where(moves_bitboard.piecearray)] = "-"
        self.logger.info("\n" + np.array2string(board_rep, formatter={"numpystr": str}))
        self.logger.info(f"To move: {self.color.value}.")
        self.logger.info(f"Legal moves: {all_moves}")

        if not board.has_moves(self.color):
            return Loc.pass_loc()

        while True:
            try:
                input_ = input("Enter a move: ")
                move = _parse_input(input_)
            except ValueError:
                print("Invalid input. Please try again.")
                continue

            if move in all_moves:
                return move

            self.logger.error(f"Illegal move: {move}. Please try again.")
