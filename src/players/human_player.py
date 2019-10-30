from string import ascii_lowercase

import numpy as np  # type: ignore

from backend import find_moves  # type: ignore
from game import BOARD_SIZE, Board, Move, PlayerABC
from utils import moves_list


def _parse_input(input_: str) -> Move:
    x_, y_ = input_
    x = ascii_lowercase[:BOARD_SIZE].index(x_)
    y = int(y_) - 1
    return Move(x, y)


class HumanPlayer(PlayerABC):
    def get_move(self, player_board: np.ndarray, opponent_board: np.ndarray) -> Move:
        moves_bitboard = find_moves(player_board, opponent_board)
        all_moves = list(sorted(moves_list(moves_bitboard)))
        board = Board(
            **{
                self.color.value: player_board,
                self.color.opponent().value: opponent_board,
            }
        )

        board_rep = board._string_array()
        board_rep[1:, 1:][np.where(moves_bitboard)] = "-"
        print("\n" + np.array2string(board_rep, formatter={"numpystr": str}))
        print(f"Legal moves: {all_moves}")

        while True:
            try:
                input_ = input("Enter a move: ")
                move = _parse_input(input_)
            except ValueError:
                print("Invalid input. Please try again.")
                continue

            if moves_bitboard[move.y, move.x]:
                return move

            print(f"Illegal move: {move}. Please try again.")
