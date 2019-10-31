from random import choice

import numpy as np  # type: ignore

from bitboard import find_moves  # type: ignore
from game import Move, PlayerABC
from utils import moves_list


class RandomPlayer(PlayerABC):
    def get_move(self, player_board: np.ndarray, opponent_board: np.ndarray) -> Move:
        moves_bitboard = find_moves(player_board, opponent_board)
        all_moves = moves_list(moves_bitboard)
        return choice(all_moves)
