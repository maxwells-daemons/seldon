from functools import wraps
from typing import Callable, List, Type

import numpy as np  # type: ignore

from game import BOARD_SIZE, Move, PlayerABC, PlayerColor
from solver import solve_game  # type: ignore


def moves_list(bitboard: np.ndarray) -> List[Move]:
    """
    Convert a bitboard of moves into a list of moves.
    """
    return [Move(x, y) for y, x in np.argwhere(bitboard)]


def with_solver(depth: int) -> Callable[[Type[PlayerABC]], Type[PlayerABC]]:
    """
    Wrap a player such that after few enough spots are left empty, moves are made
    with the endgame solver.
    """

    def decorator(player_constructor: Type[PlayerABC]) -> Type[PlayerABC]:
        class PlayerWithSolver(PlayerABC):
            @wraps(player_constructor)
            def __init__(self, color: PlayerColor) -> None:
                self.player = player_constructor(color)

            def get_move(self, player: np.ndarray, opponent: np.ndarray) -> Move:
                empties = (
                    BOARD_SIZE * BOARD_SIZE
                    - np.count_nonzero(player)
                    - np.count_nonzero(opponent)
                )

                if empties <= depth:
                    x, y, _ = solve_game(player, opponent)
                    return Move(x, y)

                return self.player.get_move(player, opponent)

        return PlayerWithSolver

    return decorator
