# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np

from bitboard import _serialize

def solve_game(np.ndarray player, np.ndarray opp):
    '''
    Solve a board with the explicit endgame solver, returning the best next move.

    Parameters
    ----------
    player : bool ndarray
        Active player's board.
    opp : bool ndarray
        Opponent's board.

    Returns
    -------
    x : int
    y : int
        Coordinates of the best move on this board.
    score : int
        Expected final score.
    '''
    cdef move c_move = c_solve_game(_serialize(player), _serialize(opp))
    return (7 - c_move.x, 7 - c_move.y, c_move.score)
