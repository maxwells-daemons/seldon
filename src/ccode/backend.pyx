# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
from backend cimport (c_popcount_64, c_resolve_move, c_make_singleton_bitboard,
                      c_find_moves, c_stability, _move, move, c_solve_game)

# C ULL <-> ndarray interface
cpdef Bitboard _serialize(np.ndarray arr):
    cdef np.ndarray bytestring = np.packbits(arr, bitorder='big')
    cdef Bitboard serialized = int.from_bytes(bytestring, byteorder='big',
                                                        signed=False)
    return serialized

cpdef np.ndarray _deserialize(Bitboard serialized):
    cdef bytes bytestring = serialized.to_bytes(length=8, byteorder='big')
    cdef np.ndarray byte_arr = np.frombuffer(bytestring, dtype=np.uint8)
    cdef np.ndarray flat = np.unpackbits(byte_arr)
    return np.reshape(flat.astype(bool), (8, 8))


# High-level interface
def resolve_move(np.ndarray player, np.ndarray opp, unsigned int x, unsigned int y):
    '''
    Find the result of a player making a move.

    Parameters
    ----------
    player : bool ndarray
        Active player's board.
    opp : bool ndarray
        Opponent's board.
    x : int
    y : int
        Coordinates of the active player's move.

    Returns
    -------
    bool ndarray
        The active player's new board.
    bool ndarray
        The opponent's new board.
    '''
    cdef player_ser = _serialize(player)
    cdef opp_ser = _serialize(opp)
    cdef move_bitboard = c_make_singleton_bitboard(x, y)
    cdef new_disks = c_resolve_move(player_ser, opp_ser, move_bitboard)
    player_ser = (player_ser ^ new_disks) | move_bitboard
    opp_ser ^= new_disks
    return (_deserialize(player_ser), _deserialize(opp_ser))

cpdef np.ndarray find_moves(np.ndarray player, np.ndarray opp):
    '''
    Find legal moves on a board.

    Parameters
    ----------
    player : bool ndarray
        Active player's board.
    opp : bool ndarray
        Opponent's board.

    Returns
    -------
    bool ndarray
        Board of the active player's legal moves.
    '''
    return _deserialize(c_find_moves(_serialize(player), _serialize(opp)))

cpdef np.ndarray stability(np.ndarray player, np.ndarray opp):
    '''
    Find stable disks (disks that cannot be flipped) on a board.

    Parameters
    ----------
    player : bool ndarray
        Active player's board.
    opp : bool ndarray
        Opponent's board.

    Returns
    -------
    bool ndarray
        Board of the active player's stable disks.
    '''
    return _deserialize(c_stability(_serialize(player), _serialize(opp)))

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
