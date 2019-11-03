# cython: language_level=3, boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np

DEF BOARD_SIZE = 8

# C ULL <-> ndarray interface
cpdef Bitboard serialize_piecearray(np.ndarray arr):
    cdef np.ndarray bytestring = np.packbits(arr, bitorder='big')
    cdef Bitboard serialized = int.from_bytes(bytestring, byteorder='big',
                                                        signed=False)
    return serialized

cpdef np.ndarray deserialize_piecearray(Bitboard serialized):
    cdef bytes bytestring = serialized.to_bytes(length=BOARD_SIZE, byteorder='big')
    cdef np.ndarray byte_arr = np.frombuffer(bytestring, dtype=np.uint8)
    cdef np.ndarray flat = np.unpackbits(byte_arr)
    return np.reshape(flat.astype(bool), (BOARD_SIZE, BOARD_SIZE))


# High-level functions
def resolve_move(Bitboard player, Bitboard opp, unsigned int x, unsigned int y):
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
    cdef move_bitboard = make_singleton_bitboard(x, y)
    cdef new_disks = bitboard_resolve_move(player, opp, move_bitboard)
    player = (player ^ new_disks) | move_bitboard
    opp ^= new_disks
    return (player, opp)
