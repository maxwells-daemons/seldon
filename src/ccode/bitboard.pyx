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
