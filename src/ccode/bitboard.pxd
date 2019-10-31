# cython: language_level=3, boundscheck=False, wraparound=False
from libc cimport stdint

ctypedef stdint.uint64_t Bitboard

cdef extern from "cbitboard.c":
    cpdef unsigned int c_popcount_64(Bitboard x)
    cpdef Bitboard c_resolve_move(Bitboard player, Bitboard opp, Bitboard new_disk)
    cpdef Bitboard c_make_singleton_bitboard(unsigned int x, unsigned int y)
    cpdef Bitboard c_find_moves(Bitboard player, Bitboard opp)
    cpdef Bitboard c_stability(Bitboard player, Bitboard opp)

cdef extern from "csolver.h":
    cdef struct _move:
        int x
        int y
        int score
    ctypedef _move move
