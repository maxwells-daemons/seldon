# cython: language_level=3, boundscheck=False, wraparound=False
from libc cimport stdint
from typing import NamedTuple

ctypedef stdint.uint64_t Bitboard

cdef extern from "cbitboard.c":
    cpdef unsigned int popcount(Bitboard x)
    cpdef Bitboard bitboard_resolve_move(Bitboard player, Bitboard opp,
                                         Bitboard new_disk)
    cpdef Bitboard make_singleton_bitboard(unsigned int x, unsigned int y)
    cpdef Bitboard bitboard_find_moves(Bitboard player, Bitboard opp)
    cpdef Bitboard bitboard_stability(Bitboard player, Bitboard opp)

cdef extern from "csolver.h":
    cdef struct _move:
        int x
        int y
        int score
    ctypedef _move move
