# cython: language_level=3, boundscheck=False, wraparound=False
from bitboard cimport Bitboard, move

cdef extern from "csolver.c":
    cdef move bitboard_solve_game(Bitboard player, Bitboard opp)
