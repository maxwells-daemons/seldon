# cython: language_level=3, boundscheck=False, wraparound=False
from cpython cimport array
from libc.stdlib cimport malloc, free, rand

from bitboard cimport Bitboard, bitboard_find_moves, popcount, bitboard_resolve_move

from board import GameOutcome, PlayerColor

cdef extern from "cbitboard.c":
    cdef unsigned int select_bit(Bitboard bitboard, unsigned int rank)

cpdef enum Rollout_Result: ACTIVE, OPPONENT, DRAW

### Internals ###
cdef Bitboard random_select(Bitboard bitboard):
    cdef int n_moves = popcount(bitboard)
    cdef unsigned int loc_idx = rand() % n_moves
    cdef unsigned int move_loc = select_bit(bitboard, loc_idx + 1)
    return 1ULL << (move_loc - 1)


cdef Rollout_Result _random_rollout(Bitboard active_bitboard, Bitboard other_bitboard):
    cdef bint same_player = True
    cdef bint just_passed = False
    cdef Bitboard moves
    cdef Bitboard new_move
    cdef Bitboard new_disks

    while True:
        moves = bitboard_find_moves(active_bitboard, other_bitboard)
        if moves == 0:  # Pass
            if just_passed:  # Game over
                break
            just_passed = True
        else:  # Make a move
            just_passed = False
            new_move = random_select(moves)
            new_disks = bitboard_resolve_move(active_bitboard, other_bitboard, new_move)
            active_bitboard = (active_bitboard ^ new_disks) | new_move
            other_bitboard ^= new_disks

        same_player = not same_player
        active_bitboard, other_bitboard = other_bitboard, active_bitboard

    cdef int score = popcount(active_bitboard) - popcount(other_bitboard)
    if score == 0:
        return DRAW
    if (score > 0) == same_player:
        return ACTIVE
    return OPPONENT

### Top-level functions ###
def random_rollout(
    active_bitboard: int, opponent_bitboard: int, player: PlayerColor
) -> GameOutcome:
    result = _random_rollout(active_bitboard, opponent_bitboard)
    return {
        DRAW: GameOutcome.DRAW,
        ACTIVE: player.winning_outcome,
        OPPONENT: player.opponent.winning_outcome
    }[result]
