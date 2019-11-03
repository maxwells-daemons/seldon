#include "csolver.h"
#include "cbitboard.h"

#define FASTEST_FIRST_CUTOFF 5 // Depth below which we stop using fastest-first
#define MAX_MOVES 32 // Maximum number of moves we'll need to consider
#define _INFINITY 999
#define BOARD_SIZE 8
#define MAX_SCORE 64

// When enabled, uses "winner gets empties" scoring and tries to maximize score.
// Comment out in practice for speed.
/* #define BENCHMARK */

#ifdef BENCHMARK
#define INITIAL_BOUND MAX_SCORE
#else
#define INITIAL_BOUND 1
#endif

static inline int evaluate(uint64_t player, uint64_t opp);
static inline unsigned int mobility(uint64_t player, uint64_t opp);
static inline int negamax(uint64_t player, uint64_t opp, int alpha, int beta,
                          bool passed);
static inline int negamax_fastest_first(uint64_t player, uint64_t opp, int alpha,
                                        int beta, bool passed, int depth);
static inline int move_index(uint64_t bitboard);

move c_solve_game(uint64_t player, uint64_t opp) {
    move result;

    int depth = MAX_SCORE - (int) popcount(player) - (int) popcount(opp);
    uint64_t moves = bitboard_find_moves(player, opp);
    uint64_t new_move;
    uint64_t new_disks;
    uint64_t player_board;
    uint64_t opp_board;
    int score;
    int max_score = -_INFINITY;
    int index = -1;

    while (moves != 0) {
        new_move = bitboard_extract_disk(moves);
        moves &= ~new_move;
        new_disks = bitboard_resolve_move(player, opp, new_move);
        player_board = (player ^ new_disks) | new_move;
        opp_board = opp ^ new_disks;
        score = -negamax_fastest_first(opp_board, player_board, -INITIAL_BOUND,
                                       INITIAL_BOUND, false, depth);

        if (score > max_score) {
            max_score = score;
            index = move_index(new_move);
        }
    }

    if (index < 0) { // Indicates no legal move
        result.x = -1;
        result.y = -1;
        result.score = _INFINITY;
    } else {
        result.x = index % BOARD_SIZE;
        result.y = index / BOARD_SIZE;
        result.score = max_score;
    }

    return result;
};

// Return the maximum value of children of this node if not pruning, or the
// value of alpha if pruning. Updates alpha.
static inline int negamax(uint64_t player, uint64_t opp, int alpha, int beta, bool
                          passed) {
    uint64_t moves = bitboard_find_moves(player, opp);
    if (moves == 0) {
        if (passed) {
            return evaluate(player, opp); // Game is over
        }

        // Skip our turn
        return -negamax(opp, player, -beta, -alpha, true);
    }

    int max_score = -_INFINITY;
    int score;
    uint64_t player_boards[MAX_MOVES];
    uint64_t opp_boards[MAX_MOVES];
    uint64_t new_move;
    uint64_t new_disks;
    unsigned char n_moves = 0;

    while (moves != 0) {
        new_move = bitboard_extract_disk(moves);
        moves &= ~new_move;
        new_disks = bitboard_resolve_move(player, opp, new_move);
        player_boards[n_moves] = (player ^ new_disks) | new_move;
        opp_boards[n_moves] = opp ^ new_disks;
        n_moves += 1;
    }

    for (int i = 0; i < n_moves; i++) {
        score = -negamax(opp_boards[i], player_boards[i], -beta, -alpha, false);

        if (score > max_score) {
            max_score = score;

            if (max_score > alpha) {
                alpha = max_score;

                if (alpha >= beta) { // Prune
                    return alpha;
                }
            }
        }
    }

    return max_score;
}

// Reimplemented for efficiency in time and stack space
static inline int negamax_fastest_first(uint64_t player, uint64_t opp, int alpha,
                                        int beta, bool passed, int depth) {
    if (depth < FASTEST_FIRST_CUTOFF) {
        return negamax(player, opp, alpha, beta, passed);
    }

    uint64_t moves = bitboard_find_moves(player, opp);
    if (moves == 0) {
        if (passed) {
            return evaluate(player, opp); // Game is over
        }

        // Skip our turn
        return -negamax_fastest_first(opp, player, -beta, -alpha, true, depth);
    }

    int max_score = -_INFINITY;
    int score;
    uint64_t player_board;
    uint64_t opp_board;
    uint64_t player_boards[MAX_MOVES];
    uint64_t opp_boards[MAX_MOVES];
    unsigned int opp_mobilities[MAX_MOVES];
    uint64_t new_move;
    uint64_t new_disks;
    unsigned char n_moves = 0;

    while (moves != 0) {
        new_move = bitboard_extract_disk(moves);
        moves &= ~new_move;
        new_disks = bitboard_resolve_move(player, opp, new_move);

        player_board = (player ^ new_disks) | new_move;
        opp_board = opp ^ new_disks;
        player_boards[n_moves] = player_board;
        opp_boards[n_moves] = opp_board;
        opp_mobilities[n_moves] = mobility(opp_board, player_board);

        n_moves += 1;
    }

    unsigned int best_mobility;
    int best_index;
    uint64_t opp_mobility;

    for (int i = 0; i < n_moves; i++) {
        best_mobility = MAX_MOVES + 1;
        best_index = -1;

        // For small numbers of possible moves, this outperforms explicit sorting
        for (int j = 0; j < n_moves; j++) {
            opp_mobility = opp_mobilities[j];
            if (opp_mobility < best_mobility) {
                best_mobility = opp_mobility;
                best_index = j;
            }
        }
        opp_mobilities[best_index] = MAX_MOVES + 1;

        score = -negamax_fastest_first(opp_boards[best_index],
                                       player_boards[best_index], -beta, -alpha,
                                       false, depth - 1);

        if (score > max_score) {
            max_score = score;

            if (max_score > alpha) {
                alpha = max_score;

                if (alpha >= beta) { // Prune
                    return alpha;
                }
            }
        }
    }

    return max_score;
}

static inline int evaluate(uint64_t player, uint64_t opp) {
#ifdef BENCHMARK
    // "Winner gets empties" scoring
    int score = (int) popcount(player) - (int) popcount(opp);
    int empties = popcount(~player & ~opp);
    return score > 0 ? score + empties : score - empties;
#else
    return (int) popcount(player) - (int) popcount(opp);
#endif
}

static inline unsigned int mobility(uint64_t player, uint64_t opp) {
    uint64_t _moves = bitboard_find_moves(player, opp);
    return popcount(_moves);
}

static inline int move_index(uint64_t bitboard) {
    int i = 0;
    while (!(bitboard & (uint64_t) 1)) {
        bitboard = bitboard >> 1ULL;
        i += 1;
    }
    return i;
}
