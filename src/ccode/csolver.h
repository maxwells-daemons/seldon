#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct _move {
    int x;
    int y;
    int score;
} move;

move bitboard_solve_game(uint64_t player, uint64_t opp);
