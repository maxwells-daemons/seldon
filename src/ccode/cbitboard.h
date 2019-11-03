#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

unsigned int popcount(uint64_t x);
uint64_t bitboard_extract_disk(uint64_t bitboard);
uint64_t bitboard_resolve_move(uint64_t player, uint64_t opp, uint64_t new_disk);
uint64_t make_singleton_bitboard(unsigned int x, unsigned int y);
uint64_t bitboard_find_moves(uint64_t player, uint64_t opp);
uint64_t bitboard_stability(uint64_t player, uint64_t opp);
unsigned int select_bit(uint64_t bitboard, unsigned int rank);
