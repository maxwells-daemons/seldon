// Code from:
// https://www.hanshq.net/othello.html
// https://github.com/Whillikers/Asimov-othello

#include "cbitboard.h"

#define NUM_DIRS 8

/*
 * Occluded Fill operations
 *
 * gen: sources of fill
 * pro: propagators of fill
 */
unsigned int c_popcount_64(uint64_t x) {
#ifdef __GNUC__
    return __builtin_popcountll(x);
#else
    x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
    return (x * 0x0101010101010101ULL) >> 56;
#endif
}

//bitmasks to filter out certain Files
const uint64_t _notAFile = 0xfefefefefefefefe; // ~0x0101010101010101
const uint64_t _notHFile = 0x7f7f7f7f7f7f7f7f; // ~0x8080808080808080

static inline uint64_t soutOccl(uint64_t gen, uint64_t pro) {
    gen |= pro & (gen >>  8U); // NOLINT
    pro &=       (pro >>  8U); // NOLINT
    gen |= pro & (gen >> 16U); // NOLINT
    pro &=       (pro >> 16U); // NOLINT
    gen |= pro & (gen >> 32U); // NOLINT
    return gen;
}

static inline uint64_t sout_one(uint64_t gen) {
    return gen >> 8U; // NOLINT
}

static inline uint64_t nortOccl(uint64_t gen, uint64_t pro) {
    gen |= pro & (gen <<  8U); // NOLINT
    pro &=       (pro <<  8U); // NOLINT
    gen |= pro & (gen << 16U); // NOLINT
    pro &=       (pro << 16U); // NOLINT
    gen |= pro & (gen << 32U); // NOLINT
    return gen;
}

static inline uint64_t nort_one(uint64_t gen) {
    return gen << 8U; // NOLINT
}

static inline uint64_t eastOccl(uint64_t gen, uint64_t pro) {
    pro &= _notAFile;
    gen |= pro & (gen << 1U); // NOLINT
    pro &=       (pro << 1U); // NOLINT
    gen |= pro & (gen << 2U); // NOLINT
    pro &=       (pro << 2U); // NOLINT
    gen |= pro & (gen << 4U); // NOLINT
    return gen;
}

static inline uint64_t east_one(uint64_t gen) {
    gen &= _notAFile;
    return gen << 1U; // NOLINT
}

static inline uint64_t noEaOccl(uint64_t gen, uint64_t pro) {
    pro &= _notAFile;
    gen |= pro & (gen <<  9U); // NOLINT
    pro &=       (pro <<  9U); // NOLINT
    gen |= pro & (gen << 18U); // NOLINT
    pro &=       (pro << 18U); // NOLINT
    gen |= pro & (gen << 36U); // NOLINT
    return gen;
}

static inline uint64_t noea_one(uint64_t gen) {
    gen &= _notAFile;
    return gen << 9U; // NOLINT
}

static inline uint64_t soEaOccl(uint64_t gen, uint64_t pro) {
    pro &= _notAFile;
    gen |= pro & (gen >>  7U); // NOLINT
    pro &=       (pro >>  7U); // NOLINT
    gen |= pro & (gen >> 14U); // NOLINT
    pro &=       (pro >> 14U); // NOLINT
    gen |= pro & (gen >> 28U); // NOLINT
    return gen;
}

static inline uint64_t soea_one(uint64_t gen) {
    gen &= _notAFile;
    return gen >> 7; // NOLINT
}

static inline uint64_t westOccl(uint64_t gen, uint64_t pro) {
    pro &= _notHFile;
    gen |= pro & (gen >> 1U); // NOLINT
    pro &=       (pro >> 1U); // NOLINT
    gen |= pro & (gen >> 2U); // NOLINT
    pro &=       (pro >> 2U); // NOLINT
    gen |= pro & (gen >> 4U); // NOLINT
    return gen;
}

static inline uint64_t west_one(uint64_t gen) {
    gen &= _notHFile;
    return gen >> 1U; // NOLINT
}

static inline uint64_t soWeOccl(uint64_t gen, uint64_t pro) {
    pro &= _notHFile;
    gen |= pro & (gen >>  9U); // NOLINT
    pro &=       (pro >>  9U); // NOLINT
    gen |= pro & (gen >> 18U); // NOLINT
    pro &=       (pro >> 18U); // NOLINT
    gen |= pro & (gen >> 36U); // NOLINT
    return gen;
}

static inline uint64_t sowe_one(uint64_t gen) {
    gen &= _notHFile;
    return gen >> 9U; // NOLINT
}

static inline uint64_t noWeOccl(uint64_t gen, uint64_t pro) {
    pro &= _notHFile;
    gen |= pro & (gen <<  7U);  // NOLINT
    pro &=       (pro <<  7U);  // NOLINT
    gen |= pro & (gen << 14U);  // NOLINT
    pro &=       (pro << 14U);  // NOLINT
    gen |= pro & (gen << 28U);  // NOLINT
    return gen;
}

static inline uint64_t nowe_one(uint64_t gen) {
    gen &= _notHFile;
    return gen << 7U; // NOLINT
}

/*
 * Mid-level bitboard operations
 */
// Get a bitboard containing just the "lowest" disk in a bitboard
uint64_t c_extract_disk(uint64_t bitboard) {
    return bitboard & (uint64_t) (- (int64_t) bitboard);
}

uint64_t c_resolve_move(uint64_t player, uint64_t opp, uint64_t new_disk) {
    uint64_t msk = 0;
    msk |= soutOccl(player, opp) & nortOccl(new_disk, opp);
    msk |= nortOccl(player, opp) & soutOccl(new_disk, opp);
    msk |= eastOccl(player, opp) & westOccl(new_disk, opp);
    msk |= westOccl(player, opp) & eastOccl(new_disk, opp);
    msk |= soWeOccl(player, opp) & noEaOccl(new_disk, opp);
    msk |= noEaOccl(player, opp) & soWeOccl(new_disk, opp);
    msk |= soEaOccl(player, opp) & noWeOccl(new_disk, opp);
    msk |= noWeOccl(player, opp) & soEaOccl(new_disk, opp);
    return msk;
}

uint64_t c_make_singleton_bitboard(unsigned int x, unsigned int y) {
    return 1ULL << ((7 - y) * 8 + (7 - x));
}

void print_board(uint64_t white, uint64_t black) {
    uint64_t mask;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            mask = 1ULL << ((7-y) * 8 + (7-x));

            if (mask & white) {
                printf("O");
            } else if (mask & black) {
                printf("X");
            } else {
                printf("-");
            }
        }
        printf("\n");
    }
}


/*
 * Top-level bitboard operations
 */
// Our pieces, opponent pieces
uint64_t c_find_moves(uint64_t gen, uint64_t pro) {
    uint64_t moves = 0;
    uint64_t empty = ~(gen | pro);
    uint64_t tmp;

    tmp = soutOccl(gen, pro);
    tmp &= pro;
    moves |= (tmp >> 8U) & empty; // NOLINT

    tmp = nortOccl(gen, pro);
    tmp &= pro;
    moves |= (tmp << 8U) & empty; // NOLINT

    tmp = eastOccl(gen, pro);
    tmp &= pro;
    moves |= (tmp << 1U) & _notAFile & empty;

    tmp = westOccl(gen, pro);
    tmp &= pro;
    moves |= (tmp >> 1U) & _notHFile & empty;

    tmp = soEaOccl(gen, pro);
    tmp &= pro;
    moves |= (tmp >> 7U) & _notAFile & empty; // NOLINT

    tmp = soWeOccl(gen, pro);
    tmp &= pro;
    moves |= (tmp >> 9U) & _notHFile & empty; // NOLINT

    tmp = noEaOccl(gen, pro);
    tmp &= pro;
    moves |= (tmp << 9U) & _notAFile & empty; // NOLINT

    tmp = noWeOccl(gen, pro);
    tmp &= pro;
    moves |= (tmp << 7U) & _notHFile & empty; // NOLINT

    return moves;
}

uint64_t c_stability(uint64_t player, uint64_t opp) {
    const uint64_t top = 255ULL;
    const uint64_t bot = 18374686479671623680ULL;
    const uint64_t lft = 72340172838076673ULL;
    const uint64_t rht = 9259542123273814144ULL;

    uint64_t pcs = player | opp;

    uint64_t vrt = nortOccl(bot & pcs, pcs) & soutOccl(top & pcs, pcs);
    uint64_t hrz = eastOccl(lft & pcs, pcs) & westOccl(rht & pcs, pcs);
    uint64_t dg1 = noEaOccl((bot|lft) & pcs, pcs) &
        soWeOccl((top|rht) & pcs, pcs);
    uint64_t dg2 = noWeOccl((bot|rht) & pcs, pcs) &
        soEaOccl((top|lft) & pcs, pcs);

    uint64_t stb = (0x8100000000000081ULL | (vrt & hrz & dg1 & dg2)) & // NOLINT
        player;

    //expand stable areas. At most 16 iterations necessary to reach from one
    //corner to the other
    for (size_t i = 0; i < 16; i++) { // NOLINT
        stb |= player & (
            (nort_one(stb) | sout_one(stb) | vrt) &
            (east_one(stb) | west_one(stb) | hrz) &
            (noea_one(stb) | sowe_one(stb) | dg1) &
            (nowe_one(stb) | soea_one(stb) | dg2)
        );
    }

    return stb;
}
