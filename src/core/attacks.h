#pragma once
#include "bitboard.h"
#include "types.h"

namespace Attacks {
    extern Bitboard knightAttacks[SQUARE_NB];
    extern Bitboard kingAttacks[SQUARE_NB];
    extern Bitboard pawnAttacks[COLOR_NB][SQUARE_NB];

    void init();

    Bitboard bishopAttacks(Square sq, Bitboard occupancy);
    Bitboard rookAttacks(Square sq, Bitboard occupancy);
    Bitboard queenAttacks(Square sq, Bitboard occupancy);
}