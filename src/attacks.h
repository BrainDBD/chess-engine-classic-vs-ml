#pragma once
#include "bitboard.h"
#include "types.h"

namespace Attacks {
    extern Bitboard knight[SQUARE_NB];
    extern Bitboard king[SQUARE_NB];
    extern Bitboard pawn[COLOR_NB][SQUARE_NB];

    void init();

    Bitboard bishop(Square sq, Bitboard occupancy);
    Bitboard rook(Square sq, Bitboard occupancy);
    Bitboard queen(Square sq, Bitboard occupancy);
}