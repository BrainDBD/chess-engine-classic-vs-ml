#pragma once
#include "types.h"

namespace Zobrist {
    extern uint64_t piece    [COLOR_NB][PIECE_TYPE_NB][SQUARE_NB];
    extern uint64_t sideToMove;
    extern uint64_t castling [16];       // indexed by raw castling rights (4 bits -> 16 combos)
    extern uint64_t enPassant[FILE_NB];  // indexed by file of the en passant square

    void init();
}
