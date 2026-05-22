#include "zobrist.h"
#include <random>

namespace Zobrist {
    uint64_t piece    [COLOR_NB][PIECE_TYPE_NB][SQUARE_NB];
    uint64_t sideToMove;
    uint64_t castling [16];
    uint64_t enPassant[FILE_NB];

    void init() {
        std::mt19937_64 rng(1070372ULL);

        for (int c  = 0; c  < COLOR_NB;      ++c)
        for (int pt = 0; pt < PIECE_TYPE_NB;  ++pt)
        for (int sq = 0; sq < SQUARE_NB;      ++sq)
            piece[c][pt][sq] = rng();

        sideToMove = rng();

        for (int i = 0; i < 16;      ++i) castling [i] = rng();
        for (int f = 0; f < FILE_NB; ++f) enPassant[f] = rng();
    }
}
