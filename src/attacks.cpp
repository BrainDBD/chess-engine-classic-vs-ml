#include "attacks.h"

namespace Attacks
{
    static constexpr Bitboard NOT_A_FILE = 0xfefefefefefefefeULL;
    static constexpr Bitboard NOT_H_FILE = 0x7f7f7f7f7f7f7f7fULL;
    static constexpr Bitboard NOT_AB_FILE = 0xfcfcfcfcfcfcfcfcULL;
    static constexpr Bitboard NOT_GH_FILE = 0x3f3f3f3f3f3f3f3fULL;

    Bitboard knightAttacks[SQUARE_NB];
    Bitboard kingAttacks[SQUARE_NB];
    Bitboard pawnAttacks[COLOR_NB][SQUARE_NB];

    void init()
    {
        for (int sq = 0; sq < SQUARE_NB; ++sq)
        {

            Bitboard board = 1ULL << sq;

            knightAttacks[sq] = ((board & NOT_H_FILE) << 17) | ((board & NOT_A_FILE) << 15) | ((board & NOT_GH_FILE) << 10) | ((board & NOT_AB_FILE) << 6) | ((board & NOT_GH_FILE) >> 6) | ((board & NOT_AB_FILE) >> 10) | ((board & NOT_H_FILE) >> 15) | ((board & NOT_A_FILE) >> 17);
            kingAttacks[sq] = ((board & NOT_H_FILE) << 9) | ((board & NOT_A_FILE) << 7) | (board << 8) | ((board & NOT_H_FILE) << 1) | ((board & NOT_A_FILE) >> 1) | ((board & NOT_A_FILE) >> 9) | ((board & NOT_H_FILE) >> 7) | (board >> 8);
            pawnAttacks[WHITE][sq] = ((board & NOT_H_FILE) << 9) | ((board & NOT_A_FILE) << 7);
            pawnAttacks[BLACK][sq] = ((board & NOT_A_FILE) >> 9) | ((board & NOT_H_FILE) >> 7);
        }
    }

    Bitboard bishopAttacks(Square sq, Bitboard occ)
    {
        Bitboard attacks = 0;
        int rank = sq / 8, file = sq % 8;
        for (int toRank = rank + 1, toFile = file + 1; toRank < 8 && toFile < 8; ++toRank, ++toFile)
        {
            int toSq = toRank * 8 + toFile;
            attacks |= 1ULL << toSq;
            if (occ & (1ULL << toSq))
                break;
        }
        for (int toRank = rank + 1, toFile = file - 1; toRank < 8 && toFile >= 0; ++toRank, --toFile)
        {
            int toSq = toRank * 8 + toFile;
            attacks |= 1ULL << toSq;
            if (occ & (1ULL << toSq))
                break;
        }
        for (int toRank = rank - 1, toFile = file + 1; toRank >= 0 && toFile < 8; --toRank, ++toFile)
        {
            int toSq = toRank * 8 + toFile;
            attacks |= 1ULL << toSq;
            if (occ & (1ULL << toSq))
                break;
        }
        for (int toRank = rank - 1, toFile = file - 1; toRank >= 0 && toFile >= 0; --toRank, --toFile)
        {
            int toSq = toRank * 8 + toFile;
            attacks |= 1ULL << toSq;
            if (occ & (1ULL << toSq))
                break;
        }
        return attacks;
    }

    Bitboard rookAttacks(Square sq, Bitboard occ)
    {
        Bitboard attacks = 0;
        int rank = sq / 8, file = sq % 8;
        for (int toRank = rank + 1; toRank < 8; ++toRank)
        {
            int toSq = toRank * 8 + file;
            attacks |= 1ULL << toSq;
            if (occ & (1ULL << toSq))
                break;
        }
        for (int toRank = rank - 1; toRank >= 0; --toRank)
        {
            int toSq = toRank * 8 + file;
            attacks |= 1ULL << toSq;
            if (occ & (1ULL << toSq))
                break;
        }
        for (int toFile = file + 1; toFile < 8; ++toFile)
        {
            int toSq = rank * 8 + toFile;
            attacks |= 1ULL << toSq;
            if (occ & (1ULL << toSq))
                break;
        }
        for (int toFile = file - 1; toFile >= 0; --toFile)
        {
            int toSq = rank * 8 + toFile;
            attacks |= 1ULL << toSq;
            if (occ & (1ULL << toSq))
                break;
        }
        return attacks;
    }

    Bitboard queenAttacks(Square sq, Bitboard occ)
    {
        return bishopAttacks(sq, occ) | rookAttacks(sq, occ);
    }
} // namespace Attacks