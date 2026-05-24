#include "attacks.h"

namespace Attacks
{
    static constexpr Bitboard NOT_A_FILE = 0xfefefefefefefefeULL;
    static constexpr Bitboard NOT_H_FILE = 0x7f7f7f7f7f7f7f7fULL;
    static constexpr Bitboard NOT_AB_FILE = 0xfcfcfcfcfcfcfcfcULL;
    static constexpr Bitboard NOT_GH_FILE = 0x3f3f3f3f3f3f3f3fULL;

    Bitboard knight[SQUARE_NB];
    Bitboard king[SQUARE_NB];
    Bitboard pawn[COLOR_NB][SQUARE_NB];

    void init()
    {
        for (int sq = 0; sq < SQUARE_NB; ++sq)
        {

            Bitboard b = 1ULL << sq;

            knight[sq] = ((b & NOT_H_FILE) << 17) | ((b & NOT_A_FILE) << 15) | ((b & NOT_GH_FILE) << 10) | ((b & NOT_AB_FILE) << 6) | ((b & NOT_GH_FILE) >> 6) | ((b & NOT_AB_FILE) >> 10) | ((b & NOT_H_FILE) >> 15) | ((b & NOT_A_FILE) >> 17);

            king[sq] = ((b & NOT_H_FILE) << 9) | ((b & NOT_A_FILE) << 7) | (b << 8) | ((b & NOT_H_FILE) << 1) | ((b & NOT_A_FILE) >> 1) | ((b & NOT_A_FILE) >> 9) | ((b & NOT_H_FILE) >> 7) | (b >> 8);

            pawn[WHITE][sq] = ((b & NOT_H_FILE) << 9) | ((b & NOT_A_FILE) << 7);
            pawn[BLACK][sq] = ((b & NOT_A_FILE) >> 9) | ((b & NOT_H_FILE) >> 7);
        }
    }

    Bitboard bishop(Square sq, Bitboard occ)
    {
        Bitboard attacks = 0;
        int r = sq / 8, f = sq % 8;
        for (int rr = r + 1, ff = f + 1; rr < 8 && ff < 8; ++rr, ++ff)
        {
            int s = rr * 8 + ff;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s))
                break;
        }
        for (int rr = r + 1, ff = f - 1; rr < 8 && ff >= 0; ++rr, --ff)
        {
            int s = rr * 8 + ff;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s))
                break;
        }
        for (int rr = r - 1, ff = f + 1; rr >= 0 && ff < 8; --rr, ++ff)
        {
            int s = rr * 8 + ff;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s))
                break;
        }
        for (int rr = r - 1, ff = f - 1; rr >= 0 && ff >= 0; --rr, --ff)
        {
            int s = rr * 8 + ff;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s))
                break;
        }
        return attacks;
    }

    Bitboard rook(Square sq, Bitboard occ)
    {
        Bitboard attacks = 0;
        int r = sq / 8, f = sq % 8;
        for (int rr = r + 1; rr < 8; ++rr)
        {
            int s = rr * 8 + f;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s))
                break;
        }
        for (int rr = r - 1; rr >= 0; --rr)
        {
            int s = rr * 8 + f;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s))
                break;
        }
        for (int ff = f + 1; ff < 8; ++ff)
        {
            int s = r * 8 + ff;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s))
                break;
        }
        for (int ff = f - 1; ff >= 0; --ff)
        {
            int s = r * 8 + ff;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s))
                break;
        }
        return attacks;
    }

    Bitboard queen(Square sq, Bitboard occ)
    {
        return bishop(sq, occ) | rook(sq, occ);
    }
} // namespace Attacks