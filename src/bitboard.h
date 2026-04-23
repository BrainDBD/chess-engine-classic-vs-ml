#pragma once
#include <cstdint>
#include <iostream>

using Bitboard = uint64_t;

namespace BitboardUtils {
    inline bool getBit(Bitboard board, int square) {
        return (board >> square) & 1ULL;
    }

    inline void setBit(Bitboard& board, int square) {
        board |= (1ULL << square);
    }

    inline void clearBit(Bitboard& board, int square) {
        board &= ~(1ULL << square);
    }

    inline void toggleBit(Bitboard& board, int square) {
        board ^= (1ULL << square);
    }

    inline int countBits(Bitboard board) {
        return __builtin_popcountll(board);
    }

    void printBitboard(Bitboard board) {
        for (int rank = 7; rank >= 0; --rank) {
            for (int file = 0; file < 8; ++file) {
                int square = rank * 8 + file;
                std::cout << (getBit(board, square) ? "1 " : ". ");
            }
            std::cout << std::endl;
        }
    }
}