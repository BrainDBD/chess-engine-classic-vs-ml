#ifndef BOARD_H
#define BOARD_H

#include "bitboard.h"
#include "types.h"
#include "zobrist.h"
#include <array>
#include <string>

class Board {
    private:
        Bitboard bb_[COLOR_NB][PIECE_TYPE_NB] = {}; // [color][pieceType]
        std::array<Piece, SQUARE_NB> board_;
        bool whiteToMove_;
        int castlingRights_;
        Square enPassantSquare_;
        int halfmoveClock_;
        int fullmoveNumber_;
        uint64_t hash_;

    public:
        Board();
        ~Board() = default;

        void loadFEN(const std::string& fen);
        std::string toFEN() const;
        uint64_t hash() const { return hash_; }

        void makeMove();
        void undoMove();
        bool canCastle(CastlingRights rights) const;
    
    private:
        uint64_t computeHash() const;

};

#endif // #ifndef BOARD_H