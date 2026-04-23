#ifndef BOARD_H
#define BOARD_H

#include "bitboard.h"
#include "types.h"
#include <array>
#include <string>

class Board {
    private:
        Bitboard bb_[COLOR_NB][PIECE_TYPE_NB] = {}; // [color][pieceType]
        std::array<Color,     SQUARE_NB> color_on_ = {};
        std::array<PieceType, SQUARE_NB> piece_on_ = {};
        bool whiteToMove;
        int castlingRights;

    public:
        Board();
        ~Board() = default;

        void loadFEN(const std::string& fen);
        std::string toFEN() const;

        void makeMove();
        void undoMove();
        bool canCastle(int rights) const;

};

#endif // #ifndef BOARD_H