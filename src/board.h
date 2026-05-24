#ifndef BOARD_H
#define BOARD_H

#include "bitboard.h"
#include "types.h"
#include "zobrist.h"
#include <array>
#include <string>
#include <vector>

struct UndoInfo {
    Piece captured;
    Square enPassantSquare;
    int castlingRights;
    int halfmoveClock;
    uint64_t hash;
};
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
        std::vector<UndoInfo> history_;

    public:
        Board();
        ~Board() = default;

        void loadFEN(const std::string& fen);
        std::string toFEN() const;
        uint64_t hash() const { return hash_; }

        Color sideToMove() const { return whiteToMove_ ? WHITE : BLACK; }
        void makeMove(Move move);
        void undoMove(Move move);
        bool canCastle(CastlingRights rights) const;
    
    private:
        uint64_t computeHash() const;
        void putPiece(Square sq, Piece p);
        void removePiece(Square sq);

};

#endif // #ifndef BOARD_H