#pragma once
#include "board.h"

namespace MoveGen {
    Move* generateMoves(const Board& board, Move* moves);
    Move* generateLegalMoves(Board& board, Move* moves);
    bool isSquareAttacked(const Board& board, Square sq, Color by);
    bool isInCheck(const Board& board, Color c);
}