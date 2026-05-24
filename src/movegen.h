#pragma once
#include "board.h"
#include <vector>

namespace MoveGen {
    void generateMoves(const Board& board, std::vector<Move>& moves);
    bool isSquareAttacked(const Board& board, Square sq, Color by);
    bool isInCheck(const Board& board, Color c);
    void generateLegalMoves(Board& board, std::vector<Move>& moves);
}