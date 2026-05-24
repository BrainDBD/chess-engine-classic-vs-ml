#pragma once
#include "board.h"

namespace Search {
    struct SearchResult {
        Move bestMove;
        int score;
        int depth;
    };

    SearchResult search(Board& board, int maxDepth);
}