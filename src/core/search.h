#pragma once
#include "board.h"
#include <atomic>
namespace Search {
    struct SearchResult {
        Move bestMove;
        int score;
        int depth;
    };

    struct Limits {
        int wtime = 0, btime = 0; // Remaining time for white and black in milliseconds
        int winc = 0, binc = 0;   // Increment per move for white and black in milliseconds
        int movestogo = 0;        // Number of moves until time control (if known)
        int movetime = 0;         // Time to spend on this move in milliseconds
        int depth = 64;           // Maximum search depth
        bool infinite = false;    // Search until stopped
    };

    extern std::atomic<bool> stop;

    void clearTT();
    SearchResult search(Board& board, const Limits& limits);
}