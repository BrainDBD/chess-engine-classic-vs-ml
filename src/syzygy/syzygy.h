#pragma once
#include "board.h"
namespace Syzygy {
    struct RootProbeResult {
        bool ok = false;
        Move move = Move::none();
        int score = 0;
        int wdl = 0;   // -2 loss, -1 blessed loss, 0 draw, +1 cursed win, +2 win
        int dtz = 0;
    };
    bool init(const char* paths);   
    void shutdown();             
    bool isReady();   
    int  maxPieces();
    bool canProbe(const Board& board);
    bool probeWDL(const Board& board, int& wdl);
    bool probeRoot(Board& board, RootProbeResult& out);
}