#pragma once
#include "board.h"
namespace Syzygy {
    bool init(const char* paths);   
    void shutdown();                
    int  maxPieces();
    bool probeWDL(const Board& board, int& wdl);
}