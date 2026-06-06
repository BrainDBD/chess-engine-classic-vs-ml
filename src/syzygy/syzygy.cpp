#include "syzygy.h"
#include "tbprobe.h"
#include "bitboard.h"
#include "types.h"

namespace Syzygy {

    bool init(const char* paths) { return tb_init(paths); }
    void shutdown()              { tb_free(); }
    int  maxPieces()             { return (int)TB_LARGEST; }

    bool probeWDL(const Board& board, int& wdl) {
        // Fathom takes plain bitboards + flags, not board objects
        unsigned res = tb_probe_wdl(
            board.pieces(WHITE), board.pieces(BLACK),
            board.pieces(WHITE, KING)   | board.pieces(BLACK, KING),
            board.pieces(WHITE, QUEEN)  | board.pieces(BLACK, QUEEN),
            board.pieces(WHITE, ROOK)   | board.pieces(BLACK, ROOK),
            board.pieces(WHITE, BISHOP) | board.pieces(BLACK, BISHOP),
            board.pieces(WHITE, KNIGHT) | board.pieces(BLACK, KNIGHT),
            board.pieces(WHITE, PAWN)   | board.pieces(BLACK, PAWN),
            0,   // rule50  -- Fathom ignores it (returns FAILED if nonzero)
            0,   // castling -- MUST be 0; Fathom returns FAILED otherwise
            board.enpassantSquare() == SQ_NONE ? 0u
                                               : (unsigned)board.enpassantSquare(),
            board.sideToMove() == WHITE);   // turn: true = white to move
        if (res == TB_RESULT_FAILED) return false;
        // TB_LOSS=0, BLESSED_LOSS=1, DRAW=2, CURSED_WIN=3, WIN=4 -> map to -2..+2
        wdl = (int)TB_GET_WDL(res) - 2;
        return true;
    }
}