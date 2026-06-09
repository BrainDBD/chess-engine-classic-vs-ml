#include "syzygy.h"
#include "tbprobe.h"
#include "bitboard.h"
#include "types.h"
#include "movegen.h"

#include <algorithm>

namespace Syzygy {

    static bool g_ready = false;

    static constexpr int TB_WIN_SCORE = 899000;

    static unsigned epSquareForTB(const Board& board) {
        return board.enpassantSquare() == SQ_NONE
            ? 0u
            : static_cast<unsigned>(board.enpassantSquare());
    }

    static unsigned promotionToTB(PieceType pt) {
        switch (pt) {
            case QUEEN:  return TB_PROMOTES_QUEEN;
            case ROOK:   return TB_PROMOTES_ROOK;
            case BISHOP: return TB_PROMOTES_BISHOP;
            case KNIGHT: return TB_PROMOTES_KNIGHT;
            default:     return TB_PROMOTES_NONE;
        }
    }

    static PieceType tbPromotionToPieceType(unsigned promo) {
        switch (promo) {
            case TB_PROMOTES_QUEEN:  return QUEEN;
            case TB_PROMOTES_ROOK:   return ROOK;
            case TB_PROMOTES_BISHOP: return BISHOP;
            case TB_PROMOTES_KNIGHT: return KNIGHT;
            default:                 return QUEEN;
        }
    }

    static Move findLegalMove(Board& board, int from, int to, unsigned tbPromo) {
        Move moves[256];
        Move* end = MoveGen::generateLegalMoves(board, moves);

        for (Move* m = moves; m != end; ++m) {
            if (m->from() != from || m->to() != to)
                continue;

            if (!m->isPromotion()) {
                if (tbPromo == TB_PROMOTES_NONE)
                    return *m;
                continue;
            }

            if (tbPromo != TB_PROMOTES_NONE &&
                m->promotionType() == tbPromotionToPieceType(tbPromo))
                return *m;
        }

        return Move::none();
    }

    bool init(const char* paths) {
        bool ok = tb_init(paths);

        // tb_init can return true even if no files were found
        g_ready = ok && TB_LARGEST > 0;

        return g_ready;
    }

    void shutdown() {
        tb_free();
        g_ready = false;
    }

    bool isReady() {
        return g_ready && TB_LARGEST > 0;
    }

    int maxPieces() {
        return static_cast<int>(TB_LARGEST);
    }

    bool canProbe(const Board& board) {
        if (!isReady())
            return false;

        if (board.castlingRights() != 0)
            return false;

        if (BitboardUtils::countBits(board.occupancy()) > maxPieces())
            return false;

        if (BitboardUtils::countBits(board.pieces(WHITE, KING)) != 1)
            return false;

        if (BitboardUtils::countBits(board.pieces(BLACK, KING)) != 1)
            return false;

        return true;
    }

    bool probeWDL(const Board& board, int& wdl) {
        if (!canProbe(board))
            return false;

        unsigned res = tb_probe_wdl(
            board.pieces(WHITE),
            board.pieces(BLACK),
            board.pieces(WHITE, KING)   | board.pieces(BLACK, KING),
            board.pieces(WHITE, QUEEN)  | board.pieces(BLACK, QUEEN),
            board.pieces(WHITE, ROOK)   | board.pieces(BLACK, ROOK),
            board.pieces(WHITE, BISHOP) | board.pieces(BLACK, BISHOP),
            board.pieces(WHITE, KNIGHT) | board.pieces(BLACK, KNIGHT),
            board.pieces(WHITE, PAWN)   | board.pieces(BLACK, PAWN),
            0,                      // rule50 MUST be 0: tb_probe_wdl rejects nonzero
            0,                      // castling MUST be 0
            epSquareForTB(board),
            board.sideToMove() == WHITE
        );

        if (res == TB_RESULT_FAILED)
            return false;

        wdl = static_cast<int>(TB_GET_WDL(res)) - 2;
        return true;
    }

    bool probeRoot(Board& board, RootProbeResult& out) {
        out = RootProbeResult{};

        if (!canProbe(board))
            return false;

        unsigned results[TB_MAX_MOVES];

        unsigned res = tb_probe_root(
            board.pieces(WHITE),
            board.pieces(BLACK),
            board.pieces(WHITE, KING)   | board.pieces(BLACK, KING),
            board.pieces(WHITE, QUEEN)  | board.pieces(BLACK, QUEEN),
            board.pieces(WHITE, ROOK)   | board.pieces(BLACK, ROOK),
            board.pieces(WHITE, BISHOP) | board.pieces(BLACK, BISHOP),
            board.pieces(WHITE, KNIGHT) | board.pieces(BLACK, KNIGHT),
            board.pieces(WHITE, PAWN)   | board.pieces(BLACK, PAWN),
            board.halfmoveClock(),
            0,
            epSquareForTB(board),
            board.sideToMove() == WHITE,
            results
        );

        if (res == TB_RESULT_FAILED)
            return false;

        if (res == TB_RESULT_CHECKMATE) {
            out.ok = true;
            out.move = Move::none();
            out.score = TB_WIN_SCORE;
            out.wdl = 2;
            return true;
        }

        if (res == TB_RESULT_STALEMATE) {
            out.ok = true;
            out.move = Move::none();
            out.score = 0;
            out.wdl = 0;
            return true;
        }

        const int from = static_cast<int>(TB_GET_FROM(res));
        const int to = static_cast<int>(TB_GET_TO(res));
        const unsigned promo = TB_GET_PROMOTES(res);

        Move move = findLegalMove(board, from, to, promo);
        if (move.isNone())
            return false;

        const int wdl = static_cast<int>(TB_GET_WDL(res)) - 2;
        const int dtz = static_cast<int>(TB_GET_DTZ(res));

        int score = 0;
        if (wdl > 0) {
            score = TB_WIN_SCORE - std::min(dtz, 1000);
        } else if (wdl < 0) {
            score = -TB_WIN_SCORE + std::min(dtz, 1000);
        } else {
            score = 0;
        }

        out.ok = true;
        out.move = move;
        out.score = score;
        out.wdl = wdl;
        out.dtz = dtz;

        return true;
    }
}