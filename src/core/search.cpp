#include "search.h"
#include "eval.h"
#include "movegen.h"
#include "transposition.h"
#include "syzygy.h"
#include "endgame_mode.h"
#include "endgame_net.h"
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstring>

static constexpr int INF = 1'000'000;
static constexpr int MATE_SCORE = 900'000;
static constexpr int MATE_THRESHOLD = MATE_SCORE - 500; // anything above this is considered a Mate score
static constexpr int TB_WIN_SCORE = MATE_SCORE - 1000; // Syzygy tablebase win
static constexpr int QUIESCENCE_DELTA_MARGIN = 900; // margin for delta pruning in quiescence search
static constexpr int RFP_MARGIN = 100; // per-depth step for reverse futility pruning
static constexpr int ASPIRATION_DELTA = 25; // initial aspiration window size in centipawns
static constexpr int ASPIRATION_MIN_DEPTH = 4; // minimum depth to apply aspiration search

static constexpr int MAX_PLY = 256;
static TranspositionTable TT;
static Move killerMoves[MAX_PLY][2]; // two killer moves per ply
static int history[SQUARE_NB][SQUARE_NB]; // [from][to] history heuristic scores
static constexpr int HISTORY_MAX = 2048;

std::atomic<bool> Search::stop = false;
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;
static Clock::time_point searchStartTime;
static int timeLimitMs = 0;

static int allocateTime(const Search::Limits& limit, Color sideToMove) {
    if (limit.infinite || limit.depth < 64) return 0; // no clock pressure
    if (limit.movetime > 0) return std::max(1, limit.movetime - 20); //20ms safety buffer

    int remainingTime = (sideToMove == WHITE) ? limit.wtime : limit.btime;
    if (remainingTime <= 0) return 50; // emergency fallback: 50ms
    int increment = (sideToMove == WHITE) ? limit.winc : limit.binc;
    int movesToGo = (limit.movestogo > 0) ? limit.movestogo : 20; // assume 20 moves to next time control if unknown
    return remainingTime / movesToGo + int(increment * 0.8);
}

static uint64_t nodesSearched = 0;

static bool shouldStop() {
    if (Search::stop.load(std::memory_order_relaxed)) return true;
    if (timeLimitMs > 0 && (nodesSearched & 4095) == 0) {
        auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - searchStartTime).count();
        if (elapsed >= timeLimitMs) {
            Search::stop = true;
            return true;
        }
    }
    return false;
}

static void orderMoves(Move* moves, int count, const Board& board, Move ttMove, int ply) {
    auto score = [&](const Move move) -> int {
        if (move.data == ttMove.data) return 20'000; // search PV first
        if (move.isPromotion()) return 10'000 + static_cast<int>(move.promotionType());

        const Piece takenPiece = board.pieceAt(move.to());
        if (takenPiece != NO_PIECE) {
            static constexpr int TAKEN_VALUE[PIECE_TYPE_NB] = {100, 200, 300, 500, 900, 0};
            static constexpr int TAKER_VALUE[PIECE_TYPE_NB] = {1, 2, 3, 4, 5, 6};
            return 5'000 + TAKEN_VALUE[typeOf(takenPiece)] - TAKER_VALUE[typeOf(board.pieceAt(move.from()))];
        }

        if (move.data == killerMoves[ply][0].data) return 4'000;
        if (move.data == killerMoves[ply][1].data) return 3'000;
        return history[move.from()][move.to()];
    };

    std::sort(moves, moves + count, [&](Move a, Move b) {
        return score(a) > score(b);
    });
};

static int quiescence(Board& board, int alpha, int beta, int ply) {
    nodesSearched++;

    if (ply >= MAX_PLY - 1)
        return Eval::evaluate(board);

    if (shouldStop()) return 0;

    const bool inCheck = MoveGen::isInCheck(board, board.sideToMove());
    if (!inCheck) {
        const int standPat = Eval::evaluate(board);
        if (standPat >= beta) return beta;
        if (standPat + QUIESCENCE_DELTA_MARGIN < alpha) return alpha; // delta pruning
        if (standPat > alpha) alpha = standPat;
    }

    Move moves[256];
    Move* end = MoveGen::generateLegalMoves(board, moves);
    // Keep captures and promotions
    Move* searchEnd = end;
    if (!inCheck) { // Out of check, only consider captures and promotions for quiescence search
        Move* captureEnd = moves;
        for (Move* move = moves; move != end; ++move)
                if (board.pieceAt(move->to()) != NO_PIECE || move->isPromotion() || move->isEnPassant())
                    *captureEnd++ = *move;
        searchEnd = captureEnd;
    }
    const int numCaptures = static_cast<int>(searchEnd - moves);

    if (inCheck && numCaptures == 0) {
        // Checkmate found in quiescence search — return a large negative score adjusted for mate distance
        return -(MATE_SCORE - ply);
    }
    orderMoves(moves, numCaptures, board, Move::none(), 0); // ply 0: avoids killer OOB

    for (int i = 0; i < numCaptures; ++i) {
        board.makeMove(moves[i]);
        const int score = -quiescence(board, -beta, -alpha, ply + 1);
        board.undoMove(moves[i]);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

static int scoreToTT(int score, int ply) {
    if (score > MATE_THRESHOLD) return score + ply; // prefer shorter mates
    if (score < -MATE_THRESHOLD) return score - ply; // prefer longer losses
    return score;
}

static int scoreFromTT(int score, int ply) {
    if (score > MATE_THRESHOLD) return score - ply;
    if (score < -MATE_THRESHOLD) return score + ply;
    return score;
}

void Search::clearTT() {
    TT.clear();
    std::memset(killerMoves, 0, sizeof(killerMoves));
    std::memset(history, 0, sizeof(history));
}

static int negamax(Board& board, int depth, int ply, int alpha, int beta, Move& bestMove, bool allowNull = true) {
    if (ply > 0 && board.isRepetition()) return 0; // draw by repetition
    if (board.halfmoveClock() >= 100) return 0; // draw by 50-move rule

    // In-search endgame verdict given by real Syzygy probing or MLP approximation
    if (ply > 0) {
        const int pieces = BitboardUtils::countBits(board.occupancy());
        if (Endgame::g_mode == Endgame::Mode::Syzygy &&
            board.castlingRights() == 0 && pieces <= Syzygy::maxPieces()) {
            int wdl;
            if (Syzygy::probeWDL(board, wdl)) {
                if (wdl == 2) return  (TB_WIN_SCORE - ply); // prefer shorter wins
                if (wdl == -2) return -(TB_WIN_SCORE - ply); // prefer longer losses
                return 0;                                   // draw (incl. cursed/blessed)
            }
        } else if (Endgame::g_mode == Endgame::Mode::MLP && EndgameNet::canProbe(board)) {
            const int v = EndgameNet::wdlVerdict(board);     // learned 3-way verdict
            if (v > 0) return  (TB_WIN_SCORE - ply);
            if (v < 0) return -(TB_WIN_SCORE - ply);
            return 0;
        }
    }

    if (depth == 0 || ply >= MAX_PLY - 1) return quiescence(board, alpha, beta, ply);
    ++nodesSearched;
    if (shouldStop()) return 0;

    //Mate distance pruning
    alpha = std::max(alpha, -(MATE_SCORE - ply));
    beta = std::min(beta, MATE_SCORE - ply - 1);
    if (alpha >= beta) return alpha;

    Move ttMove = Move::none();
    const TTEntry* entry = TT.probe(board.hash());
    if (entry && entry->depth >= depth && ply > 0) {
        const int ttScore = scoreFromTT(entry->score, ply);
        if (entry->flag == TT_EXACT) return ttScore;
        if (entry->flag == TT_LOWER && ttScore >= beta) return ttScore;
        if (entry->flag == TT_UPPER && ttScore <= alpha) return ttScore;
    }
    if (entry) {
        ttMove = entry->move;
        if(!ttMove.isNone() && board.pieceAt((ttMove.from())) == NO_PIECE)
            ttMove = Move::none(); // hash collision — from-square is empty in this position
    }

    const bool inCheck = MoveGen::isInCheck(board, board.sideToMove());

    //Reverse futility pruning
    if (depth <= 3 && !inCheck && ply > 0 && beta < MATE_THRESHOLD) {
        const int staticEval = Eval::evaluate(board);
        if (staticEval - RFP_MARGIN * depth >= beta)
            return beta;
    }

    //Null move pruning
    if (allowNull && !inCheck && depth >= 3 && ply > 0 && beta < MATE_THRESHOLD) {
        const bool hasNonPawns = (
            board.pieces(board.sideToMove(), KNIGHT) |
            board.pieces(board.sideToMove(), BISHOP) |
            board.pieces(board.sideToMove(), ROOK)   |
            board.pieces(board.sideToMove(), QUEEN)
        ) != 0;

        if (hasNonPawns) {
            const int R = 2 + depth / 4; // reduction based on depth
            board.makeNullMove();
            Move dummyMove = Move::none();
            const int nullScore = -negamax(board, depth - R - 1, ply + 1, -beta, -beta + 1, dummyMove, false);
            board.undoNullMove();
            if (!shouldStop() && nullScore >= beta)
                return beta; // fail-hard beta cutoff
        }
    }

    Move moves[256];
    Move* end = MoveGen::generateLegalMoves(board, moves);
    const int moveCount = static_cast<int>(end - moves);
    
    if (moveCount == 0) {
        // No legal moves: checkmate or stalemate
        if (inCheck)
            return -(MATE_SCORE - ply);  // prefer shorter mates
        return 0;              // stalemate
    }

    orderMoves(moves, moveCount, board, ttMove, ply);

    const int originalAlpha = alpha;
    Move localBestMove = Move::none();
    for (int i = 0; i < moveCount; ++i) {
        const Move move = moves[i];
        const bool isQuiet = board.pieceAt(move.to()) == NO_PIECE && !move.isEnPassant() && !move.isPromotion();
        board.makeMove(move);
        const bool givesCheck = MoveGen::isInCheck(board, board.sideToMove());
        const int extension = givesCheck ? 1 : 0; // check extension
        Move childBestMove = Move::none();
        const bool doLMR = depth >= 3 && i >= 4 && isQuiet && !inCheck && !givesCheck; // late move reduction
        const int searchDepth = (doLMR ? depth - 2 : depth - 1) + extension;
        
        int score = -negamax(board, searchDepth, ply + 1, -beta, -alpha, childBestMove);
        if (doLMR && score > alpha && score < beta)
            score = -negamax(board, depth - 1, ply + 1, -beta, -alpha, childBestMove); // re-search at full depth if LMR move is better than alpha
        board.undoMove(move);

        if (shouldStop()) break; // move score is unreliable

        if (score >= beta) { // Only quiet moves teach us something useful; captures are already ordered by MVV-LVA
            if (!Search::stop && isQuiet) {
                const int safePly = std::min(ply, MAX_PLY - 1);
                if (safePly < MAX_PLY)
                    if (move.data != killerMoves[safePly][0].data) {
                        killerMoves[safePly][1] = killerMoves[safePly][0];
                        killerMoves[safePly][0] = move;
                    }
                int bonus = std::min(depth * depth, HISTORY_MAX); // reward deeper cutoffs
                int &historyScore = history[move.from()][move.to()];
                historyScore += bonus - historyScore * bonus / HISTORY_MAX; // gravity: asymptotically bounded to ±HISTORY_MAX
            }
            if(!Search::stop)
                TT.store(board.hash(), scoreToTT(beta, ply), move, depth, TT_LOWER);
            bestMove = move;
            return beta;
        }
        if (score > alpha) {
            alpha = score;
            localBestMove = move;
        }
    }

    if (!Search::stop) {
        const TTFlag flag = (alpha > originalAlpha) ? TT_EXACT : TT_UPPER;
        TT.store(board.hash(), scoreToTT(alpha, ply), localBestMove, depth, flag);
    }
    bestMove = localBestMove;
    return alpha;
}

Search::SearchResult Search::search(Board& board, const Limits& limits) {
    stop.store(false, std::memory_order_relaxed);
    nodesSearched = 0;
    searchStartTime = Clock::now();
    timeLimitMs = allocateTime(limits, board.sideToMove());
    std::memset(killerMoves, 0, sizeof(killerMoves));
    std::memset(history, 0, sizeof(history));

    // DTZ root probe (move selection via exact conversion distances). Orthogonal to the verdict source: available in both Syzygy and MLP modes, gated by probeDTZ
    // Turning it OFF forces the search to convert using only the in-search verdict, which is the matched condition for comparing the WDL table against the MLP as verdict sources
    if ((Endgame::g_mode == Endgame::Mode::Syzygy ||
        Endgame::g_mode == Endgame::Mode::MLP) && Endgame::probeDTZ) {
        Syzygy::RootProbeResult rp;
        if (Syzygy::probeRoot(board, rp) && rp.ok && !rp.move.isNone()) {
            std::cout << "info depth 1 score cp " << rp.score
                    << " nodes 1 time 0" << std::endl;
            std::cout.flush();
            return SearchResult{rp.move, rp.score, 1};
        }
    }

    SearchResult result{Move::none(), 0, 0};
    int maxDepth = (limits.depth < 64) ? limits.depth : 64;
    
    for (int depth = 1; depth <= maxDepth && !stop; ++depth) {
        for (auto& row : history)
                for (auto& h : row)
                    h >>= 2; // age history scores to avoid overvaluing old information
        
        int delta = ASPIRATION_DELTA;
        int alpha, beta;

        if (depth >= ASPIRATION_MIN_DEPTH && std::abs(result.score) < MATE_THRESHOLD) {
            alpha = std::max(result.score - delta, -MATE_SCORE);
            beta = std::min(result.score + delta, MATE_SCORE);
        } else {
            alpha = -INF;
            beta = INF;
        }

        Move bestMove = Move::none();
        int score = 0;

        while (true) { // aspiration search loop
            score = negamax(board, depth, 0, alpha, beta, bestMove);
            if(shouldStop()) break;

            if (score <= alpha) {
                alpha = std::max(alpha - delta, -INF); // widen window downwards
                delta += delta;
                bestMove = Move::none();
            } else if (score >= beta) {
                beta = std::min(beta + delta, INF); // widen window upwards
                delta += delta;
                bestMove = Move::none();
            } else break;
            if (delta >= INF / 4) { // collapse to full window
                alpha = -INF;
                beta = INF;
            }
        }

        if(stop) { // if stopped and no move found, return last result instead of "best move none"
            if (!bestMove.isNone()) result.bestMove = bestMove;
            break;
        }

        if(!bestMove.isNone()) {
            result.bestMove = bestMove;
            result.score = score;
            result.depth = depth;
        }

        auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - searchStartTime).count();
        std::cout << "info depth " << depth
                << " score cp " << score
                << " nodes " << nodesSearched
                << " time " << elapsed
                << std::endl;
        std::cout.flush();
        if (timeLimitMs > 0 && elapsed >= timeLimitMs / 2) break; // stop deepening if not enough time left
    }
    return result;
}