#include "search.h"
#include "eval.h"
#include "movegen.h"
#include "transposition.h"
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstring>

static constexpr int INF = 1'000'000;
static constexpr int MATE_SCORE = 900'000;
static constexpr int MATE_THRESHOLD = MATE_SCORE - 500; // anything above this is considered a Mate score
std::atomic<bool> Search::stop = false;

static TranspositionTable TT;
static Move killerMoves[64][2]; // two killer moves per ply
static int history[SQUARE_NB][SQUARE_NB]; // [from][to] history heuristic scores

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;
static Clock::time_point searchStartTime;
static int timeLimitMs = 0;

static int allocateTime(const Search::Limits& limit, Color sideToMove) {
    if (limit.infinite || limit.depth < 64) return 0; // no clock pressure
    if (limit.movetime > 0) return limit.movetime - 20; //20ms safety buffer

    int remainingTime = (sideToMove == WHITE) ? limit.wtime : limit.btime;
    int increment = (sideToMove == WHITE) ? limit.winc : limit.binc;
    int movesToGo = (limit.movestogo > 0) ? limit.movestogo : 20; // assume 20 moves to next time control if unknown
    return remainingTime / movesToGo + int(increment * 0.8);
}

static void orderMoves(std::vector<Move>& moves, const Board& board, Move ttMove, int ply) {
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

    std::sort(moves.begin(), moves.end(), [&](Move a, Move b) {
        return score(a) > score(b);
    });
};

static int quiescence(Board& board, int alpha, int beta) {
    const int standPat = Eval::evaluate(board);
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;

    std::vector<Move> moves;
    MoveGen::generateLegalMoves(board, moves);
    // Keep captures and promotions
    moves.erase(std::remove_if(moves.begin(), moves.end(), [&](const Move m) {return board.pieceAt(m.to()) == NO_PIECE && !m.isPromotion() && !m.isEnPassant();}), moves.end());
    orderMoves(moves, board, Move::none(), 0);

    for (const Move move : moves) {
        board.makeMove(move);
        const int score = -quiescence(board, -beta, -alpha);
        board.undoMove(move);

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

static uint64_t nodesSearched = 0;

static bool shouldStop() {
    if (Search::stop.load(std::memory_order_relaxed)) return true;
    if (timeLimitMs > 0 && nodesSearched % 4096 == 0) { // check time ever 4096 nodes
        auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - searchStartTime).count();
        if (elapsed >= timeLimitMs) return true;
    }
    return false;
}
static int negamax(Board& board, int depth, int ply, int alpha, int beta, Move& bestMove) {
    ++nodesSearched;
    if (shouldStop()) return 0;

    if (depth == 0) return quiescence(board, alpha, beta);

    Move ttMove = Move::none();
    const TTEntry* entry = TT.probe(board.hash());
    if (entry && entry->depth >= depth && ply > 0) {
        const int ttScore = scoreFromTT(entry->score, ply);
        if (entry->flag == TT_EXACT) return ttScore;
        if (entry->flag == TT_LOWER && ttScore >= beta) return ttScore;
        if (entry->flag == TT_UPPER && ttScore <= alpha) return ttScore;
    }
    if (entry) ttMove = entry->move;

    std::vector<Move> moves;
    MoveGen::generateLegalMoves(board, moves);
    
    if (moves.empty()) {
        // No legal moves: checkmate or stalemate
        if (MoveGen::isInCheck(board, board.sideToMove()))
            return -(MATE_SCORE - ply);  // prefer shorter mates
        return 0;              // stalemate
    }

    orderMoves(moves, board, ttMove, ply);

    const int originalAlpha = alpha;
    Move localBestMove = Move::none();
    for (const Move move : moves) {
        board.makeMove(move);
        const int score = -negamax(board, depth - 1, ply + 1, -beta, -alpha, localBestMove);
        board.undoMove(move);

        if (score >= beta) { // Only quiet moves teach us something useful; captures are already ordered by MVV-LVA
            if (board.pieceAt(move.to()) == NO_PIECE && !move.isEnPassant()) {
                if (move.data != killerMoves[ply][0].data) {
                    killerMoves[ply][1] = killerMoves[ply][0];
                    killerMoves[ply][0] = move;
                }
                history[move.from()][move.to()] += depth * depth; // reward deeper cutoffs more
            }
            return beta;
        }
        if (score > alpha) {
            alpha = score;
            localBestMove = move;
        }
    }

    TTFlag flag;
    if (alpha <= originalAlpha) flag = TT_UPPER; // no move improved alpha
    else if (alpha >= beta) flag = TT_LOWER; // move caused beta cutoff
    else flag = TT_EXACT; // exact score
    TT.store(board.hash(), scoreToTT(alpha, ply), localBestMove, depth, flag);
    bestMove = localBestMove;
    return alpha;
}

Search::SearchResult Search::search(Board& board, const Limits& limits) {
    nodesSearched = 0;
    searchStartTime = Clock::now();
    timeLimitMs = allocateTime(limits, board.sideToMove());
    std::memset(killerMoves, 0, sizeof(killerMoves));
    std::memset(history, 0, sizeof(history));

    SearchResult result{Move::none(), 0, 0};
    int maxDepth = (limits.depth < 64) ? limits.depth : 64;
    
    for (int depth = 1; depth <= maxDepth && !stop; ++depth) {
        Move bestMove = Move::none();
        const int score = negamax(board, depth, 0, -INF, INF, bestMove);

        if(stop && bestMove.isNone()) break; // if stopped and no move found, return last result instead of "best move none"
        
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