#include "search.h"
#include "eval.h"
#include "movegen.h"
#include "transposition.h"
#include <algorithm>
#include <vector>

static TranspositionTable TT;
static constexpr int INF = 1'000'000;
static constexpr int MATE_SCORE = 900'000;
static constexpr int MATE_THRESHOLD = MATE_SCORE - 500; // anything above this is considered a Mate score

static void orderMoves(std::vector<Move>& moves, const Board& board, Move ttMove) {
    auto score = [&](const Move move) -> int {
        if (move.data == ttMove.data) return 20'000; // search PV first
        if (move.isPromotion()) return 10'000 + static_cast<int>(move.promotionType());

        const Piece takenPiece = board.pieceAt(move.to());
        if (takenPiece != NO_PIECE) {
            static constexpr int TAKEN_VALUE[PIECE_TYPE_NB] = {100, 200, 300, 500, 900, 0};
            static constexpr int TAKER_VALUE[PIECE_TYPE_NB] = {1, 2, 3, 4, 5, 6};
            return 5'000 + TAKEN_VALUE[typeOf(takenPiece)] - TAKER_VALUE[typeOf(board.pieceAt(move.from()))];
        }

        return 0; // quiet moves last
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
    orderMoves(moves, board, Move::none());

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

static int negamax(Board& board, int depth, int ply, int alpha, int beta, Move& bestMove) {
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

    orderMoves(moves, board, ttMove);

    const int originalAlpha = alpha;
    Move localBestMove = Move::none();
    for (const Move move : moves) {
        board.makeMove(move);
        const int score = -negamax(board, depth - 1, ply + 1, -beta, -alpha, localBestMove);
        board.undoMove(move);

        if (score >= beta) return beta; // beta cutoff - opponent won't allow this
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

Search::SearchResult Search::search(Board& board, int maxDepth) {
    SearchResult result{Move::none(), 0, 0};
    
    for (int depth = 1; depth <= maxDepth; ++depth) {
        Move bestMove = Move::none();
        const int score = negamax(board, depth, 0, -INF, INF, bestMove);
        if(!bestMove.isNone()) {
            result.bestMove = bestMove;
            result.score = score;
            result.depth = depth;
        }
    }
    return result;
}