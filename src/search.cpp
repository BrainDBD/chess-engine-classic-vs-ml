#include "search.h"
#include "eval.h"
#include "movegen.h"
#include <algorithm>
#include <vector>

static constexpr int INF = 1'000'000;
static constexpr int MATE_SCORE = 900'000;

static void orderMoves(std::vector<Move>& moves, const Board& board, Move ttMove) {
    auto score = [&](const Move move) -> int {
        if (move.data == ttMove.data) return 20'000; // search PV first
        if (move.isPromotion()) return 10'000 + static_cast<int>(move.promotionType());

        const Piece captured = board.pieceAt(move.to());
        if (captured != NO_PIECE) {
            static constexpr int TAKEN_VALUE[PIECE_TYPE_NB] = {100, 200, 300, 500, 900, 0};
            static constexpr int TAKER_VALUE[PIECE_TYPE_NB] = {1, 2 , 3, 4, 5, 6};
            return 5'000 + TAKEN_VALUE[typeOf(captured)] - TAKER_VALUE[typeOf(board.pieceAt(move.from()))];
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

static int negamax(Board& board, int depth, int alpha, int beta, Move& bestMove) {
    if (depth == 0) return quiescence(board, alpha, beta);

    std::vector<Move> moves;
    MoveGen::generateLegalMoves(board, moves);
    
    if (moves.empty()) {
        // No legal moves: checkmate or stalemate
        return MoveGen::isInCheck(board, board.sideToMove())
            ? -(MATE_SCORE - depth)  // prefer shorter mates
            :  0;                    // stalemate
    }

    orderMoves(moves, board, Move::none());

    Move localBestMove = Move::none();
    for (const Move move : moves) {
        board.makeMove(move);
        const int score = -negamax(board, depth - 1, -beta, -alpha, localBestMove);
        board.undoMove(move);

        if (score >= beta) return beta; // beta cutoff - opponent won't allow this
        if (score > alpha) {
            alpha = score;
            localBestMove = move;
        }
    }
    bestMove = localBestMove;
    return alpha;
}

Search::SearchResult Search::search(Board& board, int maxDepth) {
    SearchResult result{Move::none(), 0, 0};
    
    for (int depth = 1; depth <= maxDepth; ++depth) {
        Move bestMove = Move::none();
        const int score = negamax(board, depth, -INF, INF, bestMove);
        if(!bestMove.isNone()) {
            result.bestMove = bestMove;
            result.score = score;
            result.depth = depth;
        }
    }
    return result;
}