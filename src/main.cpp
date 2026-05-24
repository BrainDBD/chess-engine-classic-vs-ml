#include "board.h"
#include "zobrist.h"
#include "attacks.h"
#include "movegen.h"
#include "eval.h"
#include <iostream>

static void evalCheck(Board& board, const std::string& fen, const std::string& label) {
    board.loadFEN(fen);
    std::cout << label << ": " << Eval::evaluate(board) << " cp\n";
}

uint64_t perft(Board& board, int depth) {
    if (depth == 0) return 1;

    std::vector<Move> moves;
    MoveGen::generateLegalMoves(board, moves);

    uint64_t nodes = 0;
    for (Move m : moves) {
        board.makeMove(m);
        nodes += perft(board, depth - 1);
        board.undoMove(m);
    }
    return nodes;
}

void perftDivide(Board& board, int depth) {
    std::vector<Move> moves;
    MoveGen::generateLegalMoves(board, moves);

    uint64_t total = 0;
    for (Move m : moves) {
        board.makeMove(m);
        uint64_t count = perft(board, depth - 1);
        board.undoMove(m);

        // print move in algebraic notation
        char fromFile = 'a' + (m.from() % 8);
        char fromRank = '1' + (m.from() / 8);
        char toFile   = 'a' + (m.to()   % 8);
        char toRank   = '1' + (m.to()   / 8);
        std::cout << fromFile << fromRank << toFile << toRank;
        if (m.isPromotion()) {
            const char promoChars[] = {'n','b','r','q'};
            std::cout << promoChars[m.promotionType() - KNIGHT];
        }
        std::cout << ": " << count << '\n';
        total += count;
    }
    std::cout << "\nTotal: " << total << '\n';
}

int main() {
    Zobrist::init();
    Attacks::init();
    Board board;

    std::cout << "--- Eval sanity checks ---\n";
    evalCheck(board, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",   "Start      (expect ~0)");
    evalCheck(board, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4 (expect small +, white to move gone so score is for black)");
    evalCheck(board, "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 1","After 1.e4 e5 (expect ~0)");
    evalCheck(board, "8/8/8/8/8/8/3P4/8 w - - 0 1", "Single pawn d2          ");
    evalCheck(board, "8/8/8/8/8/3P4/3P4/8 w - - 0 1", "Doubled pawns d2+d3 (doubled penalty should make this < 2x single)");

    std::cout << '\n';

    // board.loadFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

    // for (int depth = 1; depth <= 3; ++depth)
    //     std::cout << "Depth " << depth << ": " << perft(board, depth) << '\n';

    // std::cout << "\n--- Divide depth 2 ---\n";
    // perftDivide(board, 3);
    return 0;
}