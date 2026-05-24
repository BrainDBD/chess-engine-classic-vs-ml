#include "board.h"
#include "zobrist.h"
#include "attacks.h"
#include "movegen.h"
#include <iostream>

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
    board.loadFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

    for (int depth = 1; depth <= 3; ++depth)
        std::cout << "Depth " << depth << ": " << perft(board, depth) << '\n';

    std::cout << "\n--- Divide depth 2 ---\n";
    perftDivide(board, 2);
    return 0;
}