#include "board.h"
#include "zobrist.h"
#include "attacks.h"
#include "movegen.h"
#include "eval.h"
#include "search.h"
#include "uci.h"
#include <iostream>

// static void evalCheck(Board& board, const std::string& fen, const std::string& label) {
//     board.loadFEN(fen);
//     std::cout << label << ": " << Eval::evaluate(board) << " cp\n";
// }

// uint64_t perft(Board& board, int depth) {
//     if (depth == 0) return 1;

//     std::vector<Move> moves;
//     MoveGen::generateLegalMoves(board, moves);

//     uint64_t nodes = 0;
//     for (Move m : moves) {
//         board.makeMove(m);
//         nodes += perft(board, depth - 1);
//         board.undoMove(m);
//     }
//     return nodes;
// }

// void perftDivide(Board& board, int depth) {
//     std::vector<Move> moves;
//     MoveGen::generateLegalMoves(board, moves);

//     uint64_t total = 0;
//     for (Move m : moves) {
//         board.makeMove(m);
//         uint64_t count = perft(board, depth - 1);
//         board.undoMove(m);

//         // print move in algebraic notation
//         char fromFile = 'a' + (m.from() % 8);
//         char fromRank = '1' + (m.from() / 8);
//         char toFile   = 'a' + (m.to()   % 8);
//         char toRank   = '1' + (m.to()   / 8);
//         std::cout << fromFile << fromRank << toFile << toRank;
//         if (m.isPromotion()) {
//             const char promoChars[] = {'n','b','r','q'};
//             std::cout << promoChars[m.promotionType() - KNIGHT];
//         }
//         std::cout << ": " << count << '\n';
//         total += count;
//     }
//     std::cout << "\nTotal: " << total << '\n';
// }

int main() {
    UCI::loop();
    return 0;
}

// how to run: mingw32-make, ./chess.exe