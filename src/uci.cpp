#include "uci.h"
#include "board.h"
#include "movegen.h"
#include "search.h"
#include "attacks.h"
#include "zobrist.h"
#include "endgame_mode.h"
#include "syzygy.h"
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

static constexpr auto START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
static constexpr const char* SYZYGY_PATH =
    "C:/Users/light/OneDrive/Documents/Schoolwork/tablebases/3-4-5-wdl;"
    "C:/Users/light/OneDrive/Documents/Schoolwork/tablebases/6-wdl;"
    "C:/Users/light/OneDrive/Documents/Schoolwork/tablebases/3-4-5-dtz;"
    "C:/Users/light/OneDrive/Documents/Schoolwork/tablebases/6-dtz";
    
static Move parseMove(Board& board, const std::string& s) {
    Square from = Square((s[1] - '1') * 8 + (s[0] - 'a'));
    Square to = Square((s[3] - '1') * 8 + (s[2] - 'a'));
    PieceType promo = QUEEN;
    if (s.size() == 5) {
        switch (s[4]) {
            case 'r': promo = ROOK; break;
            case 'b': promo = BISHOP; break;
            case 'n': promo = KNIGHT; break;
            default: promo = QUEEN; break;
        }
    }
    Move moves[256];
    Move* end = MoveGen::generateLegalMoves(board, moves);
    for (Move* move = moves; move != end; ++move)
        if (move->from() == from && move->to() == to)
            if (!move->isPromotion() || move->promotionType() == promo)
                return *move;
    return Move::none();
}

static std::string moveToUCI(Move move) {
    std::string s;
    s+= char('a' + move.from() % 8);
    s+= char('1' + move.from() / 8);
    s+= char('a' + move.to() % 8);
    s+= char('1' + move.to() / 8);
    if (move.isPromotion()) {
        const char possiblePromotions[] = {'q', 'r', 'b', 'n'};
        s += possiblePromotions[move.promotionType() - KNIGHT];
    }
    return s;
}

static void parsePosition(Board& board, std::istringstream& iss) {
    std::string token;
    iss >> token;
    if (token == "startpos") {
        board.loadFEN(START_FEN);
        iss >> token;// consume "moves" if present
    } else if (token == "fen") {
        std::string fen, part;
        while (iss >> part && part != "moves")
            fen += part + " ";
        board.loadFEN(fen);
        token = "moves"; //already consumed if present
    }
    if (token ==  "moves") {
        while (iss >> token) {
            board.makeMove(parseMove(board, token));
        }
    }
}

static void parseGo(Board& board, std::istringstream& iss) {
    Search::Limits limits;
    std::string token;
    while (iss >> token) {
        if (token == "wtime")     iss >> limits.wtime;
        else if (token == "btime") iss >> limits.btime;
        else if (token == "winc")  iss >> limits.winc;
        else if (token == "binc")  iss >> limits.binc;
        else if (token == "movestogo") iss >> limits.movestogo;
        else if (token == "movetime")  iss >> limits.movetime;
        else if (token == "depth")     iss >> limits.depth;
        else if (token == "infinite")  limits.infinite = true;
    }

    // Run the search in a separate thread to avoid blocking the UCI loop
    std::thread([board, limits]() mutable {
        Search::SearchResult result = Search::search(board, limits);
        std::cout << "bestmove " << moveToUCI(result.bestMove) << std::endl;
        std::cout.flush();
    }).detach();
}

void UCI::loop() {
    Zobrist::init();
    Attacks::init();
    Board gameBoard;
    gameBoard.loadFEN(START_FEN);

    std::string line, token;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        iss >> token;
        if (token == "uci") {
            std::cout   << "id name ChessEngine\n"
                        << "id author BrainDBD\n"
                        << "option name EndgameMode type combo default Classic var Classic var MLPBlend var MLPReplace var Syzygy\n"
                        << "option name SyzygyPath type string default <empty>\n"
                        << "uciok\n";
        } else if (token == "isready") {
            std::cout << "readyok\n";
        } else if (token == "ucinewgame") {
            gameBoard.loadFEN(START_FEN);
            Search::clearTT();
        } else if (token == "position") {
            parsePosition(gameBoard, iss);
        } else if (token == "setoption") {
            std::string name, kw;
            iss >> kw;                    // "name"
            iss >> name;                  // option name
            iss >> kw;                    // "value"
            std::string value;
            std::getline(iss, value);     // remainder of line (may be empty)
            if (!value.empty() && value.front() == ' ') value.erase(0, 1);

            if (name == "EndgameMode") {
                if      (value == "MLPReplace") Endgame::g_mode = Endgame::Mode::MLPReplace;
                else if (value == "MLPBlend")   Endgame::g_mode = Endgame::Mode::MLPBlend;
                else if (value == "Syzygy") {
                    Endgame::g_mode = Endgame::Mode::Syzygy;
                    // If tables were not loaded via SyzygyPath yet, fall back to the compiled-in default path (convenient for manual testing).
                    if (!Syzygy::isReady()) {
                        bool ok = Syzygy::init(SYZYGY_PATH);
                        std::cout << "info string Syzygy init " << (ok ? "OK" : "FAILED")
                                << " max=" << Syzygy::maxPieces() << std::endl;
                    }
                }
                else                            Endgame::g_mode = Endgame::Mode::Classic;
                Search::clearTT();        // scores from the old mode are now invalid
            } else if (name == "SyzygyPath") {
                // path supplied by the GUI / fastchess. Loads WDL+DTZ dirs.
                if (!value.empty() && value != "<empty>") {
                    bool ok = Syzygy::init(value.c_str());
                    std::cout << "info string Syzygy init " << (ok ? "OK" : "FAILED")
                            << " max=" << Syzygy::maxPieces() << std::endl;
                }
            }
        } else if (token == "go") {
            Search::stop = false; // arm the search
            parseGo(gameBoard, iss);
        } else if (token == "stop") {
            Search::stop = true;
        } else if (token == "quit") {
            Search::stop = true;
            break;
        }
        std::cout.flush();
    }
}