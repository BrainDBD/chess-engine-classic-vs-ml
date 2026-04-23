#include "board.h"
#include <iostream>
#include <cctype>
#include <sstream>
#include <stdexcept>

Board::Board() : whiteToMove(true), castlingRights(0) {
    loadFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::loadFEN(const std::string& fen) {
    // Clear the board
    for (int color = 0; color < COLOR_NB; ++color)
        for (int pieceType = 0; pieceType < PIECE_TYPE_NB; ++pieceType)
            bb_[color][pieceType] = 0ULL;
    color_on_.fill(COLOR_NB);
    piece_on_.fill(PIECE_TYPE_NB);
    whiteToMove = true;
    castlingRights = NO_CASTLING;
    
    int rank = 7, file = 0;

    // Parse FEN string
    std::istringstream iss(fen);
    std::string fenBoard, activeColor, castling, enPassant;
    int halfmoveClock = 0, fullmoveNumber = 1;
    if (!(iss >> fenBoard >> activeColor >> castling >> enPassant >> halfmoveClock >> fullmoveNumber))
        throw std::invalid_argument("Invalid FEN: not enough fields");
    
    for (char symbol: fenBoard) {
        if (symbol == '/') {
            rank--;
            file = 0;
            continue;
        }
        if(std::isdigit(static_cast<unsigned char>(symbol))) {
            file += symbol - '0';
            continue;
        }
        const Color color = std::isupper(static_cast<unsigned char>(symbol)) ? WHITE : BLACK;
        const char pieceChar = static_cast<char>(std::tolower(static_cast<unsigned char>(symbol)));
        PieceType pieceType;
        switch (pieceChar) {
            case 'p': pieceType = PAWN; break;
            case 'n': pieceType = KNIGHT; break;
            case 'b': pieceType = BISHOP; break;
            case 'r': pieceType = ROOK; break;
            case 'q': pieceType = QUEEN; break;
            case 'k': pieceType = KING; break;
            default: throw std::invalid_argument("Invalid FEN: unknown piece symbol");
        }

        const int square = rank * 8 + file;
        BitboardUtils::setBit(bb_[color][pieceType], square);
        color_on_[square] = color;
        piece_on_[square] = pieceType;
        file++;
    }

    whiteToMove = (activeColor == "w");
    castlingRights = NO_CASTLING;
    if (castling != "-") {
        for (char c : castling) {
            switch (c) {
                case 'K': castlingRights |= WHITE_OO; break;
                case 'Q': castlingRights |= WHITE_OOO; break;
                case 'k': castlingRights |= BLACK_OO; break;
                case 'q': castlingRights |= BLACK_OOO; break;
                default: throw std::invalid_argument("Invalid FEN: unknown castling symbol");
            }
        }
    }
}

std::string Board::toFEN() const {
    std::string fen;
    for (int rank = 7; rank >= 0; --rank) {
        int emptySquares = 0;
        for (int file = 0; file < 8; ++file) {
            int square = rank * 8 + file;
            if(piece_on_[square] == PIECE_TYPE_NB)
                ++emptySquares;
            else {
                if (emptySquares) {
                    fen += std::to_string(emptySquares);
                    emptySquares = 0;
                }
                char pieceChar;
                switch (piece_on_[square]) {
                    case PAWN:   pieceChar = 'p'; break;
                    case KNIGHT: pieceChar = 'n'; break;
                    case BISHOP: pieceChar = 'b'; break;
                    case ROOK:   pieceChar = 'r'; break;
                    case QUEEN:  pieceChar = 'q'; break;
                    case KING:   pieceChar = 'k'; break;
                    default:
                        throw std::logic_error("Invalid piece type on board!");
                }
                if (color_on_[square] == WHITE)
                    pieceChar = static_cast<char>(std::toupper(static_cast<unsigned char>(pieceChar)));
                fen += pieceChar;
            }
        }
        if (emptySquares)
            fen += std::to_string(emptySquares);
        if (rank)
            fen += '/';
    }
    fen += whiteToMove ? " w " : " b ";

    if(castlingRights == NO_CASTLING) {
        fen += "-";
    } else {
        if (castlingRights & WHITE_OO) fen += "K";
        if (castlingRights & WHITE_OOO) fen += "Q";
        if (castlingRights & BLACK_OO) fen += "k";
        if (castlingRights & BLACK_OOO) fen += "q";
    }
    fen += " - 0 1";
    return fen;
}

bool Board::canCastle(int rights) const {
    return castlingRights & rights;
}