#include "board.h"
#include <iostream>
#include <cctype>
#include <sstream>
#include <stdexcept>

Board::Board() : whiteToMove_(true), castlingRights_(0) {
    loadFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::loadFEN(const std::string& fen) {
    // Clear the board
    for (int color = 0; color < COLOR_NB; ++color)
        for (int pieceType = 0; pieceType < PIECE_TYPE_NB; ++pieceType)
            bb_[color][pieceType] = 0ULL;
    board_.fill(NO_PIECE);
    whiteToMove_ = true;
    castlingRights_ = NO_CASTLING;
    
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
        board_[square] = makePiece(color, pieceType);
        file++;
    }

    whiteToMove_ = (activeColor == "w");
    castlingRights_ = NO_CASTLING;
    if (castling != "-") {
        for (char c : castling) {
            switch (c) {
                case 'K': castlingRights_ |= WHITE_OO; break;
                case 'Q': castlingRights_ |= WHITE_OOO; break;
                case 'k': castlingRights_ |= BLACK_OO; break;
                case 'q': castlingRights_ |= BLACK_OOO; break;
                default: throw std::invalid_argument("Invalid FEN: unknown castling symbol");
            }
        }
    }

    enPassantSquare_ = SQ_NONE;
    if (enPassant != "-") {
        int epFile = enPassant[0] - 'a';
        int epRank = enPassant[1] - '1';
        enPassantSquare_ = static_cast<Square>(epRank * 8 + epFile);
    }

    halfmoveClock_  = halfmoveClock;
    fullmoveNumber_ = fullmoveNumber;
    hash_ = computeHash();
}

std::string Board::toFEN() const {
    std::string fen;
    for (int rank = 7; rank >= 0; --rank) {
        int emptySquares = 0;
        for (int file = 0; file < 8; ++file) {
            int square = rank * 8 + file;
            if(board_[square] == NO_PIECE)
                ++emptySquares;
            else {
                if (emptySquares) {
                    fen += std::to_string(emptySquares);
                    emptySquares = 0;
                }
                char pieceChar;
                switch (typeOf(board_[square])) {
                    case PAWN:   pieceChar = 'p'; break;
                    case KNIGHT: pieceChar = 'n'; break;
                    case BISHOP: pieceChar = 'b'; break;
                    case ROOK:   pieceChar = 'r'; break;
                    case QUEEN:  pieceChar = 'q'; break;
                    case KING:   pieceChar = 'k'; break;
                    default:
                        throw std::logic_error("Invalid piece type on board!");
                }
                if (colorOf(board_[square]) == WHITE)
                    pieceChar = static_cast<char>(std::toupper(static_cast<unsigned char>(pieceChar)));
                fen += pieceChar;
            }
        }
        if (emptySquares)
            fen += std::to_string(emptySquares);
        if (rank)
            fen += '/';
    }
    fen += whiteToMove_ ? " w " : " b ";

    if(castlingRights_ == NO_CASTLING) {
        fen += "-";
    } else {
        if (castlingRights_ & WHITE_OO) fen += "K";
        if (castlingRights_ & WHITE_OOO) fen += "Q";
        if (castlingRights_ & BLACK_OO) fen += "k";
        if (castlingRights_ & BLACK_OOO) fen += "q";
    }
    fen += ' ';
    if (enPassantSquare_ == SQ_NONE)
        fen += '-';
    else {
        fen += static_cast<char>('a' + (enPassantSquare_ % 8));
        fen += static_cast<char>('1' + (enPassantSquare_ / 8));
    }
    fen += ' ';
    fen += std::to_string(halfmoveClock_);
    fen += ' ';
    fen += std::to_string(fullmoveNumber_);
    return fen;
}

bool Board::canCastle(CastlingRights rights) const {
    return castlingRights_ & rights;
}

uint64_t Board::computeHash() const {
    uint64_t h = 0;

    for (int sq = 0; sq < SQUARE_NB; ++sq) {
        Piece p = board_[sq];
        if (p != NO_PIECE)
            h ^= Zobrist::piece[colorOf(p)][typeOf(p)][sq];
    }

    if (whiteToMove_)
        h ^= Zobrist::sideToMove;

    h ^= Zobrist::castling[castlingRights_];

    if (enPassantSquare_ != SQ_NONE)
        h ^= Zobrist::enPassant[enPassantSquare_ % 8];

    return h;
}