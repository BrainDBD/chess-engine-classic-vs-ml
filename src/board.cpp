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
    history_.clear();
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
    uint64_t hash = 0;

    for (int sq = 0; sq < SQUARE_NB; ++sq) {
        Piece piece = board_[sq];
        if (piece != NO_PIECE)
            hash ^= Zobrist::piece[colorOf(piece)][typeOf(piece)][sq];
    }

    if (whiteToMove_)
        hash ^= Zobrist::sideToMove;

    hash ^= Zobrist::castling[castlingRights_];

    if (enPassantSquare_ != SQ_NONE)
        hash ^= Zobrist::enPassant[enPassantSquare_ % 8];

    return hash;
}

void Board::putPiece(Square sq, Piece p) {
    board_[sq] = p;
    BitboardUtils::setBit(bb_[colorOf(p)][typeOf(p)], sq);
}

void Board::removePiece(Square sq) {
    Piece piece = board_[sq];
    board_[sq] = NO_PIECE;
    BitboardUtils::clearBit(bb_[colorOf(piece)][typeOf(piece)], sq);
}

// Precompute a mask for which castling rights are lost when a piece moves from or to each square
static const std::array<int, SQUARE_NB> initCastlingMask = []() {
    std::array<int, SQUARE_NB> m;
    m.fill(ANY_CASTLING);
    m[SQ_A1] &= ~WHITE_OOO;
    m[SQ_E1] &= ~WHITE_CASTLING;
    m[SQ_H1] &= ~WHITE_OO;
    m[SQ_A8] &= ~BLACK_OOO;
    m[SQ_E8] &= ~BLACK_CASTLING;
    m[SQ_H8] &= ~BLACK_OO;
    return m;
}();

void Board::makeMove(Move move) {
    // Record undo information
    const Square from = move.from();
    const Square to = move.to();
    const Color us = sideToMove();
    const Piece moving = board_[from];
    const Piece captured = move.isEnPassant() ? makePiece(~us, PAWN) : board_[to];

    history_.push_back({captured, enPassantSquare_, castlingRights_, halfmoveClock_, hash_});

    // XOR out the state that will change
    hash_ ^= Zobrist::castling[castlingRights_];
    if (enPassantSquare_ != SQ_NONE)
        hash_ ^= Zobrist::enPassant[enPassantSquare_ % 8];

    // Remove captured piece
    if (captured != NO_PIECE) {
        Square captureSquare = move.isEnPassant() ? static_cast<Square>(to + (us == WHITE ? -8 : 8)) : to;
        hash_ ^= Zobrist::piece[colorOf(captured)][typeOf(captured)][captureSquare];
        removePiece(captureSquare);
    }

    // Move piece
    hash_ ^= Zobrist::piece[colorOf(moving)][typeOf(moving)][from];
    removePiece(from);

    Piece pieceToPlace = move.isPromotion() ? makePiece(us, move.promotionType()) : moving;
    putPiece(to, pieceToPlace);
    hash_ ^= Zobrist::piece[colorOf(pieceToPlace)][typeOf(pieceToPlace)][to];

    //If castling, move rook
    if (move.isCastling()) {
        const Square rookFrom = (to == SQ_G1) ? SQ_H1 : (to == SQ_C1) ? SQ_A1 : (to == SQ_G8) ? SQ_H8 : SQ_A8;
        const Square rookTo   = (to == SQ_G1) ? SQ_F1 : (to == SQ_C1) ? SQ_D1: (to == SQ_G8) ? SQ_F8 : SQ_D8;
        Piece rook = board_[rookFrom];
        hash_ ^= Zobrist::piece[colorOf(rook)][typeOf(rook)][rookFrom];
        removePiece(rookFrom);
        putPiece(rookTo, rook);
        hash_ ^= Zobrist::piece[colorOf(rook)][typeOf(rook)][rookTo];
    }

    //Update castling rights
    castlingRights_ &= initCastlingMask[from] & initCastlingMask[to];

    // Update en passant square
    enPassantSquare_ = SQ_NONE;
    if (typeOf(moving) == PAWN && (to - from == 16 || from - to == 16))
        enPassantSquare_ = static_cast<Square>((from + to) / 2);
    
    //Update clocks
    halfmoveClock_ = (typeOf(moving) == PAWN || captured != NO_PIECE) ? 0 : halfmoveClock_ + 1;
    if (!whiteToMove_) ++fullmoveNumber_;

    // Flip side
    whiteToMove_ = !whiteToMove_;
    hash_ ^= Zobrist::sideToMove;

    // XOR in new state
    hash_ ^= Zobrist::castling[castlingRights_];
    if (enPassantSquare_ != SQ_NONE)
        hash_ ^= Zobrist::enPassant[enPassantSquare_ % 8];
}

void Board::undoMove(Move move) {
    const UndoInfo& info = history_.back();
    const Square from = move.from();
    const Square to = move.to();
    whiteToMove_ = !whiteToMove_;
    const Color us = sideToMove();

    // Undo promotion: piece at `to` is the promoted piece, put pawn back
    Piece moveBack = move.isPromotion()? makePiece(us, PAWN) : board_[to];
    removePiece(to);
    putPiece(from, moveBack);

    // Restore captured piece
    if (info.captured != NO_PIECE) {
        Square captureSquare = move.isEnPassant() ? static_cast<Square>(to + (us == WHITE ? -8 : 8)) : to;
        putPiece(captureSquare, info.captured);
    }

    // Undo castling
    if (move.isCastling()) {
        const Square rookFrom = (to == SQ_G1) ? SQ_H1 : (to == SQ_C1) ? SQ_A1 : (to == SQ_G8) ? SQ_H8 : SQ_A8;
        const Square rookTo   = (to == SQ_G1) ? SQ_F1 : (to == SQ_C1) ? SQ_D1 : (to == SQ_G8) ? SQ_F8 : SQ_D8;
        removePiece(rookTo);
        putPiece(rookFrom, makePiece(us, ROOK)); 
    }

    //  Restore state
    castlingRights_ = info.castlingRights;
    enPassantSquare_ = info.enPassantSquare;
    halfmoveClock_ = info.halfmoveClock;
    hash_ = info.hash;

    history_.pop_back();
}

void Board::makeNullMove() {
    history_.push_back({NO_PIECE, enPassantSquare_, castlingRights_, halfmoveClock_, hash_});

    if (enPassantSquare_ != SQ_NONE)
        hash_ ^= Zobrist::enPassant[enPassantSquare_ % 8];
    hash_ ^= Zobrist::sideToMove;

    enPassantSquare_ = SQ_NONE;
    whiteToMove_ = !whiteToMove_;
    ++halfmoveClock_;
}

void Board::undoNullMove() {
    const UndoInfo& info = history_.back();
    enPassantSquare_ = info.enPassantSquare;
    castlingRights_ = info.castlingRights;
    halfmoveClock_ = info.halfmoveClock;
    hash_ = info.hash;
    whiteToMove_ = !whiteToMove_;
    history_.pop_back();
}

Bitboard Board::pieces(Color c) const {
    Bitboard occ = 0;
    for (int pt = 0; pt < PIECE_TYPE_NB; ++pt)
        occ |= bb_[c][pt];
    return occ;
}

bool Board::isRepetition() const {
    const int size = static_cast<int>(history_.size());
    const int limit = std::max(0, size - halfmoveClock_);
    for (int i = size - 2; i >= limit; i -= 2)
        if (history_[i].hash == hash_)
            return true;
    return false;
}