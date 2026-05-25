#include "movegen.h"
#include "attacks.h"

using namespace BitboardUtils;

static constexpr Bitboard RANK_1_BB = 0x00000000000000FFULL;
static constexpr Bitboard RANK_2_BB = 0x000000000000FF00ULL;
static constexpr Bitboard RANK_7_BB = 0x00FF000000000000ULL;
static constexpr Bitboard RANK_8_BB = 0xFF00000000000000ULL;
static constexpr Bitboard NOT_A_FILE = 0xFEFEFEFEFEFEFEFEULL;
static constexpr Bitboard NOT_H_FILE = 0x7F7F7F7F7F7F7F7FULL;

static void generatePawnMoves(const Board& board, Color us, std::vector<Move>& moves) {
    const Bitboard pawns = board.pieces(us, PAWN);
    const Bitboard empty = ~board.occupancy();
    const Bitboard enemies = board.pieces(~us);
    const Square enPassantSquare = board.enpassantSquare();
    const Bitboard promoteRank = (us == WHITE) ? RANK_8_BB : RANK_1_BB;

    Bitboard singlePush = (us == WHITE) ? (pawns << 8) & empty : (pawns >> 8) & empty;
    Bitboard startRank = (us == WHITE) ? RANK_2_BB : RANK_7_BB;
    Bitboard doublePush = (us == WHITE ? (pawns & startRank) << 16 : (pawns & startRank) >> 16) & empty & (us == WHITE ? empty << 8 : empty >> 8);

    Bitboard capturesLeft = (us == WHITE ? (pawns & NOT_A_FILE) << 7 : (pawns & NOT_H_FILE) >> 7) & enemies;
    Bitboard capturesRight = ((us == WHITE ? (pawns & NOT_H_FILE) << 9 : (pawns & NOT_A_FILE) >> 9) & enemies);

    // Helper to extract moves from target bitboard, deriving from square via offset
    auto extractMoves = [&](Bitboard targets, int fromOffset, MoveFlag flag = NORMAL) {
        while (targets) {
            Square to = static_cast<Square>(popLsb(targets));
            Square from = static_cast<Square>(to - fromOffset);
            if ((1ULL << to) & promoteRank) {
                moves.push_back(Move(from, to, PROMOTION, KNIGHT));
                moves.push_back(Move(from, to, PROMOTION, BISHOP));
                moves.push_back(Move(from, to, PROMOTION, ROOK));
                moves.push_back(Move(from, to, PROMOTION, QUEEN));
            }
            else {
                moves.push_back(Move(from, to, flag));
            }
        }
    };

    const int direction = (us == WHITE) ? 1 : -1;
    extractMoves(singlePush, direction * 8);
    extractMoves(doublePush, direction * 16);
    extractMoves(capturesLeft, direction * 7);
    extractMoves(capturesRight, direction * 9);

    // En passant captures
    if (enPassantSquare != SQ_NONE) {
        Bitboard enPassantCaptures = Attacks::pawnAttacks[~us][enPassantSquare] & pawns;
        while (enPassantCaptures) {
            Square from = static_cast<Square>(popLsb(enPassantCaptures));
            moves.push_back(Move(from, enPassantSquare, EN_PASSANT));
        }
    }
}

static void generatePieceMoves(const Board& board, Color us, PieceType pt, std::vector<Move>& moves) {
    const Bitboard friendlyPieces = board.pieces(us);
    const Bitboard occupied = board.occupancy();
    Bitboard pieces = board.pieces(us, pt);
    while (pieces) {
        Square from = static_cast<Square>(popLsb(pieces));
        Bitboard attacks;
        switch (pt) {
            case KNIGHT: attacks = Attacks::knightAttacks[from]; break;
            case BISHOP: attacks = Attacks::bishopAttacks(from, occupied); break;
            case ROOK:   attacks = Attacks::rookAttacks(from, occupied); break;
            case QUEEN:  attacks = Attacks::queenAttacks(from, occupied); break;
            case KING:   attacks = Attacks::kingAttacks[from]; break;
            default: attacks = 0; break;
        }
        attacks &= ~friendlyPieces; // can't capture own pieces
        while (attacks) {
            Square to = static_cast<Square>(popLsb(attacks));
            moves.push_back(Move(from, to));
        }
    }
}

static bool isSquareAttacked(const Board& board, Square sq, Color by) {
    const Bitboard occ = board.occupancy();
    if (Attacks::pawnAttacks[~by][sq] & board.pieces(by, PAWN))
        return true;
    if (Attacks::knightAttacks[sq] & board.pieces(by, KNIGHT))
        return true;
    if (Attacks::kingAttacks[sq] & board.pieces(by, KING))
        return true;
    if (Attacks::bishopAttacks(sq, occ) & (board.pieces(by, BISHOP) | board.pieces(by, QUEEN)))
        return true;
    if (Attacks::rookAttacks(sq, occ) & (board.pieces(by, ROOK) | board.pieces(by, QUEEN)))
        return true;

    return false;
}

static void generateCastling(const Board& board, Color us, std::vector<Move>& moves) {
    const int rights = board.castlingRights();
    const Bitboard occ  = board.occupancy();
    const Color them   = ~us;

    if (us == WHITE) {
        if ((rights & WHITE_OO)  && !(occ & 0x60ULL)
            && !isSquareAttacked(board, SQ_E1, them)
            && !isSquareAttacked(board, SQ_F1, them))
            moves.push_back(Move(SQ_E1, SQ_G1, CASTLING));

        if ((rights & WHITE_OOO) && !(occ & 0x0EULL)
            && !isSquareAttacked(board, SQ_E1, them)
            && !isSquareAttacked(board, SQ_D1, them))
            moves.push_back(Move(SQ_E1, SQ_C1, CASTLING));
    } else {
        if ((rights & BLACK_OO)  && !(occ & 0x6000000000000000ULL)
            && !isSquareAttacked(board, SQ_E8, them)
            && !isSquareAttacked(board, SQ_F8, them))
            moves.push_back(Move(SQ_E8, SQ_G8, CASTLING));

        if ((rights & BLACK_OOO) && !(occ & 0x0E00000000000000ULL)
            && !isSquareAttacked(board, SQ_E8, them)
            && !isSquareAttacked(board, SQ_D8, them))
            moves.push_back(Move(SQ_E8, SQ_C8, CASTLING));
    }
}

namespace MoveGen {
    void generateMoves(const Board& board, std::vector<Move>& moves) {
        const Color us = board.sideToMove();
        generatePawnMoves (board, us, moves);
        generatePieceMoves(board, us, KNIGHT, moves);
        generatePieceMoves(board, us, BISHOP, moves);
        generatePieceMoves(board, us, ROOK, moves);
        generatePieceMoves(board, us, QUEEN, moves);
        generatePieceMoves(board, us, KING, moves);
        generateCastling  (board, us, moves);
    }

    bool isSquareAttacked(const Board& board, Square sq, Color by) {
        return ::isSquareAttacked(board, sq, by);
    }

    bool isInCheck(const Board& board, Color c) {
        Square kingSq = static_cast<Square>(lsb(board.pieces(c, KING)));
        return ::isSquareAttacked(board, kingSq, ~c);
    }

    void generateLegalMoves(Board& board, std::vector<Move>& moves) {
        std::vector<Move> pseudoLegalMoves;
        generateMoves(board, pseudoLegalMoves);

        for (Move move : pseudoLegalMoves) {
            board.makeMove(move);
            if (!isInCheck(board, ~board.sideToMove()))
                moves.push_back(move);
            board.undoMove(move);
        }
    }
}