#pragma once
#include "bitboard.h"
#include "types.h"
#include "board.h"
#include <cstdlib>
// squares are rank*8+file, identical to python-chess.
inline int sqRank(int sq) { return sq >> 3; }
inline int sqFile(int sq) { return sq & 7; }
inline int chebyshev(int a, int b) {
    int dr = sqRank(a) - sqRank(b); if (dr < 0) dr = -dr;
    int df = sqFile(a) - sqFile(b); if (df < 0) df = -df;
    return dr > df ? dr : df;          // king-move (Chebyshev) distance
}
inline int kingEdgeDist(int sq) {
    int r = sqRank(sq), f = sqFile(sq);
    int a = r < 7-r ? r : 7-r;
    int b = f < 7-f ? f : 7-f;
    return a < b ? a : b;
}

template <class B>
struct EndgameFeatures {
    // passed pawns of `color` into a 64-bit set (squares)
    static Bitboard passedPawns(const B& board, Color color) {
        Bitboard pawns = board.pieces(color, PAWN);
        Bitboard enemy = board.pieces(~color, PAWN);
        Bitboard out = 0;
        Bitboard pp = pawns;
        while (pp) {
            int sq = BitboardUtils::popLsb(pp);
            int f = sqFile(sq), r = sqRank(sq);
            bool blocked = false;
            Bitboard ep = enemy;
            while (ep) {
                int es = BitboardUtils::popLsb(ep);
                if (std::abs(sqFile(es) - f) > 1) continue;
                int er = sqRank(es);
                if ((color == WHITE && er > r) || (color == BLACK && er < r)) { blocked = true; break; }
            }
            if (!blocked) BitboardUtils::setBit(out, sq);
        }
        return out;
    }

    static int bishopColor(const B& board, Color color) {
        Bitboard b = board.pieces(color, BISHOP);
        if (BitboardUtils::countBits(b) != 1) return -1;
        int sq = BitboardUtils::lsb(b);
        return (sqFile(sq) + sqRank(sq)) & 1;   // 1=light, 0=dark
    }

    static void pawnStructure(const B& board, Color color, int& doubled, int& isolated) {
        int fileCount[8] = {0,0,0,0,0,0,0,0};
        Bitboard p = board.pieces(color, PAWN);
        while (p) fileCount[sqFile(BitboardUtils::popLsb(p))]++;
        doubled = isolated = 0;
        for (int f = 0; f < 8; ++f) {
            if (fileCount[f] > 1) doubled += fileCount[f] - 1;
            if (fileCount[f] > 0) {
                bool adj = (f>0 && fileCount[f-1]>0) || (f<7 && fileCount[f+1]>0);
                if (!adj) isolated += fileCount[f];
            }
        }
    }

    static int connectedPassers(Bitboard passers) {
        bool hasFile[8] = {false,false,false,false,false,false,false,false};
        Bitboard p = passers;
        while (p) hasFile[sqFile(BitboardUtils::popLsb(p))] = true;
        for (int f = 0; f < 7; ++f) if (hasFile[f] && hasFile[f+1]) return 1;
        return 0;
    }

    static int kingSq(const B& board, Color color) {
        Bitboard k = board.pieces(color, KING);
        return k ? BitboardUtils::lsb(k) : -1;
    }

    // per-side features into a struct (color-agnostic keys)
    struct Side {
        int pawns, knights, bishops, rooks, queens;
        int king_edge_dist, bishop_color, king_dist_own_pawn;
        int doubled, isolated, passed_count, connected_passers;
        int rook_on_semiopen_file, rook_dist_enemy_passer;
        int pawn_steps, own_king_dist_promo, enemy_king_dist_promo, enemy_king_outside_square;
    };

    static Side side(const B& board, Color color) {
        Side f{};
        f.pawns   = BitboardUtils::countBits(board.pieces(color, PAWN));
        f.knights = BitboardUtils::countBits(board.pieces(color, KNIGHT));
        f.bishops = BitboardUtils::countBits(board.pieces(color, BISHOP));
        f.rooks   = BitboardUtils::countBits(board.pieces(color, ROOK));
        f.queens  = BitboardUtils::countBits(board.pieces(color, QUEEN));

        int ownK = kingSq(board, color);
        int enemyK = kingSq(board, ~color);
        f.king_edge_dist = (ownK >= 0) ? kingEdgeDist(ownK) : 0;
        f.bishop_color   = bishopColor(board, color);

        // king distance to nearest own pawn (any pawn)
        Bitboard pawns = board.pieces(color, PAWN);
        if (ownK >= 0 && pawns) {
            int best = 99; Bitboard p = pawns;
            while (p) { int s = BitboardUtils::popLsb(p); int d = chebyshev(ownK, s); if (d < best) best = d; }
            f.king_dist_own_pawn = best;
        } else f.king_dist_own_pawn = 0;

        pawnStructure(board, color, f.doubled, f.isolated);

        Bitboard passers = passedPawns(board, color);
        f.passed_count = BitboardUtils::countBits(passers);
        f.connected_passers = connectedPassers(passers);

        // rook features (trimmed keepers)
        f.rook_on_semiopen_file = 0;
        f.rook_dist_enemy_passer = 0;
        Bitboard rooks = board.pieces(color, ROOK);
        if (rooks) {
            // semi-open = a rook on a file with no OWN pawn
            int ownPawnFile = 0;
            { Bitboard p = pawns; while (p) ownPawnFile |= (1 << sqFile(BitboardUtils::popLsb(p))); }
            Bitboard rr = rooks;
            while (rr) { int rs = BitboardUtils::popLsb(rr);
                if (!(ownPawnFile & (1 << sqFile(rs)))) { f.rook_on_semiopen_file = 1; break; } }
            // distance from nearest rook to most-advanced enemy passer
            Bitboard ep = passedPawns(board, ~color);
            if (ep) {
                int advEnemy = -1; Bitboard e = ep;
                if (~color == WHITE) { int best=-1; while(e){int s=BitboardUtils::popLsb(e); if(sqRank(s)>best){best=sqRank(s);advEnemy=s;}} }
                else { int best=99; while(e){int s=BitboardUtils::popLsb(e); if(sqRank(s)<best){best=sqRank(s);advEnemy=s;}} }
                int best = 99; Bitboard r2 = rooks;
                while (r2) { int s = BitboardUtils::popLsb(r2); int d = chebyshev(s, advEnemy); if (d < best) best = d; }
                f.rook_dist_enemy_passer = best;
            }
        }

        // most-advanced passer geometry
        if (!passers) {
            f.pawn_steps = f.own_king_dist_promo = f.enemy_king_dist_promo = f.enemy_king_outside_square = 0;
            return f;
        }
        int adv = -1, steps = 0, promo = -1;
        if (color == WHITE) {
            int best = -1; Bitboard p = passers;
            while (p) { int s = BitboardUtils::popLsb(p); if (sqRank(s) > best) { best = sqRank(s); adv = s; } }
            steps = 7 - sqRank(adv);
            promo = sqFile(adv) + 7*8;           // file, rank 7
        } else {
            int best = 99; Bitboard p = passers;
            while (p) { int s = BitboardUtils::popLsb(p); if (sqRank(s) < best) { best = sqRank(s); adv = s; } }
            steps = sqRank(adv);
            promo = sqFile(adv) + 0*8;            // file, rank 0
        }
        f.pawn_steps = steps;
        f.own_king_dist_promo = (ownK >= 0) ? chebyshev(ownK, promo) : 0;
        int ekd = (enemyK >= 0) ? chebyshev(enemyK, promo) : 0;
        f.enemy_king_dist_promo = ekd;
        if (enemyK >= 0) {
            int effective = (board.sideToMove() == color) ? steps - 1 : steps;
            f.enemy_king_outside_square = (ekd > effective) ? 1 : 0;
        } else f.enemy_king_outside_square = 0;
        return f;
    }

    // assemble the 40-vector in EG_FEATURE_ORDER ----
    static void extract(const B& board, float* out) {
        Color stm = board.sideToMove();
        Side s = side(board, stm);
        Side o = side(board, ~stm);

        // symmetric king-vs-king geometry
        int wk = kingSq(board, WHITE), bk = kingSq(board, BLACK);
        int king_dist = 0, collinear = 0, gap_parity = 0;
        if (wk >= 0 && bk >= 0) {
            king_dist = chebyshev(wk, bk);
            collinear = (sqFile(wk) == sqFile(bk) || sqRank(wk) == sqRank(bk)) ? 1 : 0;
            gap_parity = (collinear && ((king_dist - 1) % 2 == 1)) ? 1 : 0;
        }
        int cs = bishopColor(board, WHITE), co = bishopColor(board, BLACK);
        int opp_bishops = (cs != -1 && co != -1 && cs != co) ? 1 : 0;

        int i = 0;
        out[i++] = (float)king_dist;
        out[i++] = (float)collinear;
        out[i++] = (float)gap_parity;
        out[i++] = (float)opp_bishops;
        // per-side stm then opp, in _SIDE_KEYS order
        #define PAIR(field) do { out[i++] = (float)s.field; out[i++] = (float)o.field; } while(0)
        PAIR(pawns); PAIR(knights); PAIR(bishops); PAIR(rooks); PAIR(queens);
        PAIR(king_edge_dist); PAIR(bishop_color);
        PAIR(king_dist_own_pawn);
        PAIR(doubled); PAIR(isolated); PAIR(passed_count); PAIR(connected_passers);
        PAIR(rook_on_semiopen_file); PAIR(rook_dist_enemy_passer);
        PAIR(pawn_steps); PAIR(own_king_dist_promo); PAIR(enemy_king_dist_promo);
        PAIR(enemy_king_outside_square);
        #undef PAIR
    }
};