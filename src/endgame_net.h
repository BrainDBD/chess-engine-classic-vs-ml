#pragma once
#include "board.h"
#include "bitboard.h"
#include "types.h"
#include "endgame_net_features.h"
#include "endgame_net_weights.h"
#include <cmath>
#include <algorithm>

namespace EndgameNet {

    inline constexpr int MAX_PIECES = 6;

    // win-probability p -> centipawns via the logit (inverse sigmoid):
    // cp = K * ln(p / (1-p)),  p=0.5 -> 0cp.
    // K sets the cp scale; ~200 keeps a 0.75 win ~ +220cp, comparable to the
    // classical eval (~100cp/pawn) so search margins still behave.
    inline constexpr float K_LOGIT = 200.0f;
    // Clamp well below MATE_THRESHOLD so the net never masquerades as a mate;
    // the search still finds real mates via its own mate scoring at the leaves.
    inline constexpr int CP_CAP = 2000;

    inline float forward(const float* feat) {
        float x[EG_NET_IN];
        for (int i = 0; i < EG_NET_IN; ++i)
            x[i] = (feat[i] - EG_FEAT_MEAN[i]) / EG_FEAT_STD[i];
        float h1[EG_NET_H1];
        for (int j = 0; j < EG_NET_H1; ++j) {
            float s = EG_B1[j];
            for (int i = 0; i < EG_NET_IN; ++i) s += EG_W1[j][i] * x[i];
            h1[j] = s > 0.f ? s : 0.f;
        }
        float h2[EG_NET_H2];
        for (int j = 0; j < EG_NET_H2; ++j) {
            float s = EG_B2[j];
            for (int i = 0; i < EG_NET_H1; ++i) s += EG_W2[j][i] * h1[i];
            h2[j] = s > 0.f ? s : 0.f;
        }
        float s = EG_B3;
        for (int i = 0; i < EG_NET_H2; ++i) s += EG_W3[i] * h2[i];
        return 1.f / (1.f + std::exp(-s));        // win probability, STM POV
    }

    // STM-POV centipawn evaluation. Drop-in below MAX_PIECES.
    inline int evaluate(const Board& board) {
        float feat[EG_NET_IN];
        EndgameFeatures<Board>::extract(board, feat);
        float p = forward(feat);
        p = std::clamp(p, 1e-4f, 1.0f - 1e-4f);
        float cp = K_LOGIT * std::log(p / (1.0f - p));
        int score = (int)std::lround(cp);
        return std::clamp(score, -CP_CAP, CP_CAP);
    }
}