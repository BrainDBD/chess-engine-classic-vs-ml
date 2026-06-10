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
    inline constexpr float T_DRAW = 0.35f;   // o0 cutoff: P(outcome >= draw)
    inline constexpr float T_WIN  = 0.65f;   // o1 cutoff: P(outcome >= win)

    // Returns the two threshold probabilities {P(>=draw), P(>=win)} into out[2]
    inline void forward(const float* feat, float out[2]) {
        float x[EG_NET_IN];
        for (int i = 0; i < EG_NET_IN; ++i)
            x[i] = (feat[i] - EG_FEAT_MEAN[i]) / EG_FEAT_STD[i];
        float h1[EG_NET_H1];
        for (int j = 0; j < EG_NET_H1; ++j) {
            float s = EG_B1[j];
            for (int i = 0; i < EG_NET_IN; ++i) s += EG_W1[j][i] * x[i];
            h1[j] = std::max(0.0f, s);
        }
        float h2[EG_NET_H2];
        for (int j = 0; j < EG_NET_H2; ++j) {
            float s = EG_B2[j];
            for (int i = 0; i < EG_NET_H1; ++i) s += EG_W2[j][i] * h1[i];
            h2[j] = std::max(0.0f, s);
        }
        // shared linear -> single logit (no bias; ordinal biases carry the offset)
        float logit = 0.0f;
        for (int i = 0; i < EG_NET_H2; ++i) logit += EG_W_SHARED[i] * h2[i];
        out[0] = 1.f / (1.f + std::exp(-(logit + EG_THRESH_BIAS[0])));  // P(>=draw)
        out[1] = 1.f / (1.f + std::exp(-(logit + EG_THRESH_BIAS[1])));  // P(>=win)
    }

    // Discrete WDL verdict, STM POV: +1 win / 0 draw / -1 loss
    // Count thresholds passed at 0.5, then center to {-1,0,+1}
    inline int wdlVerdict(const Board& board) {
        float feat[EG_NET_IN];
        EndgameFeatures<Board>::extract(board, feat);
        float o[2];
        forward(feat, o);
        int passed = (o[0] > T_DRAW ? 1 : 0) + (o[1] > T_WIN ? 1 : 0); // 0,1,2
        return passed - 1;                                            // -1,0,+1
    }
}

// Endgame net: CORAL ordinal head over geometry features.
// Head: shared trunk -> single logit; two ordered biases b0 > b1 give
//   o0 = sigmoid(logit + b0) = P(outcome >= draw)
//   o1 = sigmoid(logit + b1) = P(outcome >= win)
// Verdict = number of thresholds passed at 0.5: 0=loss, 1=draw, 2=win.
// Ordinality is structural (b0 > b1 => o0 >= o1), so no draw-band is needed.