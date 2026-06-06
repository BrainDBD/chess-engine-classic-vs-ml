#include "endgame_net_weights.h"
#include <cmath>
#include <cstdio>

static float eg_forward(const float* feat) {
    float x[EG_NET_IN];
    for (int i = 0; i < EG_NET_IN; ++i)
        x[i] = (feat[i] - EG_FEAT_MEAN[i]) / EG_FEAT_STD[i];
    float h1[EG_NET_H1];
    for (int j = 0; j < EG_NET_H1; ++j) {
        float s = EG_B1[j];
        for (int i = 0; i < EG_NET_IN; ++i) s += EG_W1[j][i] * x[i];
        h1[j] = s > 0.f ? s : 0.f;                 // ReLU
    }
    float h2[EG_NET_H2];
    for (int j = 0; j < EG_NET_H2; ++j) {
        float s = EG_B2[j];
        for (int i = 0; i < EG_NET_H1; ++i) s += EG_W2[j][i] * h1[i];
        h2[j] = s > 0.f ? s : 0.f;
    }
    float s = EG_B3;
    for (int i = 0; i < EG_NET_H2; ++i) s += EG_W3[i] * h2[i];
    return 1.f / (1.f + std::exp(-s));             // sigmoid
}

int main() {
    // read 40 floats per line from stdin, print forward() for each
    float f[EG_NET_IN];
    while (true) {
        for (int i = 0; i < EG_NET_IN; ++i)
            if (scanf("%f", &f[i]) != 1) return 0;
        printf("%.9f\n", eg_forward(f));
    }
}