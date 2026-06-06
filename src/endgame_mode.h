#pragma once
namespace Endgame {
    enum class Mode { Classic, MLPBlend, MLPReplace, Syzygy };
    inline Mode g_mode = Mode::Classic;   // C++17 inline var: one definition, default OFF
}