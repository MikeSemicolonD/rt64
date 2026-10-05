//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // Mirrors rs64lights::CutoutEntry: first triangle of the draw in the cutout BLAS, its call index and RDP tile.
    struct RS64CutoutDraw {
        uint firstTriangle;
        uint instanceIndex;
        uint tileIndex;
        uint pad;
    };
#ifdef HLSL_CPU
};
#endif
