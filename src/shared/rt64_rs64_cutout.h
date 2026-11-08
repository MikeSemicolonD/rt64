// RT64 - Cutout draw record shared by C++ and the traced shaders

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // Mirrors rs64lights::CutoutEntry: first triangle of the draw in the cutout BLAS, its call index, RDP tile and flags (1 translucent, 2 sky dome, bits 2-7 average vertex alpha, 8-31 average vertex rgb8).
    struct RS64CutoutDraw {
        uint firstTriangle;
        uint instanceIndex;
        uint tileIndex;
        uint flags;
    };
#ifdef HLSL_CPU
};
#endif
