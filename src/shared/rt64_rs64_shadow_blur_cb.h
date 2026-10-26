//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RS64ShadowBlurCB {
        float4x4 viewFromScreen;
        float4 viewport;
        // Shadow strength, depth tolerance (fraction of view distance), debug (1 = grey mask), max blur radius in pixels.
        float4 params;
        // Pixel step per tap: (1, 0) horizontal, (0, 1) vertical.
        float2 texelDir;
        float2 pad;
        // AO strength, debug view (5 shadow, 6 AO, 7 indirect, 8 reflection; 0 compose), unused x2.
        float4 aoParams;
        // Temporal pass: this frame's view position -> the previous frame's screen (final ndc x, y, depth), and -> the previous frame's view.
        float4x4 prevScreenFromView;
        float4x4 prevViewFromView;
        float4 prevViewport;
        // History valid (0/1), distance tolerance (fraction), weight of the new frame, unused.
        float4 temporal;
    };
#ifdef HLSL_CPU
};
#endif
