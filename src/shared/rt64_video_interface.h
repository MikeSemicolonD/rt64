//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct VideoInterfaceCB {
        float2 videoResolution;
        float2 textureResolution;
        float gamma;
        // Packed VI status flags: bits 0-1 = aaMode (0..3),
        //                         bit  2 = divotEnable,
        //                         bit  3 = ditherFilter,
        //                         bit  4 = gammaDitherEnable.
        // RT64 historically applied only `gamma` from the VI status; this
        // field lets the VI shader honor the rest of the post-processing
        // chain that real N64 hardware applies (the "N64 glow / soft" look).
        uint viFlags;
    };
#ifdef HLSL_CPU
};
#endif