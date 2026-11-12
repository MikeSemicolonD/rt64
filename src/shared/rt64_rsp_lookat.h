//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#define RSP_LOOKAT_INDEX_ENABLED 0x1
#define RSP_LOOKAT_INDEX_LINEAR 0x2
#define RSP_LOOKAT_INDEX_SHIFT 2

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RSPLookAt {
        float3 x;
        float xPad;
        float3 y;
        float yPad;
    };
#ifdef HLSL_CPU
    static_assert(sizeof(RSPLookAt) == 32);
};
#endif