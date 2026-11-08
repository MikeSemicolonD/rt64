//
// RT64
//

#pragma once

#include "rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RDPParams {
        float4 primColor;
        float2 primLOD;
        float2 primDepth;
        float4 envColor;
        float4 fogColor;
        float4 blendColor;
        float3 keyCenter;
        float3 keyScale;
        int convertK[6];
        // RS64 fog on a draw the game leaves unfogged: screen-depth mul and offset, fog tail end, on (w). See rs64_fog_tail_end.
        float4 modelFog;
        float4 modelFogColor;
    };
#ifdef HLSL_CPU
};
#endif