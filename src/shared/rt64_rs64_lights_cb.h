//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#define RS64_MAX_LIGHTS 32

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RS64LightsCB {
        float4x4 viewFromScreen;
        float4 viewport;
        uint lightCount;
        uint debugMode;
        float gain;
        float wrap;
        float debugRange;
        float tMin;
        float normalBias;
        float terrainTMinScale;
        float4 shadowParams;
        float4 lightPosRadius[RS64_MAX_LIGHTS];
        float4 lightColor[RS64_MAX_LIGHTS];
        float4 lightParams[RS64_MAX_LIGHTS];
    };
#ifdef HLSL_CPU
};
#endif
