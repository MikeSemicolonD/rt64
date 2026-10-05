//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    struct RS64FogShaftsCB {
        float4x4 viewFromScreen;
        float4x4 viewProj;
        float4 viewport;
        float4 sunDir;
        float4 fogColor;
        uint debugMode;
        uint steps;
        float strength;
        float fogMul;
        float fogOffset;
        float tMin;
        float tMax;
        float terrainTMinScale;
        float skyDepth;
        float visBias;
        uint cutoutDrawCount;
        float pad2;
    };
#ifdef HLSL_CPU
};
#endif
