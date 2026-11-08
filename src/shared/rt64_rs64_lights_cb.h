//
// RT64
//

#pragma once

#include "shared/rt64_hlsl.h"

#define RS64_MAX_LIGHTS 64
// debugMode of the second, alpha-blended draw that shows the glowing panel faces.
#define RS64_LIGHTS_PANEL_FACES 100

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
        // x: specular strength, y: specular exponent.
        float4 specParams;
        float4 lightPosRadius[RS64_MAX_LIGHTS];
        float4 lightColor[RS64_MAX_LIGHTS];
        float4 lightParams[RS64_MAX_LIGHTS];
        // xyz: the spot beam direction (view space); lightParams.zw hold its max / falloff cosines (RT64's PointLight convention).
        float4 lightSpot[RS64_MAX_LIGHTS];
        // xyz: a glowing panel's half-width / half-height axes (view space), facing along lightSpot.xyz; zero = a point light.
        float4 lightRectU[RS64_MAX_LIGHTS];
        float4 lightRectV[RS64_MAX_LIGHTS];
    };
#ifdef HLSL_CPU
};
#endif
